"""Coherent decisions, prefix-owned dispatch, bitwise parity and atomic caps."""
import ast
from fractions import Fraction as F
import gzip
import os
from pathlib import Path
import struct
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from direct_sympy_coherent_paths import CoherentPaths
from direct_sympy_conversions_test import cpp
from direct_sympy_checkpoint import CheckpointStrings
from direct_sympy_strings import Domain,StringCompiler,syntax
from direct_sympy_streaming_literals import StreamingLiterals,streaming_literals


def fixture(definitions):
    compiler=StringCompiler();domains={'X1':Domain(F(-1),F(1),-24,False),'X2':Domain(F(-1),F(1),-24,False)}
    # Pure, already stabilized mathematical strings; no numeric primitive
    # remains. This test fixture has no checkpoint or conversion session.
    registry=object.__new__(StreamingLiterals)
    registry.model=SimpleNamespace(compiler=compiler,domains=domains)
    registry.context=compiler.context(domains)
    registry.definitions=[compiler.stabilize(text,domains) for text in definitions]
    return registry


class Lazy(ast.NodeTransformer):
    def visit_Call(self,node):
        node=self.generic_visit(node)
        if node.func.id=='Piecewise':
            result=ast.Constant(value=0)
            for pair in reversed(node.args):result=ast.IfExp(test=pair.elts[1],body=pair.elts[0],orelse=result)
            return result
        if node.func.id in ('And','Or'):
            if len(node.args)==1:return node.args[0]
            if not node.args:return ast.Constant(value=node.func.id=='And')
            return ast.BoolOp(op=ast.And() if node.func.id=='And' else ast.Or(),values=node.args)
        if node.func.id=='Not':return ast.UnaryOp(op=ast.Not(),operand=node.args[0])
        return node


def evaluate(text,values):
    tree=ast.fix_missing_locations(ast.Expression(Lazy().visit(syntax(text))))
    return eval(compile(tree,'<actual-flat-file>','eval'),{'__builtins__':{}},values)


class CoherentPathTests(unittest.TestCase):
    def test_same_decision_reused_by_dependencies_has_two_paths_not_four(self):
        registry=fixture(['Piecewise((X1 * 0.5, X1 > 0.0), (X1 * 2.0, True))',
            'Piecewise((CompileValue0() + 1.0, X1 > 0.0), (CompileValue0() - 1.0, True))'])
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'flat.expr';plan=CoherentPaths(registry)
            report=plan.write(path,'CompileValue1()',max_characters=65536)
            self.assertEqual(report['paths'],2)
            text=path.read_text();self.assertEqual(text.count('Piecewise('),1)
            self.assertNotIn('CompileValue',text)
            for x in [-1.0,-2**-24,-0.0,0.0,2**-24,1.0]:
                expected=x*0.5+1.0 if x>0 else x*2.0-1.0
                self.assertEqual(struct.pack('d',evaluate(text,{'X1':x,'X2':0.0})),struct.pack('d',expected))
            self.assertTrue(all(event[2:4]==('factor','simplify') for event in registry.model.compiler.events))

    def test_guard_owns_prefix_before_narrower_later_branch_and_is_lazy(self):
        registry=fixture(['Piecewise((Piecewise((1.0 / X1, X1 > 0.5), (-1.0 / X1, True)), X1 != 0.0), (-0.0, True))'])
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'flat.expr'
            report=CoherentPaths(registry).write(path,'CompileValue0()',max_characters=65536)
            self.assertEqual(report['paths'],3)
            text=path.read_text();self.assertEqual(text.count('Piecewise('),1)
            for x in [-1.0,-2**-24,-0.0,0.0,2**-24,0.5,1.0]:
                expected=(1/x if x>0.5 else -1/x) if x!=0 else -0.0
                self.assertEqual(struct.pack('d',evaluate(text,{'X1':x,'X2':0.0})),struct.pack('d',expected))

    def test_independent_decisions_remain_independent_and_logical_conflicts_are_rejected(self):
        registry=fixture(['Piecewise((X1, X1>0.0), (-X1, True))',
            'Piecewise((CompileValue0()+X2, X2>0.0), (CompileValue0()-X2, True))'])
        plan=CoherentPaths(registry)
        self.assertEqual(len(list(plan.arms('CompileValue1()'))),4)
        a,b=syntax('X1>0.0'),syntax('X2>0.0')
        compound=syntax('And(X1>0.0,X2>0.0)')
        facts=plan.assume(compound,False,())
        facts=plan.assume(a,True,facts)
        self.assertFalse(plan.facts.truth(b,facts))
        self.assertIsNone(plan.assume(b,True,facts))
        facts=plan.assume(syntax('Not(Or(X1>0.0,X2>0.0))'),True,())
        self.assertFalse(plan.facts.truth(a,facts))
        self.assertFalse(plan.facts.truth(b,facts))
        self.assertIsNone(plan.assume(a,True,facts))
        self.assertIsNone(plan.assume(syntax('Or()'),True,()))
        self.assertIsNone(plan.assume(syntax('And()'),False,()))

    def test_ordered_logical_guards_do_not_evaluate_unsafe_later_operands(self):
        for logical,expected in [
            ('And(X1 != 0.0, 1.0 / X1 > 0.5)',lambda x:x!=0 and 1/x>0.5),
            ('Or(X1 == 0.0, 1.0 / X1 > 0.5)',lambda x:x==0 or 1/x>0.5),
        ]:
            registry=fixture(['Piecewise((2.0, '+logical+'), (-3.0, True))'])
            with tempfile.TemporaryDirectory() as directory:
                path=Path(directory)/'flat.expr'
                CoherentPaths(registry).write(path,'CompileValue0()',max_characters=65536)
                text=path.read_text()
                for x in [-1.0,-0.0,0.0,0.5,1.0]:
                    self.assertEqual(evaluate(text,{'X1':x,'X2':0.0}),2.0 if expected(x) else -3.0)

    def test_candidate_cost_uses_selected_path_and_restores_callback(self):
        registry=fixture(['Piecewise((X1, X1>0.0), (X1*2.0, True))'])
        compiler=registry.model.compiler;original=compiler.expression_size
        stabilize=compiler.stabilize;costs=[]
        def observe(expression,*args,**kwargs):
            if expression=='(CompileValue0())':
                costs.append(compiler.expression_size('CompileValue0()'))
            return stabilize(expression,*args,**kwargs)
        with patch.object(compiler,'stabilize',side_effect=observe):
            arms=list(CoherentPaths(registry).arms('CompileValue0()'))
        self.assertEqual(costs,[arm.view.size('CompileValue0()') for arm in arms])
        self.assertIs(compiler.expression_size,original)
        with patch.object(compiler,'stabilize',side_effect=ValueError('failed pass')):
            with self.assertRaisesRegex(ValueError,'failed pass'):
                list(CoherentPaths(registry).arms('CompileValue0()'))
        self.assertIs(compiler.expression_size,original)

    def test_budget_failure_keeps_existing_artifact_and_compressed_readback_matches(self):
        registry=fixture(['Piecewise((X1, X1>0.0), (-X1, True))'])
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path=root/'flat.expr';path.write_text('existing')
            with self.assertRaisesRegex(ValueError,'Path budget'):
                CoherentPaths(registry,max_paths=1).write(path,'CompileValue0()',max_characters=65536)
            self.assertEqual(path.read_text(),'existing')
            with self.assertRaisesRegex(ValueError,'artifact budget'):
                CoherentPaths(registry).write(path,'CompileValue0()',max_characters=10)
            self.assertEqual(path.read_text(),'existing')
            CoherentPaths(registry).write(path,'CompileValue0()',max_characters=65536)
            zipped=root/'flat.expr.gz'
            CoherentPaths(registry).write(zipped,'CompileValue0()',max_characters=65536,compressed=True)
            self.assertEqual(gzip.decompress(zipped.read_bytes()),path.read_bytes())
            self.assertEqual(set(p.name for p in root.iterdir()),{'flat.expr','flat.expr.gz'})

    @unittest.skipUnless(os.environ.get('LLM_INNER_DIRECT_JSON_CHECKPOINT'),'checkpoint not configured')
    def test_actual_checkpoint_normalization_flat_file_has_native_parity(self):
        checkpoint=os.environ['LLM_INNER_DIRECT_JSON_CHECKPOINT']
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            path=Path(os.environ.get('LLM_INNER_DIRECT_FLAT_PREFIX_OUTPUT',str(root/'pre-coordinate.expr')))
            with CheckpointStrings(checkpoint,StringCompiler(max_characters=65536)) as model:
                with streaming_literals(model) as registry:
                    value=model.norm('model.layers.0.pre','model.layers.0.input_layernorm.weight',0,lambda i:f'X{i+1}')
                    epsilon=model.config['rms_norm_eps'];gamma=model.weight('model.layers.0.input_layernorm.weight',0)
                    report=CoherentPaths(registry,max_paths=8).write(path,value,max_characters=65536)
                    self.assertEqual(report['paths'],2)
            text=path.read_text();self.assertEqual(text.count('Piecewise('),1)
            self.assertNotIn('CompileValue',text)
            source=root/'native.cpp';binary=root/'native'
            source.write_text('''#include <cstdint>
#include <cstring>
#include <cmath>
#include <cfenv>
#include <cstdio>
#include <initializer_list>
template<class T,class U>T word(U value){T result;static_assert(sizeof(T)==sizeof(U));std::memcpy(&result,&value,sizeof(result));return result;}
double candidate(double X1,double X2){return '''+cpp(syntax(text))+''';}
int main(){if(std::fesetround(FE_TONEAREST))return 2;unsigned cases=0;
for(unsigned bits=0;bits<65536;bits++){if((bits&0x7c00)==0x7c00)continue;
double x=word<_Float16>(uint16_t(bits));
for(uint16_t other:{uint16_t(0),uint16_t(0x8000),uint16_t(1),uint16_t(0x8001),uint16_t(0x03ff),uint16_t(0x83ff),uint16_t(0x0400),uint16_t(0x8400),uint16_t(0x3555),uint16_t(0xb555),uint16_t(0x3c01),uint16_t(0xbc01),uint16_t(0x7bff),uint16_t(0xfbff)}){
double y=word<_Float16>(other);float sum=float(float(x*x)+float(y*y));
float mean=float(float(sum/2.0f)+'''+repr(epsilon)+'''f);float inverse=float(1.0f/std::sqrt(mean));
double expected=static_cast<_Float16>(float(static_cast<_Float16>(float(float(x)*inverse)))*'''+gamma+'''f);cases++;
if(word<uint64_t>(candidate(x,y))!=word<uint64_t>(expected))return 1;
}}std::printf("Flat checkpoint normalization: cases=%u mismatches=0\\n",cases);}
''')
            subprocess.run(['clang++','-O3','-ffp-contract=off','-std=c++17',str(source),'-o',str(binary)],check=True,capture_output=True,text=True)
            result=subprocess.run([str(binary)],check=True,capture_output=True,text=True,timeout=120)
            self.assertIn('cases=888832 mismatches=0',result.stdout);print(result.stdout,end='')


if __name__=='__main__':unittest.main()
