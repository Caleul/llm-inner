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

from direct_sympy_coherent_paths import CoherentPaths,UnreachableNumericPath,OrderedGuardPruner,Guard,PathArm,LiteralView
from direct_sympy_conversions import FiniteSource,ConversionSession
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


def evaluate(text,values,functions=None):
    tree=ast.fix_missing_locations(ast.Expression(Lazy().visit(syntax(text))))
    return eval(compile(tree,'<actual-flat-file>','eval'),{'__builtins__':{},**(functions or {})},values)


class CoherentPathTests(unittest.TestCase):
    def test_selected_pure_scalar_admits_next_constant_cell_without_reopening(self):
        registry=fixture(['Piecewise((1.0,X1>0.0),(2.0,True))'])
        source=ConversionSession(registry.model.compiler,registry.model.domains)
        marker=source.key(syntax('CompileValue0()'))
        source.completed[marker]=FiniteSource(1,2,0);source.f32_values.add(marker)
        recipe='R32(1.0/R32(sqrt(CompileValue0())))'
        closed=source.close(recipe)
        # Reproduce word closure embedding an earlier producer's payload.
        # The selected tree no longer visits its marker, but the original
        # numerical recipe still needs its current-context certificate.
        closed=closed.replace('CompileValue0()','('+registry.definitions[0]+')')
        self.assertNotIn('CompileValue0()',closed)
        registry.definitions.append(closed)
        registry.model.conversions=source
        registry.definition_proofs=[(FiniteSource(1,2,0),'f32',True,None),(FiniteSource(.7,1,-24),'f32',True,None)]
        registry.definition_recipes={1:recipe}
        with tempfile.TemporaryDirectory()as directory:
            path=Path(directory)/'constant.expr';plan=CoherentPaths(registry)
            plan.write(path,'CompileValue1()',max_characters=65536)
            text=path.read_text();self.assertLess(len(text),128)
            self.assertNotIn('Float64',text);self.assertNotIn('CompileValue',text)
            expected=struct.unpack('f',struct.pack('f',1.0/struct.unpack('f',struct.pack('f',2**.5))[0]))[0]
            for x in (-1.,-0.,0.,1.):
                self.assertEqual(struct.pack('d',evaluate(text,{'X1':x,'X2':0.})),struct.pack('d',1. if x>0 else expected))

    def test_numerical_comparisons_use_only_prefix_bounds_and_keep_overlap(self):
        registry=fixture(['X1'])
        registry.definition_proofs=[(FiniteSource(-1,1,-24),'half',False,None)]
        plan=CoherentPaths(registry)
        self.assertIs(plan.condition_truth(syntax('CompileValue0() >= -1.0'),()),True)
        self.assertIs(plan.condition_truth(syntax('CompileValue0() > 1.0'),()),False)
        self.assertIsNone(plan.condition_truth(syntax('CompileValue0() < 0.0'),()))
        self.assertIsNone(plan.condition_truth(syntax('CompileValue0() == 0.0'),()))
        self.assertIsNone(plan.condition_truth(syntax('1.0/X1 < 0.0'),()))
        positive=plan.assume(syntax('X1>0.5'),True,())
        self.assertIs(plan.condition_truth(syntax('X1 <= 0.0'),positive),False)
        self.assertIsNone(plan.condition_truth(syntax('X1 <= 0.0'),()))
        self.assertEqual(registry.model.domains['X1'].minimum,F(-1))

    def test_certified_empty_numeric_branch_is_pruned_but_other_errors_remain_fatal(self):
        mask=0x7fffffffffffffff
        threshold=struct.unpack('>Q',struct.pack('>d',2.0))[0]
        registry=fixture(['X1 + 0.0',f'Piecewise((CompileValue0(), U64And(Bits64(CompileValue0()), {mask}) < {threshold}), (CompileValue0() + 2.0, True))'])
        registry.definition_proofs=[(FiniteSource(-1,1,-24),'half',False,None),(None,None,False,None)]
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'flat.expr';plan=CoherentPaths(registry)
            report=plan.write(path,'CompileValue1()',max_characters=65536)
            self.assertEqual(report['paths'],1)
            self.assertEqual(plan.stats['contradictions'],0)
            self.assertEqual(plan.stats['splitContexts'],0)
            self.assertGreater(plan.stats['numericGuardImplications'],0)
            functions={'Bits64':lambda x:struct.unpack('>Q',struct.pack('>d',x))[0],'U64And':lambda x,y:x&y}
            for x in (-1.0,-2**-24,-0.0,0.0,2**-24,1.0):
                self.assertEqual(struct.pack('d',evaluate(path.read_text(),{'X1':x,'X2':0.0},functions)),struct.pack('d',x+0.0))
            original=path.read_bytes()
            with patch.object(CoherentPaths,'literal',side_effect=ValueError('Bad certificate')):
                with self.assertRaisesRegex(ValueError,'Bad certificate'):CoherentPaths(registry).write(path,'CompileValue1()',max_characters=65536)
            self.assertEqual(path.read_bytes(),original)
            with patch.object(CoherentPaths,'literal',side_effect=UnreachableNumericPath('empty')):
                with self.assertRaisesRegex(ValueError,'No reachable numerical path'):CoherentPaths(registry).write(path,'CompileValue1()',max_characters=65536)
            self.assertEqual(path.read_bytes(),original)

    def test_dispatch_prefix_removes_only_implied_guards_and_retains_signed_zero(self):
        registry=fixture(['Piecewise((11.0,And(X1>0.0,X2>0.0)),(13.0,X1>0.0),(-0.0,True))'])
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'flat.expr';plan=CoherentPaths(registry)
            plan.write(path,'CompileValue0()',max_characters=65536)
            self.assertGreater(plan.stats['orderedGuardEliminations'],0)
            actual=path.read_text()
            for x in (-1.,-0.,0.,1.):
                for y in (-1.,-0.,0.,1.):
                    expected=11. if x>0 and y>0 else 13. if x>0 else -0.
                    self.assertEqual(struct.pack('d',evaluate(actual,{'X1':x,'X2':y})),struct.pack('d',expected))
        registry=fixture(['Piecewise((1.0,X1==0.0),(2.0,And(X1!=0.0,1.0/X1>0.5)),(-0.0,True))'])
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'lazy.expr';plan=CoherentPaths(registry)
            plan.write(path,'CompileValue0()',max_characters=65536)
            for x in (-1.,-0.,0.,.5,1.):
                expected=1. if x==0 else 2. if 1/x>.5 else -0.
                self.assertEqual(struct.pack('d',evaluate(path.read_text(),{'X1':x,'X2':0.})),struct.pack('d',expected))

    def test_guard_identity_checks_actual_bytes_and_budget_falls_back_without_dropping_paths(self):
        registry=fixture([]);view=LiteralView(registry,[])
        def stats():return {'guardProofPasses':0,'guardProofBudgetStops':0,'orderedGuardEliminations':0}
        class CollidingDigest:
            def update(self,value):pass
            def digest(self):return b'same digest'
        a=Guard(view,'X1>0.0',True);b=Guard(view,'X2>0.0',True)
        with patch('direct_sympy_coherent_paths.hashlib.sha256',return_value=CollidingDigest()):
            pruner=OrderedGuardPruner(stats())
            self.assertNotEqual(pruner.atom(a),pruner.atom(b))
            self.assertEqual(pruner.atom(a),pruner.atom(Guard(view,a.expression,False)))
        own=stats();pruner=OrderedGuardPruner(own,max_atoms=1)
        first=PathArm(view,'1.0',(a,));pruner.prune(first)
        next_arm=PathArm(view,'2.0',(Guard(view,a.expression,False),b))
        self.assertIs(pruner.prune(next_arm),next_arm)
        self.assertEqual(own['orderedGuardEliminations'],0)
        self.assertEqual(own['guardProofBudgetStops'],1)
        self.assertIs(pruner.prune(first),first)

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

    def test_selected_numeric_root_is_reduced_without_reusing_other_arm_payload(self):
        registry=fixture([
            'Piecewise((Float64(U64Or(Bits64(X1),1024)),X1>0.0), (Float64(U64Or(Bits64(X1),2048)),True))',
            'Bits64(CompileValue0())'])
        compiler=registry.model.compiler;context=compiler.context(registry.model.domains)
        # Simulate a root annotation referring to an original whole value.
        # It must never determine a selected arm's replacement payload.
        compiler.register_completed_region('CompileValue0()',registry.model.domains)
        compiler._region_roots[(context,'CompileValue0()')]=('Float64',64)
        compiler._region_word_payloads[(context,'CompileValue0()')]='U64Or(Bits64(X1),4096)'
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'flat.expr'
            CoherentPaths(registry).write(path,'CompileValue1()',max_characters=65536)
            text=path.read_text()
            self.assertNotIn('Float64',text)
            self.assertNotIn('4096',text)
            functions={'Bits64':lambda x:struct.unpack('Q',struct.pack('d',x))[0],
                'U64Or':lambda x,y:x|y}
            for x in [-1.0,-2**-24,-0.0,0.0,2**-24,1.0]:
                self.assertEqual(evaluate(text,{'X1':x,'X2':0.0},functions),
                    functions['Bits64'](x)|(1024 if x>0 else 2048))

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

    def test_selected_numeric_certificates_propagate_without_changing_zero_sibling(self):
        from direct_sympy_conversions import ConversionSession,FiniteSource
        guard='U64And(Bits64(X1),9223372036854775807)<'+str(0x3fb0000000000000)
        registry=fixture(['Piecewise((-0.0,'+guard+'),(X1,True))','CompileValue0()+0.0'])
        registry.model.conversions=ConversionSession(registry.model.compiler,registry.model.domains,input_dtype='f16')
        registry.definition_proofs=[(FiniteSource(-1,1,-24),'half',False,
            (FiniteSource(0,0,-24),FiniteSource(-1,1,-24,2**-4))),
            (FiniteSource(-1,1,-24),'half',True,None)]
        session=registry.model.conversions
        alias=syntax('CompileValue0()');key=session.key(alias)
        session.completed[key]=FiniteSource(-1,1,-24);session.half_values.add(key);session.f32_values.add(key)
        session.converted_regions.add(key)
        recipe='R16(R32(CompileValue0()+5.960464477539063e-08))'
        lowered=session.close(recipe)
        numerical=fixture([registry.definitions[0],lowered])
        numerical.model.conversions=session
        numerical.definition_proofs=[registry.definition_proofs[0],(FiniteSource(-1,1,-24),'half',False,None)]
        numerical.definition_recipes={1:recipe}
        before=registry.model.conversions.completed.copy()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'numeric.expr';plan=CoherentPaths(registry)
            report=plan.write(path,'CompileValue1()',max_characters=65536)
            self.assertEqual(report['paths'],2)
            self.assertGreater(plan.stats['numericArithmeticEliminated'],0)
            text=path.read_text();self.assertNotIn('CompileValue',text)
            arms=list(CoherentPaths(registry).arms('CompileValue1()'))
            self.assertTrue(any(arm.view.definitions[1]=='CompileValue0()' for arm in arms))
            self.assertEqual(registry.model.conversions.completed,before)
            reclosed=Path(directory)/'reclosed.expr';other=CoherentPaths(numerical)
            other.write(reclosed,'CompileValue1()',max_characters=65536)
            self.assertGreater(other.stats['numericRecipeAdmissions'],0)
            self.assertEqual(registry.model.conversions.completed,before)
            self.assertNotIn('R16',reclosed.read_text());self.assertNotIn('R32',reclosed.read_text())
            source=Path(directory)/'numeric.cpp';binary=Path(directory)/'numeric'
            source.write_text('#include <cstdint>\n#include <cstring>\n#include <cmath>\n#include <cstdio>\ntemplate<class T,class U>T word(U x){T y;std::memcpy(&y,&x,sizeof(y));return y;}\n'+'double candidate(double X1,double X2){return '+cpp(syntax(text))+';}\n'+'double reclosed(double X1,double X2){return '+cpp(syntax(reclosed.read_text()))+';}\n'+"""int main(){unsigned cases=0,mismatches=0;for(unsigned bits=0;bits<65536;bits++){if((bits&0x7c00)==0x7c00)continue;double x=word<_Float16>(uint16_t(bits));if(std::abs(x)>1)continue;double expected=(std::abs(x)<0x1p-4?-0.0:x)+0.0;mismatches+=word<uint64_t>(candidate(x,0))!=word<uint64_t>(expected);double stored=double(_Float16(float((std::abs(x)<0x1p-4?-0.0:x)+0x1p-24)));mismatches+=word<uint64_t>(reclosed(x,0))!=word<uint64_t>(stored);cases++;}std::printf("Selected numeric propagation parity: cases=%u mismatches=%u\\n",cases,mismatches);return mismatches?1:0;}""")
            built=subprocess.run(['clang++','-O3','-ffp-contract=off','-std=c++17',str(source),'-o',str(binary)],capture_output=True,text=True)
            self.assertEqual(built.returncode,0,built.stderr)
            run=subprocess.run([str(binary)],capture_output=True,text=True)
            self.assertEqual(run.returncode,0,run.stdout+run.stderr);print(run.stdout,end='')

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
