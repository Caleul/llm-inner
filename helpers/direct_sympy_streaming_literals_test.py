"""Logical-cost decisions and literal emission; sharing is compiler-only."""
import ast
import gzip
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from direct_sympy_checkpoint import CheckpointStrings
from direct_sympy_conversions_test import cpp
import direct_sympy_savepoints_test as fixtures
from direct_sympy_strings import StringCompiler,syntax
from direct_sympy_streaming_literals import ALIASES,streaming_literals


class StreamingLiteralTests(unittest.TestCase):
    def test_composed_producer_retains_its_original_rounding_recipe(self):
        fixture=fixtures.SavepointTests()
        with tempfile.TemporaryDirectory() as directory:
            model=fixture.model(Path(directory));original=model.conversions.compose_closed
            with streaming_literals(model) as registry:
                first=fixture.first(model)
                def composed():
                    return model.conversions.compose_closed('R16(R32(X999999998*X999999999))',
                        {'X999999998':first,'X999999999':first})
                result=model.producer('fixture:composed-square',composed)
                index=int(ALIASES.fullmatch(result)[1]);recipe=registry.definition_recipes[index]
                self.assertIn('R16',recipe);self.assertIn('R32',recipe)
                self.assertEqual([int(m[1])for m in ALIASES.finditer(recipe)],
                    [int(ALIASES.fullmatch(first)[1])]*2)
                self.assertNotIn('Bits64',recipe)
                self.assertLess(len(recipe),128)
                self.assertNotIn('R16',registry.definitions[index])
                # An embedded composition does not certify its enclosing
                # addition as a whole product.
                embedded=model.producer('fixture:composed-residual',lambda:'R16(R32('+composed()+' + 0.03125))')
                embedded_index=int(ALIASES.fullmatch(embedded)[1])
                self.assertIn('0.03125',registry.definition_recipes[embedded_index])
            self.assertEqual(model.conversions.compose_closed,original)

    def test_shared_root_and_numeric_marker_preserve_exact_outer_cancellation(self):
        from direct_sympy_words import simplify_words
        fixture=fixtures.SavepointTests()
        with tempfile.TemporaryDirectory() as directory:
            model=fixture.model(Path(directory))
            with streaming_literals(model) as registry:
                value=model.producer('fixture:word-root',lambda:'R16(R32(X1 + 0.03125))')
                compiler=model.compiler;context=compiler.context(model.domains)
                payload=compiler._region_word_payloads[(context,value)]
                self.assertEqual(simplify_words('Bits64('+value+')',compiler,model.domains),payload)
                self.assertEqual(simplify_words(value,compiler,model.domains),value)
                marker='CASNumericRegion123456()'
                self.assertTrue(compiler.copy_completed_word_root(value,marker,model.domains))
                self.assertEqual(simplify_words('Bits64('+marker+')',compiler,model.domains),payload)
                self.assertEqual(simplify_words(marker,compiler,model.domains),marker)
                narrowed={}
                self.assertFalse(compiler.copy_completed_word_root(value,'OtherMarker()',narrowed))
                self.assertEqual(simplify_words('Bits64('+value+')',compiler,narrowed),'Bits64('+value+')')
                pending='unknown(X1)';compiler.register_completed_region(pending,model.domains)
                self.assertFalse(compiler.copy_completed_word_root(pending,'OtherMarker()',model.domains))

    def test_emitted_literal_preserves_native_rounding_and_zero_signs(self):
        fixture=fixtures.SavepointTests()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);model=fixture.model(root)
            original=model.producer
            with streaming_literals(model) as registry:
                value=fixture.second(model)
                literal=''.join(registry.chunks(value))
                self.assertEqual(registry.size(value),len(literal))
                self.assertNotIn('CompileValue',literal)
                path=root/'coordinate.expr'
                report=registry.write(path,value,max_characters=len(literal),compressed=False)
                self.assertEqual(path.read_text(),literal)
                self.assertEqual(report['sha256'],hashlib.sha256(path.read_bytes()).hexdigest())
                zipped=root/'coordinate.expr.gz'
                registry.write(zipped,value,max_characters=len(literal))
                self.assertEqual(gzip.decompress(zipped.read_bytes()),path.read_bytes())
                with self.assertRaisesRegex(ValueError,'budget'):
                    registry.write(path,value,max_characters=len(literal)-1,compressed=False)
                self.assertEqual(path.read_text(),literal)
                with self.assertRaisesRegex(ValueError,'Unknown'):
                    registry.intern('CompileValue999()')
                with self.assertRaisesRegex(ValueError,'runtime variable'):
                    registry.intern('X99 + 1')
            self.assertEqual(model.producer,original)
            self.assertIs(model.compiler.expression_size,len)
            source=root/'parity.cpp';binary=root/'parity'
            source.write_text('''#include <cstdint>
#include <cstring>
#include <cmath>
#include <cfenv>
#include <cstdio>
#include <initializer_list>
template<class T,class U>T word(U value){T result;static_assert(sizeof(T)==sizeof(U));std::memcpy(&result,&value,sizeof(result));return result;}
double candidate(double X1){return '''+cpp(syntax(path.read_text()))+''';}
int main(){if(std::fesetround(FE_TONEAREST))return 2;unsigned cases=0;
for(unsigned bits=0;bits<=0x3c00;bits++)for(unsigned sign:{0u,0x8000u}){
double x=word<_Float16>(uint16_t(bits|sign));
double first=static_cast<_Float16>(float(float(x)+float(x/2.0)));
double expected=static_cast<_Float16>(float(first*0.5));cases++;
if(word<uint64_t>(candidate(x))!=word<uint64_t>(expected))return 1;
}std::printf("Emitted streaming literal: cases=%u mismatches=0\\n",cases);}
''')
            subprocess.run(['clang++','-O3','-ffp-contract=off','-std=c++17',str(source),'-o',str(binary)],check=True,capture_output=True,text=True)
            result=subprocess.run([str(binary)],check=True,capture_output=True,text=True,timeout=120)
            self.assertIn('cases=30722 mismatches=0',result.stdout)
            print(result.stdout,end='')

    def test_default_synchronization_uses_logical_cost_and_can_override_it(self):
        compiler=StringCompiler()
        source='U64Or(Bits64(Piecewise((X1,X1>0),(1.0,True))),Bits64(Piecewise((X1,X1>0),(2.0,True))))'
        compiler.expression_size=lambda text:100 if text.count('Piecewise(')==2 else 1000
        expected=compiler.stabilize(source,{})
        self.assertEqual(compiler.synchronize(source,{}),expected)
        self.assertEqual(compiler.synchronization_events[-1][-1],'no-size-reduction')
        self.assertNotEqual(compiler.synchronize(source,{},measure=len),expected)

    def test_numeric_frontier_cost_expands_registered_literals_before_comparison(self):
        fixture=fixtures.SavepointTests()
        with tempfile.TemporaryDirectory() as directory:
            model=fixture.model(Path(directory))
            with streaming_literals(model) as registry:
                value=fixture.first(model)
                index=int(ALIASES.fullmatch(value)[1])
                self.assertIn(index,registry.definition_recipes)
                self.assertIn("R16",registry.definition_recipes[index])
                self.assertEqual(len(registry.definition_proofs),len(registry.definitions))
                captured=[]
                def check(compact,baseline,protected,pure_functions,views,measure):
                    for token,text in protected.items():
                        captured.append(token)
                        candidate='Bits64('+token+'())'
                        restored=candidate.replace(token+'()',text)
                        self.assertEqual(measure(candidate),registry.size(restored))
                    return baseline
                with patch.object(model.conversions,'close_frontier_candidates',side_effect=check):
                    model.conversions.compose_closed('R16(R32(X999999998*0.5))',{'X999999998':value})
                self.assertTrue(captured)

    def test_context_and_failure_restore_the_original_producer_and_cost(self):
        fixture=fixtures.SavepointTests()
        with tempfile.TemporaryDirectory() as directory:
            model=fixture.model(Path(directory));original=model.producer
            with self.assertRaisesRegex(ValueError,'context changed'):
                with streaming_literals(model) as registry:
                    model.domains={}
                    registry.intern('1.0')
            self.assertEqual(model.producer,original)
            self.assertIs(model.compiler.expression_size,len)

    @unittest.skipUnless(os.environ.get('LLM_INNER_DIRECT_JSON_CHECKPOINT'),'checkpoint not configured')
    def test_emitted_checkpoint_inverse_matches_native_reference_on_finite_half_pairs(self):
        checkpoint=os.environ['LLM_INNER_DIRECT_JSON_CHECKPOINT']
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            path=Path(os.environ.get('LLM_INNER_DIRECT_PREFIX_OUTPUT',str(root/'pre-inverse.expr')))
            with CheckpointStrings(checkpoint,StringCompiler(max_characters=65536)) as model:
                with streaming_literals(model) as registry:
                    model.norm('model.layers.0.pre','model.layers.0.input_layernorm.weight',0,lambda i:f'X{i+1}')
                    registry.write(path,registry.names['model.layers.0.pre:inverse'],max_characters=65536,compressed=False)
                    epsilon=model.config['rms_norm_eps']
            text=path.read_text()
            self.assertNotIn('CompileValue',text)
            self.assertNotIn('CASNumericRegion',text)
            source=root/'inverse.cpp';binary=root/'inverse'
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
double y=word<_Float16>(other);
float sum=float(float(x*x)+float(y*y));float mean=float(float(sum/2.0f)+'''+repr(epsilon)+'''f);
double expected=float(1.0f/std::sqrt(mean));cases++;
if(word<uint64_t>(candidate(x,y))!=word<uint64_t>(expected))return 1;
}}std::printf("Emitted checkpoint inverse: cases=%u mismatches=0\\n",cases);}
''')
            subprocess.run(['clang++','-O3','-ffp-contract=off','-std=c++17',str(source),'-o',str(binary)],check=True,capture_output=True,text=True)
            result=subprocess.run([str(binary)],check=True,capture_output=True,text=True,timeout=120)
            self.assertIn('cases=888832 mismatches=0',result.stdout)
            print(result.stdout,end='')

    @unittest.skipUnless(os.environ.get('LLM_INNER_DIRECT_JSON_CHECKPOINT'),'checkpoint not configured')
    def test_complete_position_zero_composition_matches_fresh_reference(self):
        # This checks the compiler's complete composition, not an emitted
        # multi-terabyte artifact and not the last token of a longer sequence.
        checkpoint=os.environ['LLM_INNER_DIRECT_JSON_CHECKPOINT']
        class LazyBranches(ast.NodeTransformer):
            def visit_Call(self,node):
                node=self.generic_visit(node)
                if node.func.id!='Piecewise':return node
                result=ast.Constant(value=0)
                for pair in reversed(node.args):result=ast.IfExp(test=pair.elts[1],body=pair.elts[0],orelse=result)
                return result
        mask=2**64-1
        functions={'F64FromU64':float,'U64FromF64':int,
            'Bits64':lambda x:struct.unpack('Q',struct.pack('d',float(x)))[0],
            'Float64':lambda x:struct.unpack('d',struct.pack('Q',x))[0],
            'U64Add':lambda a,b:(a+b)&mask,'U64Mul':lambda a,b:(a*b)&mask,
            'U64And':lambda a,b:a&b,'U64Or':lambda a,b:a|b,'U64Shr':lambda a,b:a>>b,
            'And':lambda *x:all(x),'Or':lambda *x:any(x),'Not':lambda x:not x}
        with CheckpointStrings(checkpoint,StringCompiler(max_characters=16*1024**2)) as model:
            with streaming_literals(model) as registry:
                output=model.coordinate(2)
                programs=[compile(ast.fix_missing_locations(ast.Expression(LazyBranches().visit(syntax(text)))),'<compile-literal>','eval') for text in registry.definitions]
                index=int(output.removeprefix('CompileValue').removesuffix('()'))
                size=registry.size(output)
                self.assertTrue(all(event[2:4]==('factor','simplify') for event in model.compiler.events))
        with tempfile.TemporaryDirectory() as directory:
            reference=Path(directory)/'reference.json'
            subprocess.run([sys.executable,str(Path(__file__).with_name('capture_direct_json_reference.py')),checkpoint,str(reference)],check=True,capture_output=True,text=True)
            corpus=json.loads(reference.read_text())
        for row in corpus['cases']:
            values={f'X{i+1}':struct.unpack('e',struct.pack('H',bits))[0] for i,bits in enumerate(row['inputBits'][0])}
            for i,program in enumerate(programs):
                result=eval(program,{'__builtins__':{},**functions},values)
                values['CompileValue'+str(i)]=lambda result=result:result
            actual=values['CompileValue'+str(index)]()
            self.assertEqual('0x'+struct.pack('>d',actual).hex(),row['logitF64Bits'][0][2],row['label'])
        print(f'Complete compiler composition: cases={len(corpus["cases"])} mismatches=0; position=0 dimension=2; logicalCharacters={size}; emittedFinalParity=false')


if __name__=='__main__':unittest.main()
