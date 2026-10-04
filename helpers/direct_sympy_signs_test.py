import ast
from fractions import Fraction as F
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from direct_sympy_conversions import ConversionSession
from direct_sympy_conversions_test import cpp
from direct_sympy_strings import Domain,StringCompiler,syntax
from direct_sympy_signs import project,SIGN,call


class SignProjectionTests(unittest.TestCase):
    def test_native_half_sign_projection_includes_zero_underflow_and_negative_factors(self):
        domains={name:Domain(F(-65504),F(65504),-24,False) for name in ('X1','X2')}
        session=ConversionSession(StringCompiler(),domains,input_dtype='f16')
        expressions=['X1*X2','X1/X2','-X1','X1*0.000000000000000000000000000000000000000000001','R16(X1)']
        nodes=[project(syntax(text),session) for text in expressions]
        bodies='\n'.join('uint64_t p'+str(i)+'(double X1,double X2){return '+cpp(node)+';}' for i,node in enumerate(nodes))
        header='#include <cstdint>\n#include <cstring>\n#include <cmath>\n#include <cstdio>\n#include <initializer_list>\ntemplate<class T,class U>T word(U value){T result;std::memcpy(&result,&value,sizeof(result));return result;}\n'
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'signs.cpp';binary=root/'signs'
            source.write_text(header+bodies+'''\nint main(){unsigned cases=0,mismatches=0;for(unsigned b=0;b<65536;b++){if((b&0x7c00)==0x7c00)continue;double x=word<_Float16>(uint16_t(b));for(uint16_t c:{uint16_t(0),uint16_t(0x8000),uint16_t(1),uint16_t(0x8001),uint16_t(0x0400),uint16_t(0x8400),uint16_t(0x3c00),uint16_t(0xbc00),uint16_t(0x7bff),uint16_t(0xfbff)}){double y=word<_Float16>(c);mismatches+=p0(x,y)!=(word<uint64_t>(x*y)&(UINT64_C(1)<<63));cases++;if(y!=0){mismatches+=p1(x,y)!=(word<uint64_t>(x/y)&(UINT64_C(1)<<63));cases++;}mismatches+=p2(x,y)!=(word<uint64_t>(-x)&(UINT64_C(1)<<63));cases++;mismatches+=p3(x,y)!=(word<uint64_t>(x*1e-45)&(UINT64_C(1)<<63));cases++;mismatches+=p4(x,y)!=(word<uint64_t>(double(_Float16(x)))&(UINT64_C(1)<<63));cases++;}}std::printf("Finite sign projection parity: cases=%u mismatches=%u\\n",cases,mismatches);return mismatches?1:0;}''')
            build=subprocess.run(['clang++','-O3','-ffp-contract=off','-std=c++17',str(source),'-o',str(binary)],capture_output=True,text=True)
            self.assertEqual(build.returncode,0,build.stderr)
            run=subprocess.run([str(binary)],capture_output=True,text=True)
            self.assertEqual(run.returncode,0,run.stdout+run.stderr);print(run.stdout,end='')

    def test_completed_sign_projections_reduce_down_conversion_without_changing_parity(self):
        domains={name:Domain(-F(1,64),F(1,64),-24,False) for name in ('X1','X2')}
        def build():
            session=ConversionSession(StringCompiler(),domains,input_dtype='f16')
            a=session.close('Silu16(X1)');u=session.close('R16(X2/3.0)')
            g=session.compose_closed('R16(R32(X999999998 * X999999999))',{'X999999998':a,'X999999999':u})
            d=session.compose_closed('R16(R32(X999999998 * -0.015777587890625))',{'X999999998':g})
            return session,a,u,g,d
        def original(node,session):return call('U64And',call('Bits64',node),ast.Constant(SIGN))
        with patch('direct_sympy_signs.project',side_effect=original):control,_,_,_,baseline=build()
        session,a,u,g,result=build()
        self.assertLess(len(result),len(baseline))
        self.assertIn(g,session.closed_sign_literals)
        self.assertLess(len(session.closed_sign_literals[g]),len(g))
        for name in ('CASNumericRegion','CASStableRegion','R16(','R32(','Silu16('):self.assertNotIn(name,result)
        header='#include <cstdint>\n#include <cstring>\n#include <cmath>\n#include <cstdio>\n#include <initializer_list>\ntemplate<class T,class U>T word(U value){T result;std::memcpy(&result,&value,sizeof(result));return result;}\n'
        functions='\n'.join('double '+name+'(double X1,double X2){return '+cpp(syntax(text))+';}' for name,text in [('candidate',result),('baseline',baseline),('act',a),('up',u)])
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'composition.cpp';binary=root/'composition'
            source.write_text(header+functions+'''\nint main(){unsigned cases=0,mismatches=0;for(unsigned b=0;b<65536;b++){if((b&0x7c00)==0x7c00)continue;double x=word<_Float16>(uint16_t(b));if(std::abs(x)>0x1p-6)continue;for(double y:{-0x1p-6,-0x1p-14,-0x1p-24,-0.0,0.0,0x1p-24,0x1p-14,0x1p-6}){double g=double(_Float16(float(act(x,y)*up(x,y))));double expected=double(_Float16(float(g*-0.015777587890625)));uint64_t got=word<uint64_t>(candidate(x,y));mismatches+=got!=word<uint64_t>(expected);mismatches+=got!=word<uint64_t>(baseline(x,y));cases++;}}std::printf("Completed down projection parity: cases=%u mismatches=%u\\n",cases,mismatches);return mismatches?1:0;}''')
            built=subprocess.run(['clang++','-O3','-ffp-contract=off','-std=c++17',str(source),'-o',str(binary)],capture_output=True,text=True);self.assertEqual(built.returncode,0,built.stderr)
            run=subprocess.run([str(binary)],capture_output=True,text=True);self.assertEqual(run.returncode,0,run.stdout+run.stderr);print(run.stdout,end='')
        print('Sign projection strings: '+str(len(baseline))+' -> '+str(len(result)))

    def test_checkpoint_seeded_down_reduction_preserves_the_positive_zero_accumulator(self):
        from direct_sympy_checkpoint import CheckpointStrings
        checkpoint=Path(__file__).resolve().parents[1]/'docs/evidence/direct-sympy-test-checkpoint'
        domains={name:Domain(-F(1,64),F(1,64),-24,False) for name in ('X1','X2')}
        session=ConversionSession(StringCompiler(),domains,input_dtype='f16')
        a=session.close('Silu16(X1)');u=session.close('R16(X2/3.0)')
        g=session.compose_closed('R16(R32(X999999998 * X999999999))',{'X999999998':a,'X999999999':u})
        with CheckpointStrings(checkpoint,session.compiler) as builder:
            builder.domains=domains;builder.conversions=session
            coefficient=builder.weight('model.layers.0.mlp.down_proj.weight',0,0)
            result=session.close(builder.linear('model.layers.0.mlp.down_proj.weight',0,lambda _:g))
        functions='\n'.join('double '+name+'(double X1,double X2){return '+cpp(syntax(text))+';}' for name,text in [('candidate',result),('act',a),('up',u)])
        header='#include <cstdint>\n#include <cstring>\n#include <cmath>\n#include <cstdio>\n#include <initializer_list>\ntemplate<class T,class U>T word(U value){T result;std::memcpy(&result,&value,sizeof(result));return result;}\n'
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'seeded.cpp';binary=root/'seeded'
            source.write_text(header+functions+'\nint main(){unsigned cases=0,mismatches=0,negativeProductZeros=0;for(unsigned b=0;b<65536;b++){if((b&0x7c00)==0x7c00)continue;double x=word<_Float16>(uint16_t(b));if(std::abs(x)>0x1p-6)continue;for(double y:{-0x1p-6,-0x1p-14,-0x1p-24,-0.0,0.0,0x1p-24,0x1p-14,0x1p-6}){double g=double(_Float16(float(act(x,y)*up(x,y))));float product=float(g*'+coefficient+');float reduced=0.0f+product;negativeProductZeros+=product==0 && std::signbit(product);double expected=double(_Float16(reduced));mismatches+=word<uint64_t>(candidate(x,y))!=word<uint64_t>(expected);cases++;}}std::printf("Seeded checkpoint down reduction parity: cases=%u mismatches=%u negativeProductZeros=%u\\n",cases,mismatches,negativeProductZeros);return mismatches || !negativeProductZeros?1:0;}')
            built=subprocess.run(['clang++','-O3','-ffp-contract=off','-std=c++17',str(source),'-o',str(binary)],capture_output=True,text=True);self.assertEqual(built.returncode,0,built.stderr)
            run=subprocess.run([str(binary)],capture_output=True,text=True);self.assertEqual(run.returncode,0,run.stdout+run.stderr);print(run.stdout,end='')

    def test_branch_only_sign_proofs_do_not_escape_or_override_unknown_values(self):
        domains={'X1':Domain(F(-1),F(1),-24,False)}
        session=ConversionSession(StringCompiler(),domains,input_dtype='f16')
        key=session.key(syntax('X1'))
        before=ast.unparse(project(syntax('X1'),session))
        with session.branch_context({'X1':Domain(F(1,2),F(1),-24,True)},{},()):
            self.assertEqual(ast.unparse(project(syntax('X1'),session)),'0')
            session.sign_projections[key]=ast.Constant(0)
        self.assertEqual(ast.unparse(project(syntax('X1'),session)),before)
        for expression in ('X1 + 1.0','unknown(X1)','X1 / 0.0'):
            self.assertIn('Bits64',ast.unparse(project(syntax(expression),session)))


if __name__=='__main__':unittest.main()
