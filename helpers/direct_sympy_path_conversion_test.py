import subprocess
import tempfile
from fractions import Fraction as F
from pathlib import Path
import unittest
from unittest.mock import patch

import direct_sympy_conversions as conversions
from direct_sympy_conversions import ConversionSession
from direct_sympy_conversions_test import cpp
from direct_sympy_strings import Domain,StringCompiler,syntax


class PathConversionTests(unittest.TestCase):
    def build(self,expression,domains):
        session=ConversionSession(StringCompiler(),domains,input_dtype='f16')
        return session,session.close(expression)

    def native(self,expressions,body):
        functions='\n'.join('double '+name+'(double X1,double X2){return '+cpp(syntax(text))+';}' for name,text in expressions.items())
        header='#include <cstdint>\n#include <cstring>\n#include <cmath>\n#include <cstdio>\n#include <initializer_list>\ntemplate<class T,class U>T word(U value){T result;std::memcpy(&result,&value,sizeof(result));return result;}\n'
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'paths.cpp';binary=root/'paths'
            source.write_text(header+functions+'\n'+body)
            built=subprocess.run(['clang++','-O3','-ffp-contract=off','-std=c++17',str(source),'-o',str(binary)],capture_output=True,text=True)
            self.assertEqual(built.returncode,0,built.stderr)
            run=subprocess.run([str(binary)],capture_output=True,text=True)
            self.assertEqual(run.returncode,0,run.stdout+run.stderr);print(run.stdout,end='')

    def test_disconnected_normal_path_preserves_both_signs_and_signed_zeros(self):
        domains={n:Domain(-F(1),F(1),-24,False) for n in ('X1','X2')}
        guard='U64And(Bits64(X1*X2),9223372036854775807)<4544132024016830464'
        expression='Piecewise((R16(X1*X2),'+guard+'),(R16(X1*X2),True))'
        old=conversions.lower_finite_conversion
        def legacy(*args,**kwargs):kwargs.pop('facts',None);return old(*args,**kwargs)
        with patch('direct_sympy_conversions.lower_finite_conversion',side_effect=legacy):_,baseline=self.build(expression,domains)
        session,result=self.build(expression,domains)
        self.assertLess(len(result),len(baseline))
        self.assertIn(('numeric/R16','small','path-proved-false'),session.compiler.condition_events)
        self.assertEqual(session.branch_facts,())
        self.assertEqual(session.branch_depth,0)
        _,negated=self.build('Piecewise((R16(X1*X2),Not('+guard+')),(R16(X1*X2),True))',domains)
        self.native({'candidate':result,'baseline':baseline,'negated':negated},'''int main(){unsigned cases=0,mismatches=0,negative=0,zeros=0;for(unsigned b=0;b<65536;b++){if((b&0x7c00)==0x7c00)continue;double x=word<_Float16>(uint16_t(b));if(std::abs(x)>1)continue;for(double y:{-1.0,-0x1p-13,-0x1p-24,-0.0,0.0,0x1p-24,0x1p-13,1.0}){double expected=double(_Float16(x*y));uint64_t bits=word<uint64_t>(expected);mismatches+=word<uint64_t>(candidate(x,y))!=bits || word<uint64_t>(baseline(x,y))!=bits || word<uint64_t>(negated(x,y))!=bits;negative+=expected<0;zeros+=expected==0 && std::signbit(expected);cases++;}}std::printf("Path Half conversion parity: cases=%u mismatches=%u negative=%u negativeZeros=%u\\n",cases,mismatches,negative,zeros);return mismatches || !negative || !zeros?1:0;}''')
        print('Path conversion strings: '+str(len(baseline))+' -> '+str(len(result)))

    def test_f32_path_conversion_preserves_normal_subnormal_and_midpoint_boundaries(self):
        domains={'X1':Domain(-F(1),F(1),-1074,False)}
        guard='U64And(Bits64(X1),9223372036854775807)<'+str(0x3810000000000000)
        session=ConversionSession(StringCompiler(),domains)
        result=session.close('Piecewise((R32(X1),'+guard+'),(R32(X1),True))')
        self.assertIn(('numeric/R32','small','path-proved-false'),session.compiler.condition_events)
        self.native({'candidate':result},"""int main(){unsigned cases=0,mismatches=0;for(uint64_t base:{UINT64_C(0x3800000000000000),UINT64_C(0x3810000000000000),UINT64_C(0x3820000000000000)}){for(uint64_t m=0;m<0x800000;m+=127){for(uint64_t delta:{UINT64_C(0),UINT64_C(1),UINT64_C(268435455),UINT64_C(268435456),UINT64_C(268435457),UINT64_C(536870911)}){for(uint64_t sign:{UINT64_C(0),UINT64_C(0x8000000000000000)}){double x=word<double>(sign | (base+(m<<29)+delta));double expected=double(float(x));mismatches+=word<uint64_t>(candidate(x,0))!=word<uint64_t>(expected);cases++;}}}}for(double x:{-0.0,0.0,-0x1p-150,0x1p-150,-0x1p-149,0x1p-149}){mismatches+=word<uint64_t>(candidate(x,0))!=word<uint64_t>(double(float(x)));cases++;}std::printf("Path F32 conversion parity: cases=%u mismatches=%u\\n",cases,mismatches);return mismatches?1:0;}""")

    def test_tandem_uses_its_exact_quadratic_preimage_only_in_the_selected_arm(self):
        domains={'X1':Domain(-F(3,128),F(3,128),-24,False)}
        polynomial='X1*(0.5+X1*0.25)';guard='(X1+2**-25)**2<2**-26'
        expression='Piecewise((R16(R32('+polynomial+')),'+guard+'),(R16(R32('+polynomial+')),True))'
        old=conversions.lower_tandem
        def legacy(*args,**kwargs):kwargs.pop('facts',None);return old(*args,**kwargs)
        with patch('direct_sympy_conversions.lower_tandem',side_effect=legacy):_,baseline=self.build(expression,domains)
        session,result=self.build(expression,domains)
        self.assertEqual(result,baseline)
        self.assertIn(('numeric/tandem','small','path-proved-false'),session.compiler.condition_events)
        self.native({'candidate':result},'''int main(){unsigned cases=0,mismatches=0;for(unsigned b=0;b<65536;b++){if((b&0x7c00)==0x7c00)continue;double x=word<_Float16>(uint16_t(b));if(std::abs(x)>3.0/128)continue;double expected=double(_Float16(float(x*(0.5+x*0.25))));mismatches+=word<uint64_t>(candidate(x,0))!=word<uint64_t>(expected);cases++;}std::printf("Path tandem conversion parity: cases=%u mismatches=%u\\n",cases,mismatches);return mismatches?1:0;}''')

    def test_unrelated_guards_do_not_admit_a_normal_shortcut_or_escape_to_global_closure(self):
        domains={n:Domain(-F(1),F(1),-24,False) for n in ('X1','X2')}
        session=ConversionSession(StringCompiler(),domains,input_dtype='f16')
        source='X1*X2'
        before=session.close('R16('+source+')')
        unrelated=syntax('U64And(Bits64(X1),9223372036854775807)<4544132024016830464')
        facts=session.compiler.branch_facts.assume(unrelated,False,())
        with session.branch_context(domains,{},facts):
            inside=session._close('R16('+source+')')
        self.assertEqual(inside,before)
        self.assertEqual(session.branch_facts,())
        self.assertEqual(session.close('R16('+source+')'),before)
        # A normal magnitude guard is disconnected; do not invent a signed
        # lower bound or narrow either fundamental input's accepted domain.
        guard=syntax('U64And(Bits64('+source+'),9223372036854775807)<4544132024016830464')
        self.assertEqual(session.magnitude_guard_bounds(guard,False,{}),{})
        self.assertEqual(dict(session.domains),domains)


if __name__=='__main__':unittest.main()
