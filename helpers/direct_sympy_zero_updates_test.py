"""Signed zero changes signs but never the representable finite format."""
import ast
from fractions import Fraction as F
import math
from pathlib import Path
import subprocess
import tempfile
import unittest
from direct_sympy_arithmetic import simplify_arithmetic
from direct_sympy_conversions import ConversionSession
from direct_sympy_conversions_test import cpp
from direct_sympy_strings import Domain,StringCompiler,syntax


class ZeroUpdateTests(unittest.TestCase):
    def test_all_finite_half_words_and_f32_boundaries_preserve_signed_zero_updates(self):
        expressions=['R16(R32(X1+X2))','R16(R32(X1-X2))','R16(R32(X2-X1))','R16(R32(X1+(-0.0)))','R16(R32((-0.0)+X1))']
        refs=['x+y','x-y','y-x','x+(-0.0)','(-0.0)+x']
        s=ConversionSession(StringCompiler(),{'X1':Domain(F(-65504),F(65504),-24,False),'X2':Domain(F(0),F(0),-24,False)},input_dtype='f16')
        results=[s.close(expression) for expression in expressions]
        for result in results:
            self.assertNotIn('R16',result);self.assertNotIn('R32',result);self.assertNotIn('Bits64',result)
        self.assertEqual(results[3:],[ 'X1','X1'])
        f=ConversionSession(StringCompiler(),{'X1':Domain(-F(float.fromhex('0x1.fffffep127')),F(float.fromhex('0x1.fffffep127')),-149,False),'X2':Domain(F(0),F(0),-149,False)})
        f.f32_values.add(f.key(syntax('X1')))
        fresults=[f.close(expression[4:-1]) for expression in expressions[:3]]
        self.assertEqual(fresults,['X1 + X2','X1 - X2','X2 - X1'])
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);functions=[];half_checks=[];float_checks=[]
            for index,result in enumerate(results+fresults):
                artifact=root/f'zero-{index}.expr';artifact.write_text(result)
                functions.append(f'double candidate{index}(double X1,double X2){{return '+cpp(syntax(artifact.read_text()))+';}')
                if index<5:half_checks.append(f'cases++;if(word<uint64_t>(candidate{index}(x,y))!=word<uint64_t>(double(_Float16(float({refs[index]})))))return 1;')
                else:float_checks.append(f'cases++;if(word<uint64_t>(candidate{index}(x,y))!=word<uint64_t>(double(float({refs[index-5]}))))return 1;')
            source=root/'native.cpp';binary=root/'native'
            source.write_text('''#include <cstdint>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <cfenv>
#include <initializer_list>
template<class T,class U>T word(U x){T y;std::memcpy(&y,&x,sizeof(y));return y;}
'''+ '\n'.join(functions)+'''\nint main(){if(std::fesetround(FE_TONEAREST))return 2;unsigned cases=0;
for(unsigned bits=0;bits<65536;bits++){if((bits&0x7c00)==0x7c00)continue;double x=word<_Float16>(uint16_t(bits));for(double y:{0.0,-0.0}){'''+ ''.join(half_checks)+'''}}
for(unsigned e=0;e<=254;e++)for(unsigned m:{0u,1u,2u,3u,4194304u,8388606u,8388607u})for(unsigned sign:{0u,0x80000000u}){
double x=word<float>(uint32_t(sign|(e<<23)|m));for(double y:{0.0,-0.0}){'''+ ''.join(float_checks)+'''}}
std::printf("Zero-update native parity: cases=%u mismatches=0\\n",cases);}
''')
            built=subprocess.run(['clang++','-O3','-ffp-contract=off','-std=c++17',str(source),'-o',str(binary)],capture_output=True,text=True)
            self.assertEqual(built.returncode,0,built.stderr)
            tested=subprocess.run([str(binary)],capture_output=True,text=True)
            self.assertEqual(tested.returncode,0,tested.stdout+tested.stderr)
            self.assertIn('cases=656300 mismatches=0',tested.stdout);print(tested.stdout,end='')

    def test_zero_update_proofs_are_branch_local_and_nonzero_updates_keep_rounding(self):
        s=ConversionSession(StringCompiler(),{'X1':Domain(F(-2),F(2),-24,False),'X2':Domain(-F(2)**-24,F(2)**-24,-24,False)},input_dtype='f16')
        self.assertNotEqual(s.value_kind(syntax('X1+X2')),'half')
        result=s.close('Piecewise((R16(R32(X1+X2)), X2 == 0.0), (R16(R32(X1-X2)), True))')
        node=syntax(result)
        self.assertEqual(ast.unparse(node.args[0].elts[0]),'X1 + X2')
        self.assertNotEqual(ast.unparse(node.args[1].elts[0]),'X1 - X2')
        self.assertNotEqual(s.value_kind(syntax('X1+X2')),'half')
        self.assertEqual(s.bounds(syntax('X2')).maximum,2**-24)
        self.assertEqual(s.branch_depth,0)

    def test_exact_zero_bounds_preserve_signs_and_division_barriers(self):
        from direct_sympy_conversions import FiniteSource
        s=ConversionSession(StringCompiler(),{'X1':Domain(F(-65504),F(65504),-24,False),'X2':Domain(F(0),F(0),-24,False)},input_dtype='f16')
        finite=s.bounds(syntax('X1'))
        for text in ('X1+X2','X2+X1','X1-X2'):
            self.assertEqual(s.bounds(syntax(text)),finite)
        self.assertEqual(s.bounds(syntax('X2-X1')),FiniteSource(-finite.maximum,-finite.minimum,finite.quantum))
        for text in ('X1*X2','X2*X1'):
            bound=s.bounds(syntax(text))
            self.assertEqual((bound.minimum,bound.maximum),(0,0))
            self.assertFalse(s.no_negative_zero(syntax(text)))
        self.assertIsNone(s.bounds(syntax('X2/X1')))
        self.assertIsNone(s.bounds(syntax('X2/0.0')))
        self.assertIsNone(s.bounds(syntax('unknown()*X2')))
        with s.branch_context({**s.domains,'X1':Domain(F(1),F(65504),-24,True)},{},()):
            bound=s.bounds(syntax('X2/X1'))
            self.assertEqual((bound.minimum,bound.maximum),(0,0))
        self.assertIsNone(s.bounds(syntax('X2/X1')))
        expressions=['R16(R32(X1*X2))','R16(R32(X2*X1))']
        results=[s.close(text) for text in expressions]
        with s.branch_context({**s.domains,'X1':Domain(F(1),F(65504),-24,True)},{},()):
            positive=s.close('R16(R32(X2/X1))')
        with s.branch_context({**s.domains,'X1':Domain(F(-65504),F(-1),-24,True)},{},()):
            negative=s.close('R16(R32(X2/X1))')
        # Validate actual emitted strings, including both zero signs.
        functions='\n'.join('double candidate'+str(i)+'(double X1,double X2){return '+cpp(syntax(result))+';}' for i,result in enumerate(results+[positive,negative]))
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'native.cpp';binary=root/'native'
            source.write_text('#include <cstdint>\n#include <cstring>\n#include <cmath>\n#include <cstdio>\n#include <initializer_list>\ntemplate<class T,class U>T word(U x){T y;std::memcpy(&y,&x,sizeof(y));return y;}\n'+functions+"""\nint main(){unsigned cases=0,mismatches=0;for(unsigned b=0;b<65536;b++){if((b&0x7c00)==0x7c00)continue;double x=word<_Float16>(uint16_t(b));for(double y:{0.0,-0.0}){mismatches+=word<uint64_t>(candidate0(x,y))!=word<uint64_t>(double(_Float16(float(x*y))));cases++;mismatches+=word<uint64_t>(candidate1(x,y))!=word<uint64_t>(double(_Float16(float(y*x))));cases++;if(std::abs(x)>=1.0){double got=x>0?candidate2(x,y):candidate3(x,y);mismatches+=word<uint64_t>(got)!=word<uint64_t>(double(_Float16(float(y/x))));cases++;}}}std::printf("Exact-zero interval parity: cases=%u mismatches=%u\\n",cases,mismatches);return mismatches?1:0;}""")
            built=subprocess.run(['clang++','-O3','-ffp-contract=off','-std=c++17',str(source),'-o',str(binary)],capture_output=True,text=True)
            self.assertEqual(built.returncode,0,built.stderr)
            tested=subprocess.run([str(binary)],capture_output=True,text=True)
            self.assertEqual(tested.returncode,0,tested.stdout+tested.stderr)
            self.assertIn('mismatches=0',tested.stdout);print(tested.stdout,end='')

    def test_positive_zero_and_nonfinite_unknown_operands_are_not_dropped(self):
        s=ConversionSession(StringCompiler(),{'X1':Domain(F(-1),F(1),-24,False)},input_dtype='f16')
        self.assertIn('+',simplify_arithmetic('X1+0.0',s))
        self.assertEqual(simplify_arithmetic('X1+(-0.0)',s),'X1')
        self.assertEqual(simplify_arithmetic('(-0.0)+X1',s),'X1')
        self.assertIn('unknown',simplify_arithmetic('unknown()+(-0.0)',s))
        self.assertIsNone(s.value_kind(syntax('unknown()+0.0')))


if __name__=='__main__':unittest.main()
