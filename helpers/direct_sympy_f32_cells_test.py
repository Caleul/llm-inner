"""Finite F32 cell proofs and exact compile-time elementary constants."""
import ast
from fractions import Fraction as F
import math
from pathlib import Path
import random
import struct
import subprocess
import tempfile
import unittest

from direct_sympy_arithmetic import simplify_arithmetic
from direct_sympy_conversions import ConversionSession,FiniteSource
from direct_sympy_conversions_test import cpp
from direct_sympy_strings import Domain,StringCompiler,syntax
from direct_sympy_words import reduce_call,simplify_words
from direct_sympy_signs import is_signed_zero


def session(low,high,delta,typed=True):
    s=ConversionSession(StringCompiler(),{'X1':Domain(F(low),F(high),-149),
        'X2':Domain(-F(delta),F(delta),-149)})
    if typed:s.f32_values.update(s.key(syntax(name)) for name in ('X1','X2'))
    return s


class F32CellTests(unittest.TestCase):
    def test_exact_half_membership_elides_only_a_proved_format_boundary(self):
        step=2**-21;s=ConversionSession(StringCompiler(),{'X1':Domain(F(-125*step),F(125*step),-21,False)})
        s.f32_values.add(s.key(syntax('X1')))
        self.assertEqual(s.value_kind(syntax('X1')),'half')
        after=simplify_arithmetic('R16(X1)',s);self.assertEqual(after,'X1')
        for peak,q in ((2049*2**-24,-24),(2**-25,-25),(65536,5)):
            other=ConversionSession(StringCompiler(),{'X1':Domain(F(-peak),F(peak),q,False)})
            self.assertNotEqual(other.value_kind(syntax('X1')),'half')
        zeros=ConversionSession(StringCompiler(),{'X1':Domain(F(0),F(0),-149,False)})
        zeros.f32_values.add(zeros.key(syntax('X1')))
        self.assertIn('R32',simplify_arithmetic('R32(sqrt(X1))',zeros))
        with tempfile.TemporaryDirectory()as directory:
            root=Path(directory);source=root/'membership.cpp';binary=root/'membership'
            source.write_text('#include <cstdint>\n#include <cstring>\n#include <cstdio>\n#include <cfenv>\n#include <initializer_list>\ntemplate<class T,class U>T word(U x){T y;std::memcpy(&y,&x,sizeof(y));return y;}\ndouble candidate(double X1){return '+cpp(syntax(after))+';}\nint main(){if(std::fesetround(FE_TONEAREST))return 2;unsigned cases=0,mismatches=0;for(int i=-125;i<=125;i++){double x=i*0x1p-21;cases++;mismatches+=word<uint64_t>(candidate(x))!=word<uint64_t>(double(_Float16(float(x))));}for(double x:{0.0,-0.0}){cases++;mismatches+=word<uint64_t>(candidate(x))!=word<uint64_t>(double(_Float16(float(x))));}std::printf("Exact Half membership: cases=%u mismatches=%u\\n",cases,mismatches);return mismatches?1:0;}')
            subprocess.run(['clang++','-O3','-ffp-contract=off','-std=c++17',str(source),'-o',str(binary)],check=True,capture_output=True,text=True)
            result=subprocess.run([str(binary)],check=True,capture_output=True,text=True)
            self.assertIn('cases=253 mismatches=0',result.stdout);print(result.stdout,end='')

    def test_unit_grid_scale_preserves_order_and_native_signed_zero_payloads(self):
        forms=[]
        coefficients=[-0.0,0.0,-1.5,-1.0,-.5,.5,1.,1.5,999.9999389648438,-999.9999389648438,
            65504.,-65504.,2**-149,-2**-149,2**-1074,-2**-1074,float.fromhex('0x1.fffffep127'),float.fromhex('0x1.fffffffffffffp1023')]
        for exponent in (-1074,-149,-24,-14,-1,0,5,15,100,1023):
            unit=2.0**exponent
            s=ConversionSession(StringCompiler(),{'X1':Domain(F(-unit),F(unit),exponent,False)})
            for c in coefficients:
                for text in (f'R16(R32(X1*({c!r})))',f'R32(R16((X1*({c!r}))/3.0))'):
                    result=s.unit_grid_rounding_scale(syntax(text))
                    if result is None:continue
                    after=ast.unparse(result);self.assertNotIn('R16',after);self.assertNotIn('R32',after)
                    self.assertEqual(after.count('X1'),1)
                    actual=simplify_arithmetic(text,s)
                    self.assertNotIn('R16',actual);self.assertNotIn('R32',actual)
                    self.assertEqual(simplify_arithmetic(actual,s),actual)
                    forms.append((unit,text,actual))
        self.assertGreater(len(forms),100)
        s=ConversionSession(StringCompiler(),{'X1':Domain(F(-2),F(2),0,False)})
        for text in ('R16(X1*1.5)','R16(X1*X1)','R16(1.0/X1)','R16(X1+1.0)','R16(unknown(X1)*1.5)'):
            self.assertIsNone(s.unit_grid_rounding_scale(syntax(text)))
        alias,coefficient=syntax('CompileValue0()'),syntax('CompileValue1()')
        source,constant=s.key(alias),s.key(coefficient)
        s.completed[source]=FiniteSource(-1,1,-24);s.half_values.add(source)
        s.completed[constant]=FiniteSource(999.9999389648438,999.9999389648438,-14)
        s.pure_numeric_regions.update((source,constant))
        expression=syntax('R16(R32(CompileValue0()*CompileValue1()))')
        self.assertIsNone(s.unit_grid_rounding_scale(expression))
        with s.branch_context(s.domains,{source:FiniteSource(-2**-24,2**-24,-24)},()):
            result=s.unit_grid_rounding_scale(expression)
            self.assertEqual(ast.unparse(result),'1000.0 * CompileValue0()')
            simplified=simplify_arithmetic(ast.unparse(expression),s)
            self.assertEqual(s.bounds(syntax(simplified)).quantum,-21)
        self.assertIsNone(s.unit_grid_rounding_scale(expression))
        # Numerical bounds alone cannot authorize discarding an unknown call.
        s.pure_numeric_regions.remove(constant)
        with s.branch_context(s.domains,{source:FiniteSource(-2**-24,2**-24,-24)},()):
            self.assertIsNone(s.unit_grid_rounding_scale(expression))
        def reference(node):
            if isinstance(node,ast.Call):return 'double('+('float' if node.func.id=='R32' else '_Float16')+'('+reference(node.args[0])+'))'
            if isinstance(node,ast.BinOp):return '('+reference(node.left)+('*' if isinstance(node.op,ast.Mult) else '/')+reference(node.right)+')'
            if isinstance(node,ast.UnaryOp):return '(-double('+reference(node.operand)+'))'
            return cpp(node)
        functions='\n'.join(f'double before{i}(double X1){{return {reference(syntax(a))};}}\ndouble after{i}(double X1){{return {cpp(syntax(b))};}}'for i,(_,a,b)in enumerate(forms))
        checks=''.join(f'for(double x:{{{u.hex()},-{u.hex()},0.0,-0.0}}){{cases++;mismatches+=word<uint64_t>(before{i}(x))!=word<uint64_t>(after{i}(x));}}'for i,(u,_,_)in enumerate(forms))
        with tempfile.TemporaryDirectory()as directory:
            root=Path(directory);source=root/'unit.cpp';binary=root/'unit'
            source.write_text('#include <cstdint>\n#include <cstring>\n#include <cmath>\n#include <cstdio>\n#include <cfenv>\n#include <initializer_list>\ntemplate<class T,class U>T word(U x){T y;std::memcpy(&y,&x,sizeof(y));return y;}\n'+functions+'\nint main(){if(std::fesetround(FE_TONEAREST))return 2;unsigned cases=0,mismatches=0;'+checks+'std::printf("Unit-grid scale native parity: cases=%u mismatches=%u\\n",cases,mismatches);return mismatches?1:0;}')
            subprocess.run(['clang++','-O3','-ffp-contract=off','-std=c++17',str(source),'-o',str(binary)],check=True,capture_output=True,text=True)
            run=subprocess.run([str(binary)],check=True,capture_output=True,text=True)
            self.assertIn(f'cases={len(forms)*4} mismatches=0',run.stdout);print(run.stdout,end='')

    def test_proved_grid_removes_only_exact_f32_casts(self):
        s=ConversionSession(StringCompiler(),{'X1':Domain(F(-1),F(1),-8),
            'X2':Domain(F(-1),F(1),-8)})
        result=simplify_arithmetic('R32(X1+X2)',s)
        self.assertNotIn('R32',result)
        finer=ConversionSession(StringCompiler(),{'X1':Domain(F(-1),F(1),-30),
            'X2':Domain(F(-1),F(1),-30)})
        self.assertIn('R32',simplify_arithmetic('R32(X1+X2)',finer))
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);artifact=root/'grid.expr';artifact.write_text(result)
            source=root/'grid.cpp';binary=root/'grid'
            source.write_text('''#include <cstdint>
#include <cstring>
#include <cstdio>
#include <initializer_list>
template<class T,class U>T word(U value){T result;std::memcpy(&result,&value,sizeof(result));return result;}
double candidate(double X1,double X2){return '''+cpp(syntax(artifact.read_text()))+''';}
int main(){unsigned cases=0;for(int i=-256;i<=256;i++)for(int j=-256;j<=256;j++){
double x=i/256.0,y=j/256.0;cases++;if(word<uint64_t>(candidate(x,y))!=word<uint64_t>(double(float(float(x)+float(y)))))return 1;}
for(double x:{0.0,-0.0})for(double y:{0.0,-0.0}){cases++;if(word<uint64_t>(candidate(x,y))!=word<uint64_t>(double(float(float(x)+float(y)))))return 1;}
std::printf("F32 grid native parity: cases=%u mismatches=0\\n",cases);}
''')
            subprocess.run(['clang++','-O3','-ffp-contract=off','-std=c++17',str(source),'-o',str(binary)],check=True,capture_output=True,text=True)
            outcome=subprocess.run([str(binary)],check=True,capture_output=True,text=True)
            self.assertIn('cases=263173 mismatches=0',outcome.stdout);print(outcome.stdout,end='')
    def test_cell_reduction_preserves_all_finite_half_centers_and_f32_binade_boundaries(self):
        for exponent in range(-149,128):
            low=2.0**exponent;high=min(2*low,3.4028234663852886e38)
            radius=2.0**max(-150,exponent-25)
            for sign in (1,-1):
                s=session(low if sign>0 else -high,high if sign>0 else -low,radius/4)
                self.assertEqual(simplify_arithmetic('R32(X1+X2)',s),'X1')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);artifact=root/'cell.expr';artifact.write_text('X1')
            source=root/'cell.cpp';binary=root/'cell'
            source.write_text('''#include <cstdint>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <cfenv>
#include <initializer_list>
template<class T,class U>T word(U value){T result;std::memcpy(&result,&value,sizeof(result));return result;}
double candidate(double X1,double X2){return '''+cpp(syntax(artifact.read_text()))+''';}
unsigned cases=0;
bool check(double x){int exponent;std::frexp(std::fabs(x),&exponent);double radius=std::ldexp(1.0,std::fmax(-150,exponent-26));
for(double y:{0.0,-0.0,double(float(radius/4)),double(float(-radius/4))}){cases++;double expected=float(float(x)+float(y));if(word<uint64_t>(candidate(x,y))!=word<uint64_t>(expected))return false;}return true;}
int main(){if(std::fesetround(FE_TONEAREST))return 2;
for(unsigned bits=0;bits<65536;bits++){if((bits&0x7c00)==0x7c00 || (bits&0x7fff)==0)continue;if(!check(word<_Float16>(uint16_t(bits))))return 1;}
for(unsigned e=1;e<=254;e++)for(unsigned mantissa:{0u,1u,2u,3u,4194304u,8388606u,8388607u})for(unsigned sign:{0u,0x80000000u})if(!check(word<float>(uint32_t(sign|(e<<23)|mantissa))))return 1;
for(unsigned mantissa:{1u,2u,3u,4194303u,4194304u,8388607u})for(unsigned sign:{0u,0x80000000u})if(!check(word<float>(uint32_t(sign|mantissa))))return 1;
std::printf("F32 cell native parity: cases=%u mismatches=0\\n",cases);}
''')
            subprocess.run(['clang++','-O3','-ffp-contract=off','-std=c++17',str(source),'-o',str(binary)],check=True,capture_output=True,text=True)
            result=subprocess.run([str(binary)],check=True,capture_output=True,text=True)
            self.assertIn('cases=268216 mismatches=0',result.stdout);print(result.stdout,end='')

    def test_midpoints_unknown_types_and_zero_crossing_do_not_authorize_elision(self):
        for s in [session(1,2,2**-25),session(-1,1,2**-50),session(1,2,2**-50,False)]:
            self.assertIn('R32',simplify_arithmetic('R32(X1+X2)',s))
        s=session(1,2,2**-28)
        for text in ('R32(X2+X1)','R32(X1-X2)'):
            self.assertEqual(simplify_arithmetic(text,s),'X1')

    def test_constant_casts_preserve_bits_including_signed_zero_and_nonfinite_payloads(self):
        self.assertTrue(is_signed_zero(syntax('0')))
        self.assertTrue(is_signed_zero(syntax('-0.0')))
        self.assertFalse(is_signed_zero(syntax('False')))
        self.assertFalse(is_signed_zero(syntax('9223372036854775808')))
        rng=random.Random(492)
        words=[struct.unpack('>Q',struct.pack('>d',struct.unpack('>e',struct.pack('>H',bits))[0]))[0] for bits in range(65536)]
        words.extend(rng.getrandbits(64) for _ in range(4096))
        words.extend([0x7ff0000000000001,0xfff0000000000001])
        for bits in words:
            original=ast.Call(func=ast.Name(id='Float64',ctx=ast.Load()),args=[ast.Constant(value=bits)],keywords=[])
            result=reduce_call(original,{})
            if isinstance(result,ast.Constant):
                self.assertEqual(struct.unpack('>Q',struct.pack('>d',result.value))[0],bits)
                cast=ast.Call(func=ast.Name(id='Bits64',ctx=ast.Load()),args=[result],keywords=[])
                self.assertEqual(reduce_call(cast,{}).value,bits)
            else:self.assertIs(result,original)
        for bits in (0,1<<63,0x3ff0000000000000,0x7fefffffffffffff):
            text=simplify_words(f'Float64({bits})',StringCompiler(),{})
            value=eval(text,{'__builtins__':{}})
            self.assertEqual(struct.unpack('>Q',struct.pack('>d',value))[0],bits)
        print(f'Constant cast parity: cases={len(words)} mismatches=0')


if __name__=='__main__':unittest.main()
