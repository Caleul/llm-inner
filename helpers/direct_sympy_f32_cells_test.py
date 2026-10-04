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
from direct_sympy_conversions import ConversionSession
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
