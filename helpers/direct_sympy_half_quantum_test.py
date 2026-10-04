"""Interval-specific Half grids and exact elimination of redundant F32 sums."""
from dataclasses import replace
from fractions import Fraction as F
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from direct_sympy_strings import Domain,StringCompiler,syntax
from direct_sympy_conversions import ConversionSession
from direct_sympy_conversions_test import cpp


def make(low,high,q=-1074):
    return ConversionSession(StringCompiler(),{name:Domain(F(low),F(high),q)
        for name in ('X1','X2')})


class HalfQuantumTests(unittest.TestCase):
    def test_every_finite_half_value_obeys_its_interval_certificate(self):
        cases=0
        for exponent in range(31):
            low=struct.unpack('e',struct.pack('H',exponent<<10))[0]
            high=struct.unpack('e',struct.pack('H',(exponent<<10)|1023))[0]
            for sign in (1,-1):
                s=make(low if sign>0 else -high,high if sign>0 else -low)
                source=s.bounds(syntax('R16(X1)'))
                self.assertEqual(source.quantum,max(-24,exponent-25))
                for mantissa in range(1024):
                    bits=(exponent<<10)|mantissa|(0x8000 if sign<0 else 0)
                    value=struct.unpack('e',struct.pack('H',bits))[0]
                    self.assertEqual((F(value)/(F(2)**source.quantum)).denominator,1)
                    cases+=1
        self.assertEqual(cases,63488);print(f'Half interval grid proof: cases={cases} violations=0')

    def test_zero_crossings_coarse_source_grids_and_subnormal_cells(self):
        self.assertEqual(make(-1,1).bounds(syntax('R16(X1)')).quantum,-24)
        self.assertEqual(make(1,2).bounds(syntax('R16(X1)')).quantum,-10)
        self.assertEqual(make(-2,-1).bounds(syntax('R16(X1)')).quantum,-10)
        self.assertEqual(make(0,4,-3).bounds(syntax('R16(X1)')).quantum,-3)
        s=make(2**-24,2**-14,-24)
        s.half_values.add(s.key(syntax('X1')))
        self.assertEqual(s.half_cell_radius(syntax('X1')),2**-25)
        s=make(1,2,-24);s.half_values.add(s.key(syntax('X1')))
        self.assertEqual(s.half_cell_radius(syntax('X1')),2**-12)

    def test_actual_emitted_half_sum_is_smaller_and_has_native_parity(self):
        expression='R32(R16(X1)+R16(X2))'
        current=make(1,2,-52);result=current.close(expression)
        legacy=make(1,2,-52);bounds=legacy.bounds
        def conservative(node):
            source=bounds(node)
            if source is not None and getattr(getattr(node,'func',None),'id',None)=='R16':
                return replace(source,quantum=-24)
            return source
        with patch.object(legacy,'bounds',side_effect=conservative):baseline=legacy.close(expression)
        self.assertLess(len(result),len(baseline))
        for name in ('R16(','R32(','CompileValue','CASNumericRegion'):self.assertNotIn(name,result)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path=root/'sum.expr';path.write_text(result)
            source=root/'sum.cpp';binary=root/'sum'
            source.write_text('''#include <cstdint>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cfenv>
template<class T,class U>T word(U value){T result;std::memcpy(&result,&value,sizeof(result));return result;}
double candidate(double X1,double X2){return '''+cpp(syntax(path.read_text()))+''';}
int main(){if(std::fesetround(FE_TONEAREST))return 2;unsigned cases=0;
for(unsigned a=0x3c00;a<=0x4000;a++)for(unsigned b=0x3c00;b<=0x4000;b++){
double x=word<_Float16>(uint16_t(a)),y=word<_Float16>(uint16_t(b));cases++;
double expected=float(float(_Float16(x))+float(_Float16(y)));
if(word<uint64_t>(candidate(x,y))!=word<uint64_t>(expected))return 1;
}std::printf("Half interval sum native parity: cases=%u mismatches=0\\n",cases);}
''')
            built=subprocess.run(['clang++','-O3','-ffp-contract=off','-std=c++17',str(source),'-o',str(binary)],capture_output=True,text=True)
            self.assertEqual(built.returncode,0,built.stderr)
            outcome=subprocess.run([str(binary)],check=True,capture_output=True,text=True)
            self.assertIn('cases=1050625 mismatches=0',outcome.stdout);print(outcome.stdout,end='')
        print(f'Half interval expression: characters={len(baseline)}->{len(result)}')


if __name__=='__main__':unittest.main()
