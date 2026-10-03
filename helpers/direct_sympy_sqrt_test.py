import sys
import subprocess
import tempfile
import unittest
from fractions import Fraction as F
from pathlib import Path
from direct_sympy_strings import StringCompiler,Domain,syntax
from direct_sympy_conversions import ConversionSession
from direct_sympy_conversions_test import cpp
from direct_sympy_sqrt import supported,NUMERATOR,DENOMINATOR


class SqrtWordTests(unittest.TestCase):
    def test_f32_boundary_matches_all_mantissas_parities_subnormals_and_exponents(self):
        session=ConversionSession(StringCompiler(),{'X1':Domain(F(2)**-149,F(3.4028234663852886e38),-149,True)})
        session.f32_values.add(session.key(syntax('X1')))
        result=session.close('R32(sqrt(X1))')
        self.assertEqual(session.square_roots_closed,1)
        self.assertEqual(result.count("X1"),13)
        for name in ('sqrt(','R32(','R16(','CASNumericRegion','X999999997'):self.assertNotIn(name,result)
        self.assertEqual(session.value_kind(syntax(result)),'f32')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);expression=root/'sqrt.work.expr';expression.write_text(result+'\n')
            source=root/'root.cpp';binary=root/'root'
            polynomial='const double numerator[]={'+','.join(repr(c) for c in NUMERATOR)+'},denominator[]={'+','.join(repr(c) for c in DENOMINATOR)+'};\ndouble polynomial(double x){uint64_t raw=word<uint64_t>(x);double m=word<double>((raw&UINT64_C(4503599627370495))|UINT64_C(4607182418800017408));double z=m-1.5,a=numerator[0],b=denominator[0];for(unsigned j=1;j<sizeof(numerator)/sizeof(double);j++)a=numerator[j]+z*a;for(unsigned j=1;j<sizeof(denominator)/sizeof(double);j++)b=denominator[j]+z*b;double seed=a/b;return (0.5*(seed+m/seed))*(1.0+double(((raw>>52)+1)&1)*0.4142135623730951);}\n'
            source.write_text('#include <cstdint>\n#include <cstring>\n#include <cmath>\n#include <cstdio>\n#include <initializer_list>\ntemplate<class T,class U>T word(U value){T result;std::memcpy(&result,&value,sizeof(result));return result;}\n'+polynomial+'double compiled(double X1){return '+cpp(syntax(expression.read_text()))+';}\n'+'''int main(){unsigned cases=0,mismatches=0,oddTies=0,evenTies=0;for(unsigned b=0;b<8388608;b++){for(unsigned e=127;e<129;e++){double x=word<float>(uint32_t((e<<23)|b));uint64_t p=word<uint64_t>(polynomial(x));if((p&UINT64_C(536870911))==UINT64_C(268435456)){if((p>>29)&1)oddTies++;else evenTies++;}mismatches+=word<uint64_t>(compiled(x))!=word<uint64_t>(double(float(std::sqrt(x))));cases++;}}for(unsigned e=1;e<=254;e++){for(unsigned b:{0u,1u,2u,4194303u,4194304u,8388606u,8388607u}){double x=word<float>(uint32_t((e<<23)|b));uint64_t p=word<uint64_t>(polynomial(x));if((p&UINT64_C(536870911))==UINT64_C(268435456)){if((p>>29)&1)oddTies++;else evenTies++;}mismatches+=word<uint64_t>(compiled(x))!=word<uint64_t>(double(float(std::sqrt(x))));cases++;}}for(unsigned b=1;b<8388608;b++){double x=word<float>(uint32_t(b));mismatches+=word<uint64_t>(compiled(x))!=word<uint64_t>(double(float(std::sqrt(x))));cases++;}std::printf("F32 sqrt emitted word parity: cases=%u mismatches=%u oddTies=%u evenTies=%u\\n",cases,mismatches,oddTies,evenTies);return (mismatches||oddTies||evenTies)?1:0;}''')
            build=subprocess.run(['clang++','-O3','-ffp-contract=off','-std=c++17',str(source),'-o',str(binary)],capture_output=True,text=True)
            self.assertEqual(build.returncode,0,build.stderr)
            run=subprocess.run([str(binary)],capture_output=True,text=True,timeout=120)
            self.assertEqual(run.returncode,0,run.stdout+run.stderr)
            self.assertIn('cases=25167601 mismatches=0 oddTies=0 evenTies=0',run.stdout);print(run.stdout,end='')
        # Zero/negative/infinite/non-F32 domains cannot use this certificate.
        for minimum,maximum in ((F(0),F(2)),(F(-1),F(2)),(F(1),F(2)**128)):
            other=ConversionSession(StringCompiler(),{'X1':Domain(minimum,maximum,-149,False)})
            other.f32_values.add(other.key(syntax('X1')))
            self.assertFalse(supported('X1',other))
        untyped=ConversionSession(StringCompiler(),session.domains)
        self.assertFalse(supported('X1',untyped))
        # A bare F64 sqrt has a different observable boundary; retain it.
        bare=ConversionSession(StringCompiler(),session.domains)
        bare.f32_values.add(bare.key(syntax('X1')))
        self.assertIn('sqrt(',bare.close('sqrt(X1)'))
        self.assertEqual(bare.square_roots_closed,0)


if __name__=='__main__':unittest.main()
