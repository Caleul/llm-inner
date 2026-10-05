import sys
import subprocess
import tempfile
import unittest
from fractions import Fraction as F
from pathlib import Path
from direct_sympy_strings import StringCompiler,Domain,syntax
from direct_sympy_conversions import ConversionSession
from direct_sympy_conversions_test import cpp
from direct_sympy_sqrt import supported,CONSTANT_TERM,PARTIAL_FRACTIONS,NARROW_CONSTANT_TERM,NARROW_PARTIAL_FRACTIONS
import struct


class SqrtWordTests(unittest.TestCase):
    def test_narrow_two_source_kernel_preserves_every_admitted_mantissa_and_scale(self):
        import mpmath as mp
        import sympy as sp
        with mp.workdps(100):
            n=2;z=sp.Symbol('z');center=mp.mpf('1.046875');radius=mp.mpf('0.015625')
            nodes=[radius*mp.cos(mp.pi*i/(2*n))for i in range(2*n+1)]
            matrix=mp.matrix([[x**j for j in range(n+1)]+[-mp.sqrt(x+center)*x**j for j in range(1,n+1)]for x in nodes])
            solution=mp.lu_solve(matrix,mp.matrix([mp.sqrt(x+center)for x in nodes]))
            numerator=[solution[j]for j in range(n+1)];denominator=[mp.mpf(1)]+[solution[n+j]for j in range(1,n+1)]
            self.assertEqual(NARROW_CONSTANT_TERM,float(numerator[-1]/denominator[-1]))
            roots=sp.nroots(sp.Poly.from_list([sp.Float(mp.nstr(c,100),100)for c in reversed(denominator)],z),n=70,maxsteps=500)
            fractions=[]
            for root in roots:
                r=mp.mpf(str(root))
                residue=sum(numerator[i]*r**i for i in range(n+1))/sum(i*denominator[i]*r**(i-1)for i in range(1,n+1))
                fractions.append((float(residue),float(-r)))
            fractions.sort(key=lambda row:abs(row[0]/row[1]),reverse=True)
            self.assertEqual(tuple(fractions),NARROW_PARTIAL_FRACTIONS)
        expressions=[];ranges=[]
        for exponent in (-140,-126,-25,-1,0,1,24,127):
            low,high=F(33,32)*F(2)**exponent,F(17,16)*F(2)**exponent
            session=ConversionSession(StringCompiler(),{'X1':Domain(low,high,max(-149,exponent-23),True)})
            session.f32_values.add(session.key(syntax('X1')))
            expression=session.close('R32(sqrt(X1))')
            self.assertEqual(expression.count('X1'),2)
            self.assertEqual(session.narrow_square_roots_closed,1)
            self.assertNotIn('sqrt(',expression);self.assertNotIn('R32(',expression)
            expressions.append(expression)
            ranges.append(tuple(struct.unpack('I',struct.pack('f',float(x)))[0]for x in (low,high)))
        for low,high in ((F(33,32)-F(2)**-23,F(17,16)),(F(33,32),F(17,16)+F(2)**-23),(F(33,32),F(33,16))):
            other=ConversionSession(StringCompiler(),{'X1':Domain(low,high,-23,True)})
            other.f32_values.add(other.key(syntax('X1')))
            text=other.close('R32(sqrt(X1))')
            self.assertEqual(text.count('X1'),5 if high<2 else 7)
            self.assertEqual(getattr(other,'narrow_square_roots_closed',0),0)
        with tempfile.TemporaryDirectory()as directory:
            root=Path(directory);source=root/'narrow.cpp';binary=root/'narrow'
            functions='\n'.join('double f'+str(i)+'(double X1){return '+cpp(syntax(text))+';}'for i,text in enumerate(expressions))
            approximation=repr(NARROW_CONSTANT_TERM)
            for residue,pole in NARROW_PARTIAL_FRACTIONS:
                approximation=f'(({approximation})+({residue!r}/(z+{pole!r})))'
            source.write_text('#include <cstdint>\n#include <cstring>\n#include <cmath>\n#include <cstdio>\n#include <cfenv>\ntemplate<class T,class U>T word(U x){T y;std::memcpy(&y,&x,sizeof(y));return y;}\n'+functions+
                '\ndouble polynomial(double m,unsigned p){double z=m-1.046875;return ('+approximation+')*(p?1.4142135623730951:1.0);}\n'+
                'int main(){if(std::fesetround(FE_TONEAREST))return 2;uint64_t cases=0,failures=0,ties=0;double (*functions[])(double)={f0,f1,f2,f3,f4,f5,f6,f7};unsigned ranges[][2]={'+','.join('{'+str(a)+','+str(b)+'}'for a,b in ranges)+'};'+'''
for(unsigned i=0;i<8;i++)for(unsigned b=ranges[i][0];b<=ranges[i][1];b++){
 double x=word<float>(uint32_t(b));double actual=functions[i](x),expected=double(float(std::sqrt(x)));
 failures+=word<uint64_t>(actual)!=word<uint64_t>(expected);cases++;
}std::printf("Narrow sqrt emitted parity: cases=%llu mismatches=%llu\\n",(unsigned long long)cases,(unsigned long long)failures);
uint64_t normalized=0;for(unsigned m=262144;m<=524288;m++)for(unsigned p=0;p<2;p++){
 double x=word<float>(uint32_t((127u<<23)|m));double raw=polynomial(x,p);
 failures+=word<uint32_t>(float(raw))!=word<uint32_t>(float(std::sqrt(x*(p?2.0:1.0))));
 ties+=(word<uint64_t>(raw)&UINT64_C(536870911))==UINT64_C(268435456);normalized++;
}std::printf("Narrow sqrt mantissa certificate: cases=%llu mismatches=%llu midpoints=%llu\\n",(unsigned long long)normalized,(unsigned long long)failures,(unsigned long long)ties);return failures||ties?1:0;}
''')
            subprocess.run(['clang++','-O3','-ffp-contract=off','-std=c++17',str(source),'-o',str(binary)],check=True,capture_output=True,text=True)
            tested=subprocess.run([str(binary)],check=True,capture_output=True,text=True,timeout=120)
            self.assertIn('cases=1835032 mismatches=0',tested.stdout)
            self.assertIn('cases=524290 mismatches=0 midpoints=0',tested.stdout);print(tested.stdout,end='')

    def test_fixed_scale_context_removes_only_proved_source_occurrences(self):
        expressions=[]
        for exponent in (-149,-126,-25,-1,0,1,24,127):
            low=F(2)**exponent
            high=low if exponent==-149 else low*(2-F(2)**-23)
            session=ConversionSession(StringCompiler(),{'X1':Domain(low,high,max(-149,exponent-23),True)})
            session.f32_values.add(session.key(syntax('X1')))
            expression=session.close('R32(sqrt(X1))')
            if exponent!=-149:self.assertEqual(expression.count('X1'),5)
            expressions.append((exponent,expression))
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'fixed.cpp';binary=root/'fixed'
            functions='\n'.join('double f'+str(i)+'(double X1){return '+cpp(syntax(text))+';}'for i,(_,text)in enumerate(expressions))
            source.write_text('#include <cstdint>\n#include <cstring>\n#include <cmath>\n#include <cstdio>\n#include <cfenv>\ntemplate<class T,class U>T word(U x){T y;std::memcpy(&y,&x,sizeof(y));return y;}\n'+functions+'''
int main(){std::fesetround(FE_TONEAREST);unsigned long long cases=0,mismatches=0;
double (*functions[])(double)={f0,f1,f2,f3,f4,f5,f6,f7};int exponents[]={-149,-126,-25,-1,0,1,24,127};
for(unsigned i=0;i<8;i++)for(unsigned m=0;m<(i==0?1u:8388608u);m++){
 float x=i==0?word<float>(uint32_t(1)):word<float>(uint32_t(((exponents[i]+127)<<23)|m));
 double actual=functions[i](double(x)),expected=double(float(std::sqrt(double(x))));
 mismatches+=word<uint64_t>(actual)!=word<uint64_t>(expected);cases++;
}std::printf("Fixed-scale sqrt parity: cases=%llu mismatches=%llu\\n",cases,mismatches);return mismatches?1:0;}
''')
            built=subprocess.run(['clang++','-O3','-ffp-contract=off','-std=c++17',str(source),'-o',str(binary)],capture_output=True,text=True)
            self.assertEqual(built.returncode,0,built.stderr)
            tested=subprocess.run([str(binary)],capture_output=True,text=True,timeout=120)
            self.assertEqual(tested.returncode,0,tested.stdout+tested.stderr)
            self.assertIn('cases=58720257 mismatches=0',tested.stdout);print(tested.stdout,end='')

    def test_f32_boundary_matches_all_mantissas_parities_subnormals_and_exponents(self):
        # Reproduce every coefficient from the interpolation contract.
        import mpmath as mp
        import sympy as sp
        with mp.workdps(100):
            n=5;z=sp.Symbol('z')
            nodes=[mp.mpf('0.5')*mp.cos(mp.pi*i/(2*n)) for i in range(2*n+1)]
            matrix=mp.matrix([[x**j for j in range(n+1)]+[-mp.sqrt(x+mp.mpf('1.5'))*x**j for j in range(1,n+1)] for x in nodes])
            solution=mp.lu_solve(matrix,mp.matrix([mp.sqrt(x+mp.mpf('1.5')) for x in nodes]))
            coefficients=([solution[j] for j in range(n+1)],[mp.mpf(1)]+[solution[n+j] for j in range(1,n+1)])
            numerator,denominator=coefficients
            self.assertEqual(CONSTANT_TERM,float(numerator[-1]/denominator[-1]))
            roots=sp.nroots(sp.Poly.from_list([sp.Float(mp.nstr(c,100),100) for c in reversed(denominator)],z),n=70,maxsteps=500)
            fractions=[]
            for root in roots:
                r=mp.mpf(str(root))
                residue=sum(numerator[i]*r**i for i in range(6))/sum(i*denominator[i]*r**(i-1) for i in range(1,6))
                fractions.append((float(residue),float(-r)))
            fractions.sort(key=lambda row:abs(row[0]/row[1]),reverse=True)
            self.assertEqual(tuple(fractions),PARTIAL_FRACTIONS)
        session=ConversionSession(StringCompiler(),{'X1':Domain(F(2)**-149,F(3.4028234663852886e38),-149,True)})
        session.f32_values.add(session.key(syntax('X1')))
        result=session.close('R32(sqrt(X1))')
        self.assertEqual(session.square_roots_closed,1)
        self.assertEqual(result.count("X1"),7)
        for name in ('sqrt(','R32(','R16(','CASNumericRegion','X999999997'):self.assertNotIn(name,result)
        self.assertEqual(session.value_kind(syntax(result)),'f32')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);expression=root/'sqrt.work.expr';expression.write_text(result+'\n')
            source=root/'root.cpp';binary=root/'root'
            approximation=repr(CONSTANT_TERM)
            for residue,pole in PARTIAL_FRACTIONS:approximation='('+approximation+'+('+repr(residue)+'/(z+'+repr(pole)+')))'
            polynomial='double polynomial(double x){uint64_t raw=word<uint64_t>(x);double m=word<double>((raw&UINT64_C(4503599627370495))|UINT64_C(4607182418800017408));double z=m-1.5;return ('+approximation+')*(1.0+double(((raw>>52)+1)&1)*0.4142135623730951); }\n'
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
