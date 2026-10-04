"""Native IEEE boundary proof for one-occurrence signed-binade conversion."""
import math,struct,subprocess,tempfile,unittest
from fractions import Fraction as F
from pathlib import Path
from unittest.mock import patch
from direct_sympy_conversions import FiniteSource,fixed_grid_conversion,lower_finite_conversion
from direct_sympy_conversions_test import cpp
from direct_sympy_strings import Domain,StringCompiler,syntax


class FixedGridTests(unittest.TestCase):
    def expression(self,kind,lo,hi):
        return lower_finite_conversion('X1',kind,FiniteSource(lo,hi),StringCompiler(),
            {'X1':Domain(F(lo),F(hi),-1074)})

    def test_only_one_signed_normal_binade_is_admitted(self):
        for kind,smallest,overflow in (('R16',2**-14,65520),('R32',2**-126,2**128-2**103)):
            for low,high in ((0,1),(-1,1),(smallest/2,smallest),(1,math.nextafter(2,math.inf)),
                (1,overflow),(overflow,overflow)):
                self.assertIsNone(fixed_grid_conversion(kind,FiniteSource(low,high)))
            for low,high in ((1,2),(-2,-1)):
                text=self.expression(kind,low,high)
                self.assertEqual(text.count('X1'),1)
                for residual in ('R16(','R32(','Bits64(','Piecewise('):self.assertNotIn(residual,text)
                with patch('direct_sympy_conversions.fixed_grid_conversion',return_value=None):
                    previous=self.expression(kind,low,high)
                self.assertEqual(previous.count('X1'),2)
                self.assertLess(len(text),len(previous))
        self.assertIsNone(fixed_grid_conversion('wrong',FiniteSource(1,2)))
        self.assertIsNone(fixed_grid_conversion('R16',None))
        domains={'X1':Domain(F(1),F(2),-10)}
        for kind,flags in (('R16',{'integer_word_exact':True}),('R32',{'no_odd_f32_ties':True})):
            args=('X1',kind,FiniteSource(1,2,-10),StringCompiler(),domains)
            result=lower_finite_conversion(*args,**flags)
            with patch('direct_sympy_conversions.fixed_grid_conversion',return_value=None):
                previous=lower_finite_conversion(*args,**flags)
            self.assertEqual(result,previous)
            self.assertEqual(result.count('X1'),1)

    def test_all_normalized_f32_midpoint_neighbours_and_signs(self):
        positive=cpp(syntax(self.expression('R32',1,2)))
        negative=cpp(syntax(self.expression('R32',-2,-1)))
        self.native('double candidate(double X1){return X1>0?'+positive+':'+negative+';}'+'''
int main(){std::fesetround(FE_TONEAREST);unsigned long long cases=0,fail=0;
 for(uint32_t b=0;b<8388608;b++){
  double midpoint=double(word<float>(uint32_t(0x3f800000|b)))+0x1p-24;
  for(double x:{std::nextafter(midpoint,0.0),midpoint,std::nextafter(midpoint,3.0)})
   for(double sign:{-1.0,1.0}){double value=sign*x;
    fail+=word<uint64_t>(candidate(value))!=word<uint64_t>(double(float(value)));cases++;}
 }
 std::printf("Fixed-grid F32 midpoint parity: cases=%llu mismatches=%llu\\n",cases,fail);return fail?1:0;}
''','Fixed-grid F32 midpoint parity: cases=50331648 mismatches=0')

    def test_half_midpoints_and_tandem_cells_across_every_normal_binade(self):
        declarations=[]
        for e in range(-14,16):
            lo=2.0**e;hi=min(2.0**(e+1),65519.0)
            for sign in (-1,1):
                a,b=(lo,hi) if sign>0 else (-hi,-lo)
                text=cpp(syntax(self.expression('R16',a,b)))
                declarations.append('double h_'+str(e+14)+'_'+str(sign+1)+'(double X1){return '+text+';}')
        functions=','.join('{'+str(e)+',h_'+str(e+14)+'_0,h_'+str(e+14)+'_2}' for e in range(-14,16))
        self.native('\n'.join(declarations)+'\nstruct Record{int e;double(*negative)(double);double(*positive)(double);};Record records[]={'+functions+'''};
int main(){std::fesetround(FE_TONEAREST);unsigned cases=0,fail=0;
 for(auto r:records)for(unsigned b=0;b<1024;b++){
  double step=std::ldexp(1.0,r.e-10),midpoint=std::ldexp(1.0,r.e)+(b+.5)*step;
  for(double x:{std::nextafter(midpoint,0.0),midpoint,std::nextafter(midpoint,INFINITY)}){
   if(x>65519.0)continue;
   for(int sign:{-1,1}){double v=x*sign,got=sign<0?r.negative(v):r.positive(v);
    fail+=word<uint64_t>(got)!=word<uint64_t>(double(_Float16(v)));cases++;
    // Test the actual F32-then-Half boundary as well, including F32
    // tie cells around Half midpoints, rather than silently dropping F32.
    double f=double(float(v));got=sign<0?r.negative(f):r.positive(f);
    fail+=word<uint64_t>(got)!=word<uint64_t>(double(_Float16(float(v))));
   }
  }
 }
 std::printf("Fixed-grid Half midpoint parity: cases=%u mismatches=%u\\n",cases,fail);return fail?1:0;}
''','Fixed-grid Half midpoint parity: cases=184314 mismatches=0')

    def test_scaled_f32_boundaries_and_endpoints(self):
        declarations=[];records=[]
        for e in range(-126,128):
            lo=2.0**e;hi=min(2.0**(e+1),float.fromhex('0x1.fffffep127'))
            if hi<lo:continue
            for sign in (-1,1):
                a,b=(lo,hi) if sign>0 else (-hi,-lo)
                name='f_'+str(e+126)+'_'+str(sign+1)
                declarations.append('double '+name+'(double X1){return '+cpp(syntax(self.expression('R32',a,b)))+';}')
                records.append('{'+str(e)+','+repr(sign)+','+name+'}')
        self.native('\n'.join(declarations)+'\nstruct Record{int e,sign;double(*f)(double);};Record records[]={'+','.join(records)+'''};
int main(){std::fesetround(FE_TONEAREST);unsigned cases=0,fail=0;
 for(auto r:records)for(uint32_t b:{0u,1u,0x3fffffu,0x400000u,0x7ffffeu,0x7fffffu}){
  double middle=std::ldexp(double(word<float>(uint32_t(0x3f800000|b)))+0x1p-24,r.e);
  double lo=std::ldexp(1.0,r.e),hi=std::fmin(std::ldexp(2.0,r.e),double(word<float>(uint32_t(0x7f7fffff))));
  for(double x:{lo,hi,std::nextafter(middle,0.0),middle,std::nextafter(middle,INFINITY)}){
   if(x>hi)continue;double v=x*r.sign;
   fail+=word<uint64_t>(r.f(v))!=word<uint64_t>(double(float(v)));cases++;
  }
 }
 std::printf("Fixed-grid scaled F32 parity: cases=%u mismatches=%u\\n",cases,fail);return fail?1:0;}
''','mismatches=0')

    def native(self,body,expected):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);source=root/'kernel.cpp';binary=root/'kernel'
            source.write_text('''#include <cmath>
#include <cstdint>
#include <cstring>
#include <cfenv>
#include <cstdio>
#include <initializer_list>
template<class T,class U>T word(U x){T y;std::memcpy(&y,&x,sizeof(y));return y;}
'''+body)
            compiled=subprocess.run(['clang++','-O3','-ffp-contract=off','-std=c++17',str(source),'-o',str(binary)],capture_output=True,text=True)
            self.assertEqual(compiled.returncode,0,compiled.stderr)
            result=subprocess.run([str(binary)],capture_output=True,text=True,timeout=120)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertIn(expected,result.stdout);print(result.stdout,end='')


if __name__=='__main__':unittest.main()
