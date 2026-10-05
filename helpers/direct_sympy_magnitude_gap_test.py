"""Native validation of magnitude exclusions through residuals and storage."""
import subprocess,tempfile,unittest
from pathlib import Path
from fractions import Fraction as F
from direct_sympy_strings import syntax,StringCompiler,Domain
from direct_sympy_conversions import ConversionSession,FiniteSource

class MagnitudeGapTests(unittest.TestCase):
 def test_residual_storage_and_square_preserve_a_proved_central_gap(self):
  domains={'X1':Domain(F(-65504),F(65504),-24,False),'X2':Domain(F(-1,128),F(1,128),-24,False)}
  s=ConversionSession(StringCompiler(),domains,input_dtype='f16')
  s.completed[s.key(syntax('X1'))]=FiniteSource(-65504,65504,-24,.25)
  addition=s.bounds(syntax('X1+X2'));subtraction=s.bounds(syntax('X1-X2'))
  stored=s.bounds(syntax('R16(R32(X1+X2))'))
  self.assertGreater(addition.minimum_magnitude,.24);self.assertGreater(subtraction.minimum_magnitude,.24)
  self.assertGreater(stored.minimum_magnitude,.24)
  node=syntax('stored()');s.completed[s.key(node)]=stored;s.half_values.add(s.key(node));s.f32_values.add(s.key(node))
  square=s.bounds(syntax('stored()**2'));self.assertGreater(square.minimum,.05)
  self.assertEqual(s.bounds(syntax('X1*X2')).minimum_magnitude,0)
  source='''#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cfenv>
#include <cmath>
#include <initializer_list>
template<class T,class U>T word(U x){T y;std::memcpy(&y,&x,sizeof(y));return y;}
int main(){std::fesetround(FE_TONEAREST);unsigned cases=0,failures=0;
for(unsigned bits=0x3400;bits<0x7c00;bits++)for(unsigned sign:{0u,0x8000u})for(unsigned small:{0u,0x8000u,0x2000u,0xa000u}){
 double x=word<_Float16>(uint16_t(bits|sign)),y=word<_Float16>(uint16_t(small));
 double a=x+y,b=x-y,h=double(_Float16(float(a))),q=h*h;cases++;
 failures+=std::abs(a)<ADD||std::abs(b)<SUB||std::abs(h)<STORED||q<SQUARE;
}std::printf("Residual magnitude exclusion: cases=%u violations=%u\\n",cases,failures);return failures?1:0;}
'''
  for name,value in (('ADD',addition.minimum_magnitude),('SUB',subtraction.minimum_magnitude),('STORED',stored.minimum_magnitude),('SQUARE',square.minimum)):source=source.replace(name,repr(value))
  with tempfile.TemporaryDirectory()as directory:
   root=Path(directory);cpp=root/'gap.cpp';binary=root/'gap';cpp.write_text(source)
   subprocess.run(['clang++','-O3','-ffp-contract=off','-std=c++17',str(cpp),'-o',str(binary)],check=True,capture_output=True,text=True)
   result=subprocess.run([str(binary)],check=True,capture_output=True,text=True)
   self.assertIn('cases=147456 violations=0',result.stdout);print(result.stdout,end='')
  # If storage rounds the entire gap to zero, it cannot be retained.
  tiny=syntax('tiny()');s.completed[s.key(tiny)]=FiniteSource(-2**-25,2**-25,-26,2**-26)
  self.assertEqual(s.bounds(syntax('R16(tiny())')).minimum_magnitude,0)
  self.assertEqual(s.bounds(syntax('X1+(-X1)')).minimum_magnitude,0)

if __name__=='__main__':unittest.main()
