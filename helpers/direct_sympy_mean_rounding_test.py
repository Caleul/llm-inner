"""Certify the emitted one-source mean conversion against native F32 storage."""
import struct,subprocess,tempfile,unittest
from pathlib import Path
from fractions import Fraction as F
from direct_sympy_strings import syntax,StringCompiler,Domain
from direct_sympy_conversions import ConversionSession
from direct_sympy_conversions_test import cpp

class MeanRoundingTests(unittest.TestCase):
 def test_every_admitted_f32_mean_operand_and_midpoint_barriers(self):
  epsilon=struct.unpack('f',struct.pack('f',1e-6))[0]
  domains={'X1':Domain(F(0),F(2)**32,-49,False)}
  session=ConversionSession(StringCompiler(),domains)
  session.f32_values.add(session.key(syntax('X1')))
  source='X1 + '+repr(epsilon)
  self.assertTrue(session.normal_word_rounding_stable(syntax(source)))
  self.assertFalse(session.encoded_word_is_exact_integer(syntax(source)))
  expression=session.close('R32('+source+')')
  self.assertEqual(expression.count('X1'),1)
  self.assertNotIn('R32(',expression)
  unknown=ConversionSession(StringCompiler(),domains)
  self.assertFalse(unknown.normal_word_rounding_stable(syntax(source)))
  self.assertFalse(session.normal_word_rounding_stable(syntax('X1 - '+repr(epsilon))))
  boundary=ConversionSession(StringCompiler(),{'X1':Domain(F(1),F(2),-23,True)})
  boundary.f32_values.add(boundary.key(syntax('X1')))
  self.assertTrue(boundary.normal_word_rounding_stable(syntax('X1+'+repr(2**-24))))
  for constant in (2**-24-2**-50,2**-24+2**-50):
   self.assertFalse(boundary.normal_word_rounding_stable(syntax('X1+'+repr(constant))))
  native='''#include <cstdint>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <cfenv>
template<class T,class U>T word(U x){T y;std::memcpy(&y,&x,sizeof(y));return y;}
double compiled(double X1){return EXPR;}
int main(){if(std::fesetround(FE_TONEAREST))return 2;uint64_t cases=0,failures=0;
for(unsigned exponent=78;exponent<159;exponent++){
 unsigned shift=exponent<101?101-exponent:0;unsigned step=1u<<shift;
 for(unsigned mantissa=0;mantissa<8388608;mantissa+=step){
  double x=word<float>(uint32_t((exponent<<23)|mantissa));
  double expected=double(float(x+EPS));failures+=word<uint64_t>(compiled(x))!=word<uint64_t>(expected);cases++;
 }}
 for(double x:{0.0,-0.0,4294967296.0}){failures+=word<uint64_t>(compiled(x))!=word<uint64_t>(double(float(x+EPS)));cases++;}
 std::printf("One-source mean rounding: cases=%llu mismatches=%llu\\n",(unsigned long long)cases,(unsigned long long)failures);return failures?1:0;}
'''.replace('EXPR',cpp(syntax(expression))).replace('EPS',repr(epsilon)).replace('#include <cfenv>','#include <cfenv>\n#include <initializer_list>')
  with tempfile.TemporaryDirectory()as directory:
   root=Path(directory);file=root/'mean.cpp';binary=root/'mean';file.write_text(native)
   subprocess.run(['clang++','-O3','-ffp-contract=off','-std=c++17',str(file),'-o',str(binary)],check=True,capture_output=True,text=True)
   result=subprocess.run([str(binary)],check=True,capture_output=True,text=True,timeout=120)
   self.assertIn('mismatches=0',result.stdout);print(result.stdout,end='')

if __name__=='__main__':unittest.main()
