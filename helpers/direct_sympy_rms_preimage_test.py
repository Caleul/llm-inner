"""Validate backward whole-vector RMS bounds with the original storage order."""
import struct,subprocess,tempfile,unittest
from fractions import Fraction as F
from pathlib import Path
from types import SimpleNamespace as NS
from direct_sympy_strings import StringCompiler,Domain
from direct_sympy_conversions import FiniteSource
from direct_sympy_projection_constraints import rms_output_source_bounds,norm_error_bound

class RMSPreimageTests(unittest.TestCase):
 def test_tiny_output_caps_use_relative_loss_and_preserve_all_zero_signs(self):
  compiler=StringCompiler();unit=2**-24
  domains={n:Domain(F(-65504),F(65504),-24,False)for n in ('X1','X2')}
  model=NS(compiler=compiler,domains=domains)
  epsilon=F(struct.unpack('f',struct.pack('f',1e-6))[0])
  vector={'width':2,'sources':('X1','X2'),'sourceBounds':(FiniteSource(-65504,65504,-24),)*2,
   'components':{0:'CompileValue0()',1:'CompileValue1()'},'gamma':(F(-1),F(1)),
   'epsilon':epsilon,'context':compiler.context(domains),'roundingError':norm_error_bound(2,epsilon)}
  model.norm_vectors={'test':vector};registry=NS(model=model)
  proofs={name:FiniteSource(-65504,65504,-24)for name in domains}
  proofs.update({i:FiniteSource(-unit,unit,-24)for i in (0,1)})
  bound=rms_output_source_bounds(registry,proofs)['X1']
  self.assertLess(bound.maximum,unit);self.assertEqual(bound.minimum,-bound.maximum)
  proofs.update({i:FiniteSource(0,0,-24)for i in (0,1)})
  zero=rms_output_source_bounds(registry,proofs)['X1']
  self.assertLess(zero.maximum,bound.maximum)
  self.assertEqual(zero.minimum_magnitude,0)
  from direct_sympy_recipe_bounds import propagate_recipe_bounds
  from direct_sympy_conversions import ConversionSession
  model.conversions=ConversionSession(compiler,domains,input_dtype='f16')
  registry.definition_proofs=[(proofs[i],'half',False,None)for i in (0,1)]
  registry.definition_recipes={}
  inferred=propagate_recipe_bounds(registry,{0:proofs[0],1:proofs[1]})
  self.assertEqual(inferred['X1'],FiniteSource(0,0,-24))
  self.assertEqual(inferred['X2'],FiniteSource(0,0,-24))
  # Every finite Half value on either axis, plus deterministic mixed pairs,
  # spans normal/subnormal bins, both signs, large magnitudes and all zeros.
  source='''#include <cstdint>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <cfenv>
#include <initializer_list>
template<class T,class U>T word(U x){T y;std::memcpy(&y,&x,sizeof(y));return y;}
int main(){if(std::fesetround(FE_TONEAREST))return 2;uint64_t cases=0,selected=0,failures=0;
auto test=[&](uint16_t a,uint16_t b){double x=word<_Float16>(a),y=word<_Float16>(b);
 float mean=float(double(float(double(float(x*x+y*y))/2))+EPS);
 float inverse=float(1.0/double(float(std::sqrt(double(mean)))));
 double h0=-double(_Float16(float(x*inverse))),h1=double(_Float16(float(y*inverse)));cases++;
 if(std::abs(h0)<=UNIT&&std::abs(h1)<=UNIT){selected++;failures+=std::abs(x)>CAP||std::abs(y)>CAP;}
};
for(unsigned i=0;i<=0x7bff;i++)for(unsigned sign:{0u,0x8000u})for(unsigned zero:{0u,0x8000u}){
 test(uint16_t(i|sign),uint16_t(zero));test(uint16_t(zero),uint16_t(i|sign));
}
for(unsigned a=0;a<256;a++)for(unsigned b=0;b<256;b++){
 unsigned x=(a*997u)%0x7c00u,y=(b*577u)%0x7c00u;
 test(uint16_t(x|((a&1)?0x8000u:0)),uint16_t(y|((b&1)?0x8000u:0)));
}
std::printf("Relative RMS tiny-output bound: cases=%llu selected=%llu violations=%llu\\n",(unsigned long long)cases,(unsigned long long)selected,(unsigned long long)failures);return failures||!selected?1:0;}
'''.replace('EPS',repr(float(epsilon))).replace('UNIT',repr(unit)).replace('CAP',repr(bound.maximum))
  with tempfile.TemporaryDirectory()as directory:
   root=Path(directory);cpp=root/'relative.cpp';binary=root/'relative';cpp.write_text(source)
   subprocess.run(['clang++','-O3','-ffp-contract=off','-std=c++17',str(cpp),'-o',str(binary)],check=True,capture_output=True,text=True)
   tested=subprocess.run([str(binary)],check=True,capture_output=True,text=True,timeout=120)
   self.assertIn('cases=319488 selected=9 violations=0',tested.stdout);print(tested.stdout,end='')

 def test_output_norm_limits_actual_sources_without_erasing_signs(self):
  compiler=StringCompiler();domains={n:Domain(F(-65504),F(65504),-24,False)for n in ('X1','X2')}
  model=NS(compiler=compiler,domains=domains);epsilon=F(struct.unpack('f',struct.pack('f',1e-6))[0])
  vector={'width':2,'sources':('X1','X2'),'sourceBounds':(FiniteSource(-65504,65504,-24),)*2,
   'components':{0:'CompileValue0()',1:'CompileValue1()'},'gamma':(F(1),F(-1)),
   'epsilon':epsilon,'context':compiler.context(domains),'roundingError':norm_error_bound(2,epsilon)}
  model.norm_vectors={'test':vector};registry=NS(model=model)
  proofs={'X1':FiniteSource(-65504,65504,-24),'X2':FiniteSource(-65504,65504,-24),
   0:FiniteSource(-1/64,1/64,-24),1:FiniteSource(-1/32,1/32,-24)}
  result=rms_output_source_bounds(registry,proofs)
  self.assertEqual(set(result),{'X1','X2'});self.assertLess(result['X1'].maximum,2**-14)
  self.assertEqual(result['X1'].minimum,-result['X1'].maximum)
  self.assertEqual(result['X1'].minimum_magnitude,0)
  source='''#include <cstdint>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <cfenv>
#include <initializer_list>
template<class T,class U>T word(U x){T y;std::memcpy(&y,&x,sizeof(y));return y;}
int main(){if(std::fesetround(FE_TONEAREST))return 2;uint64_t cases=0,selected=0,failures=0;
for(unsigned a=0;a<=0xc00;a++)for(unsigned sa:{0u,0x8000u})for(unsigned b=0;b<=0xc00;b++)for(unsigned sb:{0u,0x8000u}){
 double x=word<_Float16>(uint16_t(a|sa)),y=word<_Float16>(uint16_t(b|sb));
 float sum=float(x*x+y*y);float mean=float(double(float(double(sum)/2))+EPS);
 float inverse=float(1.0/double(float(std::sqrt(double(mean)))));
 double h0=double(_Float16(float(x*inverse))),h1=-double(_Float16(float(y*inverse)));cases++;
 if(std::abs(h0)<=0.015625&&std::abs(h1)<=0.03125){selected++;failures+=std::abs(x)>CAP||std::abs(y)>CAP;}
}std::printf("Backward RMS vector bound: cases=%llu selected=%llu violations=%llu\\n",(unsigned long long)cases,(unsigned long long)selected,(unsigned long long)failures);return failures||!selected?1:0;}
'''.replace('EPS',repr(float(epsilon))).replace('CAP',repr(result['X1'].maximum))
  with tempfile.TemporaryDirectory()as directory:
   root=Path(directory);cpp=root/'rms.cpp';binary=root/'rms';cpp.write_text(source)
   subprocess.run(['clang++','-O3','-ffp-contract=off','-std=c++17',str(cpp),'-o',str(binary)],check=True,capture_output=True,text=True)
   tested=subprocess.run([str(binary)],check=True,capture_output=True,text=True,timeout=120)
   self.assertIn('cases=37773316',tested.stdout);self.assertIn('violations=0',tested.stdout);print(tested.stdout,end='')
  # The independent cap must not erase an older nonzero source gap:
  # intersection belongs to the caller and proves an impossible prefix.
  from direct_sympy_recipe_bounds import propagate_recipe_bounds
  from direct_sympy_conversions import ConversionSession
  model.conversions=ConversionSession(compiler,domains,input_dtype='f16')
  registry.definition_proofs=[(proofs[i],'half',False,None)for i in range(2)]
  registry.definition_recipes={}
  constrained=dict(proofs);constrained['X1']=FiniteSource(-65504,65504,-24,1)
  cap=rms_output_source_bounds(registry,constrained)['X1']
  self.assertLess(cap.maximum,1);self.assertEqual(cap.minimum_magnitude,0)
  self.assertIsNone(propagate_recipe_bounds(registry,{'X1':constrained['X1']}))
  vector['gamma']=(F(2),F(1));self.assertEqual(rms_output_source_bounds(registry,proofs),{})
  vector['gamma']=(F(1),F(1));vector['context']='foreign';self.assertEqual(rms_output_source_bounds(registry,proofs),{})
  vector['context']=compiler.context(domains);proofs[0]=FiniteSource(-2,2,-24)
  self.assertEqual(rms_output_source_bounds(registry,proofs),{})

if __name__=='__main__':unittest.main()
