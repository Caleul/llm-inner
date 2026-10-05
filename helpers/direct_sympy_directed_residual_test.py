"""Signed composition encloses original rounding and residual cell boundaries."""
import os,struct,subprocess,tempfile,unittest
from pathlib import Path
from direct_sympy_checkpoint import CheckpointStrings
from direct_sympy_strings import StringCompiler
from direct_sympy_input_partitions import interval
from direct_sympy_streaming_literals import streaming_literals
from direct_sympy_layer_bounds import composed_dot_intervals

class DirectedResidualTests(unittest.TestCase):
 def test_unsupported_intervals_and_work_limits_provide_no_certificate(self):
  class Model:
   def shape(self,name):return [2,2]
   def weight(self,name,row,col):return float(row==col)
  model=Model();inputs=[(-2.,-1.),(.25,.5)]
  self.assertIsNotNone(composed_dot_intervals(model,'a','b',inputs,[0,1]))
  for own,mapping,budget in ((inputs,[0,1],0),(inputs,[0,2],100),([(float('nan'),1),(.25,.5)],[0,1],100),([(-1.,-2.),(.25,.5)],[0,1],100)):
   self.assertIsNone(composed_dot_intervals(model,'a','b',own,mapping,max_products=budget))

 @unittest.skipUnless(os.environ.get('LLM_INNER_DIRECT_JSON_CHECKPOINT'),'Checkpoint required')
 def test_all_mixed_half_pairs_preserve_signed_updates_and_residual_cells(self):
  checkpoint=os.environ['LLM_INNER_DIRECT_JSON_CHECKPOINT'];prefix='model.layers.0.'
  with CheckpointStrings(checkpoint,StringCompiler(),input_domains={'X1':interval(-2,-.03125),'X2':interval(.031280517578125,1)}) as m:
   with streaming_literals(m):
    m.coordinate(2)
    self.assertEqual(len(m.norm_vectors),2)
    self.assertEqual(len([x for x in m.elided_updates if x[1]=='mlp']),2)
    bounds=composed_dot_intervals(m,prefix+'self_attn.v_proj.weight',prefix+'self_attn.o_proj.weight',m.norm_intervals(prefix+'input_layernorm.weight',lambda i:'X'+str(i+1)),list(range(m.width)))
    radii=m.directed_residual_radii[prefix]
   weights=[float(m.weight(prefix+'self_attn.'+name+'.weight',i,j)) for name in ('v_proj','o_proj') for i in range(2) for j in range(2)]
   eps=struct.unpack('f',struct.pack('f',m.config['rms_norm_eps']))[0]
  self.assertLess(bounds[0][1],0);self.assertGreater(bounds[1][0],0)
  with tempfile.TemporaryDirectory() as directory:
   root=Path(directory);source=root/'proof.cpp';binary=root/'proof'
   source.write_text('''#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cfenv>
#include <cstdio>
template<class T,class U>T word(U x){T y;std::memcpy(&y,&x,sizeof(y));return y;}
'''+''.join('double '+n+'[]={'+','.join(map(repr,v))+'};\n' for n,v in [('w',weights),('lo',[b[0] for b in bounds]),('hi',[b[1] for b in bounds]),('radii',radii)])+
    'double eps='+repr(eps)+';\n'+r'''
void norm(double x,double y,double &a,double &b){
 float mean=float(float(float(x*x)+float(y*y))/2.0f+float(eps));
 float inverse=float(1.0/double(float(std::sqrt(double(mean)))));
 a=double(_Float16(float(x*inverse)));b=double(_Float16(float(y*inverse)));}
double projection(double a,double b,int i){float lanes[4]={};
 lanes[0]=float(float(a*w[i])+0.0f);lanes[1]=float(float(b*w[i+1])+0.0f);
 return double(_Float16(float(float(lanes[0]+lanes[1])+float(lanes[2]+lanes[3]))));}
double radius(double x){unsigned b=word<uint16_t>(_Float16(x))&0x7fff,e=b>>10;
 if(!b||e==31)return 0;double r=std::ldexp(1.0,std::max(-25,int(e)-26));
 return e>1&&(b&1023)==0?r/2:r;}
int main(){std::fesetround(FE_TONEAREST);unsigned cases=0,interval_fail=0,cell_fail=0;
 for(unsigned i=0x2800;i<=0x4000;i++)for(unsigned j=0x2801;j<=0x3c00;j++)for(int signs=1;signs<=2;signs++){
  double x=word<_Float16>(uint16_t(i|((signs&1)?0x8000:0))),y=word<_Float16>(uint16_t(j|((signs&2)?0x8000:0)));
  double a,b;norm(x,y,a,b);double v0=projection(a,b,0),v1=projection(a,b,2);
  double updates[2]={projection(v0,v1,4),projection(v0,v1,6)};
  double residuals[2]={double(_Float16(float(x+updates[0]))),double(_Float16(float(y+updates[1])))};
  for(int k=0;k<2;k++){
   double directed=signs==1?updates[k]:-updates[k];interval_fail+=directed<lo[k]||directed>hi[k];
   cell_fail+=radius(residuals[k])<radii[k];
  }cases++;
 }std::printf("Directed residual proof: pairs=%u intervalViolations=%u cellViolations=%u\n",cases,interval_fail,cell_fail);
 return interval_fail||cell_fail?1:0;}
''')
   built=subprocess.run(['clang++','-O3','-ffp-contract=off','-std=c++17',str(source),'-o',str(binary)],capture_output=True,text=True)
   self.assertEqual(built.returncode,0,built.stderr)
   tested=subprocess.run([str(binary)],capture_output=True,text=True)
   self.assertEqual(tested.returncode,0,tested.stdout+tested.stderr)
   self.assertIn('pairs=62924800 intervalViolations=0 cellViolations=0',tested.stdout)
   print(tested.stdout,end='')

if __name__=='__main__':unittest.main()
