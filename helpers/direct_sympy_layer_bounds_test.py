"""Outward proof arithmetic and actual ordered Half layer update enclosures."""
import os,struct,subprocess,tempfile,unittest
from fractions import Fraction as F
from pathlib import Path
from direct_sympy_layer_bounds import dot,half,layer,silu_bound
from direct_sympy_checkpoint import CheckpointStrings
from direct_sympy_strings import StringCompiler
from direct_sympy_partition_run import update_threshold
from direct_sympy_input_partitions import rank


class Weights:
    def __init__(self,values):self.values=values
    def shape(self,name):return [1,len(self.values)]
    def weight(self,name,row,col):return repr(self.values[col])


class LayerBoundTests(unittest.TestCase):
    def test_silu_bound_encloses_every_certified_half_prefix(self):
        import torch
        magnitudes=[struct.unpack('e',struct.pack('H',b))[0] for b in range(0x2c01)]
        inputs=[x for b in range(0x2c01) for x in (magnitudes[b],-magnitudes[b])]
        reference=torch.nn.functional.silu(torch.tensor(inputs,dtype=torch.float16)).abs().double().tolist()
        maximum=0.0;records=[]
        for i,bound in enumerate(magnitudes):
            exact=F(bound)*(F(1,2)+F(bound)/4)
            self.assertEqual(F(bound*(.5+bound*.25)),exact)
            maximum=max(maximum,*reference[2*i:2*i+2])
            upper=silu_bound(bound)
            self.assertLessEqual(maximum,upper,(bound,maximum,upper))
            records.append('{'+repr(bound)+','+repr(upper)+'}')
        self.native('''struct Record{double b,upper;};Record records[]={'''+','.join(records)+'''};
int main(){std::fesetround(FE_TONEAREST);unsigned cases=0,fail=0;double maximum=0;
 for(auto r:records){for(double x:{r.b,-r.b}){
  double value=x*(.5+x*(.25-x*x/48.0));double stored=double(_Float16(float(value)));
  maximum=std::fmax(maximum,std::abs(stored));cases++;
 }fail+=maximum>r.upper;}
 std::printf("Certified SiLU magnitude bounds: cases=%u violations=%u\\n",cases,fail);return fail?1:0;}
''','Certified SiLU magnitude bounds: cases=22530 violations=0')
        self.assertEqual(silu_bound(2**-24),0.0)
        self.assertNotEqual(half(2**-24*(.5+2**-24*.25)),silu_bound(2**-24))
        for invalid in (None,-1.0,.1,float('nan'),float('inf')):self.assertIsNone(silu_bound(invalid))
        for outside in (.125,1.0,65504.0):self.assertEqual(silu_bound(outside),outside)

    def test_exact_sum_gamma_and_storage_bound_are_outward(self):
        for n in (1,2,4,8,33):
            values=[2**-16]*n
            result=dot(Weights(values),'weight',[65504.0]*n)[0]
            bound=F(65504)*F(2**-16)*n*F(2**24,2**24-n-3)
            # Every stored Half value below the rational bound is <= result.
            rounded=float(bound)
            if F(rounded)<bound:rounded=__import__('math').nextafter(rounded,float('inf'))
            self.assertEqual(result,half(rounded))
            self.assertLess(result,half(2*65504*2**-16*n))
        self.assertEqual(dot(Weights([2**-24]),'weight',[2**-24]),[0.0])
        for inputs in ([None],[-1.0],[float('inf')],[float('nan')],[0.1]):
            self.assertIsNone(dot(Weights([1.0]),'weight',inputs))
        for weights in ([float('inf')],[float('nan')],[0.1]):
            self.assertIsNone(dot(Weights(weights),'weight',[1.0]))
        self.assertIsNone(dot(Weights([65504.0]),'weight',[65504.0]))

    def test_ordered_native_reductions_stay_inside_tight_projection_bounds(self):
        records=[]
        for n in (1,2,4,8):
            bound=dot(Weights([2**-16]*n),'weight',[65504.0]*n)[0]
            records.append('{'+str(n)+','+repr(bound)+'}')
        self.native('''struct Record{int n;double bound;};Record records[]={'''+','.join(records)+'''};
int main(){std::fesetround(FE_TONEAREST);unsigned long long cases=0,fail=0;
for(auto record:records)for(unsigned b=0;b<65536;b++){
 if((b&0x7c00)==0x7c00)continue;double x=word<_Float16>(uint16_t(b));
 for(uint16_t a:anchors){double y=word<_Float16>(a);float lanes[4]={};
  for(int i=0;i<record.n;i++){double value=i?y:x;double weight=(i&1)?-0x1p-16:0x1p-16;
   lanes[i%4]=float(double(lanes[i%4])+float(value*weight));}
  float sum=float(float(float(double(lanes[0])+lanes[1])+lanes[2])+lanes[3]);
  double stored=double(_Float16(sum));fail+=std::abs(stored)>record.bound;cases++;
 }}std::printf("Tight projection bounds: cases=%llu violations=%llu\\n",cases,fail);return fail?1:0;}
''','Tight projection bounds: cases=3555328 violations=0')

    @unittest.skipUnless(os.environ.get('LLM_INNER_DIRECT_JSON_CHECKPOINT'),'Checkpoint required')
    def test_actual_layer_updates_stay_inside_tighter_bounds(self):
        with CheckpointStrings(os.environ['LLM_INNER_DIRECT_JSON_CHECKPOINT'],StringCompiler()) as model:
            bounds=layer(model,'model.layers.0.')
            self.assertEqual([update_threshold(max(bounds[k][i] for k in bounds)) for i in range(model.width)],[rank(8),rank(4)])
            prefix='model.layers.0.'
            def weights(name):
                shape=model.shape(name)
                if len(shape)==1:return [float(model.weight(name,i)) for i in range(shape[0])]
                return [float(model.weight(name,i,j)) for i in range(shape[0]) for j in range(shape[1])]
            vectors={'pre':weights(prefix+'input_layernorm.weight'),'post':weights(prefix+'post_attention_layernorm.weight'),
                'v':weights(prefix+'self_attn.v_proj.weight'),'o':weights(prefix+'self_attn.o_proj.weight'),
                'g':weights(prefix+'mlp.gate_proj.weight'),'u':weights(prefix+'mlp.up_proj.weight'),'d':weights(prefix+'mlp.down_proj.weight'),
                'ab':bounds['attention'],'mb':bounds['mlp']}
            self.assertEqual((model.width,model.config['intermediate_size'],model.layers),(2,1,1))
            eps=repr(model.config['rms_norm_eps'])
        declarations='\n'.join('double '+name+'[]={'+','.join(map(repr,values))+'};' for name,values in vectors.items())
        self.native(declarations+'''
double stored(double x){return double(_Float16(float(x)));}
void norm(double x,double y,double *gamma,double *result){
 float sum=float(float(x*x)+float(y*y));float mean=float(float(sum/2.0f)+EPSf);
 float inverse=float(1.0/double(float(std::sqrt(double(mean)))));
 result[0]=stored(stored(x*double(inverse))*gamma[0]);result[1]=stored(stored(y*double(inverse))*gamma[1]);}
double linear(double *w,double *x,int n){float lanes[4]={};for(int i=0;i<n;i++)lanes[i%4]=float(double(lanes[i%4])+float(x[i]*w[i]));
 return stored(float(float(float(double(lanes[0])+lanes[1])+lanes[2])+lanes[3]));}
int main(){std::fesetround(FE_TONEAREST);unsigned cases=0,fail=0;
for(unsigned b=0;b<65536;b++){if((b&0x7c00)==0x7c00)continue;
 for(uint16_t a:anchors){double x=word<_Float16>(uint16_t(b)),y=word<_Float16>(a),preValues[2],values[2],attn[2],postValues[2];
  norm(x,y,pre,preValues);for(int i=0;i<2;i++)values[i]=linear(v+2*i,preValues,2);
  for(int i=0;i<2;i++){attn[i]=linear(o+2*i,values,2);fail+=std::abs(attn[i])>ab[i];}
  norm(stored(x+attn[0]),stored(y+attn[1]),post,postValues);
  double gate=linear(g,postValues,2),up=linear(u,postValues,2);
  float sigmoid=1.0f/(1.0f+std::exp(-float(gate)));double activation=stored(float(gate)*sigmoid);
  fail+=std::abs(activation)>std::abs(gate);double product[]={stored(activation*up)};
  for(int i=0;i<2;i++)fail+=std::abs(linear(d+i,product,1))>mb[i];cases++;
 }}std::printf("Tight checkpoint update bounds: cases=%u violations=%u\\n",cases,fail);return fail?1:0;}
'''.replace('EPS',eps),'Tight checkpoint update bounds: cases=888832 violations=0')

    def native(self,body,expected):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'proof.cpp';binary=root/'proof'
            source.write_text('''#include <cmath>
#include <cstdint>
#include <cstring>
#include <cfenv>
#include <cstdio>
#include <initializer_list>
template<class T,class U>T word(U x){T y;std::memcpy(&y,&x,sizeof(y));return y;}
uint16_t anchors[]={0,0x8000,1,0x8001,0x03ff,0x83ff,0x0400,0x8400,0x3555,0xb555,0x3c01,0xbc01,0x7bff,0xfbff};
'''+body)
            built=subprocess.run(['clang++','-O3','-ffp-contract=off','-std=c++17',str(source),'-o',str(binary)],capture_output=True,text=True)
            self.assertEqual(built.returncode,0,built.stderr)
            tested=subprocess.run([str(binary)],capture_output=True,text=True)
            self.assertEqual(tested.returncode,0,tested.stdout+tested.stderr);self.assertIn(expected,tested.stdout)
            print(tested.stdout,end='')


if __name__=='__main__':unittest.main()
