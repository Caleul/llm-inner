"""Early projection proofs preserve typed intervals and avoid dead producers."""
import json,os,struct,subprocess,tempfile,unittest
from pathlib import Path
from direct_sympy_checkpoint import CheckpointStrings
from direct_sympy_strings import StringCompiler
from direct_sympy_input_partitions import interval
from direct_sympy_layer_bounds import dot_interval

class Matrix:
    def __init__(self,weights):self.weights=weights
    def shape(self,name):return [1,len(self.weights)]
    def weight(self,name,row,col):return repr(self.weights[col])

class EarlyProjectionTests(unittest.TestCase):
    def test_signed_bounds_preserve_underflow_zero_and_reject_unsafe_inputs(self):
        bits=lambda x:struct.pack('e',x)
        self.assertEqual(dot_interval(Matrix([.5]),'w',0,[(1,1)]),(.5,.5))
        negative=dot_interval(Matrix([-2**-24]),'w',0,[(2**-24,2**-24)])
        self.assertEqual(tuple(map(bits,negative)),(bits(-0.0),bits(-0.0)))
        both=dot_interval(Matrix([2**-24]),'w',0,[(-2**-24,2**-24)])
        self.assertNotEqual(bits(both[0]),bits(both[1]))
        zeros=dot_interval(Matrix([-1]),'w',0,[(-0.0,0.0)])
        self.assertEqual(tuple(map(bits,zeros)),(bits(0.0),bits(0.0)))
        for inputs in ([None],[(2,1)],[(.1,1)],[(0,float('inf'))],[]):
            self.assertIsNone(dot_interval(Matrix([1]),'w',0,inputs))
        for weights in ([.1],[float('nan')],[65504]):
            self.assertIsNone(dot_interval(Matrix(weights),'w',0,[(65504,65504)]))

    def test_constant_projection_never_requests_its_input_producer(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory,'config.json').write_text(json.dumps({'model_type':'llama','hidden_size':1,'num_hidden_layers':1}))
            model=CheckpointStrings(directory,StringCompiler())
            model.shape=lambda name:[1,1];model.weight=lambda name,row,col:'.5'
            def forbidden(i):raise AssertionError('Dead producer was expanded')
            result=model.linear('w',0,forbidden,input_bounds=lambda:[(1,1)])
            self.assertEqual(result,'0.5');self.assertEqual(model.constant_projections,1)
            requested=[]
            result=model.linear('w',0,lambda i:requested.append(i) or 'X1',input_bounds=lambda:None)
            self.assertEqual(requested,[0]);self.assertIn('X1',result)
            model.weight=lambda name,row,col:repr(2**-24);requested.clear()
            model.linear('w',0,lambda i:requested.append(i) or 'X1',input_bounds=lambda:[(-2**-24,2**-24)])
            self.assertEqual(requested,[0]) # -0/+0 are different outcomes.

    @unittest.skipUnless(os.environ.get('LLM_INNER_DIRECT_JSON_CHECKPOINT'),'Checkpoint required')
    def test_rms_enclosure_cache_is_local_to_input_context_and_builds_no_norm(self):
        with CheckpointStrings(os.environ['LLM_INNER_DIRECT_JSON_CHECKPOINT'],StringCompiler(),
            input_domains={'X1':interval(-2**-16,2**-16),'X2':interval(1,65504)}) as m:
            name='model.layers.0.input_layernorm.weight'
            values=m.norm_intervals(name,lambda i:'X'+str(i+1))
            self.assertEqual(values[1],(1.4140625,1.4140625));self.assertEqual(m.memo,{})
            self.assertIs(values,m.norm_intervals(name,lambda i:'X'+str(i+1)))
            changed={'X1':interval(-1,1),'X2':interval(1,65504)}
            m.domains=changed;m.conversions.domains=changed
            widened=m.norm_intervals(name,lambda i:'X'+str(i+1))
            self.assertNotEqual(widened,values);self.assertEqual(len(m.norm_bound_cache),2)

    @unittest.skipUnless(os.environ.get('LLM_INNER_DIRECT_JSON_CHECKPOINT'),'Checkpoint required')
    def test_constant_v_projection_for_every_half_pair_in_certified_region(self):
        with CheckpointStrings(os.environ['LLM_INNER_DIRECT_JSON_CHECKPOINT'],StringCompiler(),
            input_domains={'X1':interval(-2**-16,2**-16),'X2':interval(1,65504)}) as m:
            bounds=m.norm_intervals('model.layers.0.input_layernorm.weight',lambda i:'X'+str(i+1))
            name='model.layers.0.self_attn.v_proj.weight'
            self.assertEqual(m.shape(name),[2,2])
            intervals=[dot_interval(m,name,i,bounds) for i in range(2)]
            self.assertTrue(all(struct.pack('e',a)==struct.pack('e',b) for a,b in intervals))
            weights=[float(m.weight(name,i,j)) for i in range(2) for j in range(2)]
            gamma=[float(m.weight('model.layers.0.input_layernorm.weight',i)) for i in range(2)]
            expected=[pair[0] for pair in intervals]
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'projection.cpp';binary=root/'projection'
            source.write_text('''#include <cmath>
#include <cstdint>
#include <cstring>
#include <cfenv>
#include <cstdio>
template<class T,class U>T word(U x){T y;std::memcpy(&y,&x,sizeof(y));return y;}
'''+''.join('double '+name+'[]={'+','.join(map(repr,values))+'};\n' for name,values in [('weights',weights),('gamma',gamma),('expected',expected)])+r'''
int main(){std::fesetround(FE_TONEAREST);unsigned cases=0,fail=0;
 for(unsigned b=0;b<65536;b++){if((b&0x7c00)==0x7c00)continue;double x=word<_Float16>(uint16_t(b));if(std::abs(x)>0x1p-16)continue;
  for(unsigned c=0x3c00;c<0x7c00;c++){double y=word<_Float16>(uint16_t(c));
   float mean=float(float(float(x*x)+float(y*y))/2.0f+1e-6f);float inverse=float(1.0/double(float(std::sqrt(double(mean)))));
   double n0=double(_Float16(float(double(_Float16(float(x*inverse)))*gamma[0])));
   double n1=double(_Float16(float(double(_Float16(float(y*inverse)))*gamma[1])));
   for(int i=0;i<2;i++){float lanes[4]={};lanes[0]=float(float(n0*weights[2*i])+0.0f);lanes[1]=float(float(n1*weights[2*i+1])+0.0f);
    float sum=float(float(lanes[0]+lanes[1])+float(lanes[2]+lanes[3]));double result=double(_Float16(sum));
    fail+=word<uint64_t>(result)!=word<uint64_t>(expected[i]);}cases++;
  }
 }std::printf("Early constant V projection: pairs=%u mismatches=%u\n",cases,fail);return fail?1:0;}
''')
            built=subprocess.run(['clang++','-O3','-ffp-contract=off','-std=c++17',str(source),'-o',str(binary)],capture_output=True,text=True)
            self.assertEqual(built.returncode,0,built.stderr)
            tested=subprocess.run([str(binary)],capture_output=True,text=True)
            self.assertEqual(tested.returncode,0,tested.stdout+tested.stderr)
            self.assertIn('pairs=8421376 mismatches=0',tested.stdout);print(tested.stdout,end='')

if __name__=='__main__':unittest.main()
