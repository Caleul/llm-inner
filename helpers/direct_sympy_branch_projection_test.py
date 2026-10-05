"""Branch projection certificates preserve original ordered numeric bodies."""
import json,os,struct,subprocess,tempfile,unittest
from fractions import Fraction as F
from pathlib import Path
from types import SimpleNamespace as NS
from direct_sympy_checkpoint import CheckpointStrings
from direct_sympy_conversions import FiniteSource
from direct_sympy_projection_constraints import projection_branch_bounds
from direct_sympy_strings import StringCompiler,syntax
from direct_sympy_layer_bounds import dot_interval

class ProjectionTests(unittest.TestCase):
    def test_context_and_dependency_order_are_required(self):
        compiler=StringCompiler();domains={};context=compiler.context(domains)
        bounds={0:FiniteSource(-2**-14,-2**-24,-24,2**-24),1:FiniteSource(1,1,-24,1)}
        def bound(node):
            if getattr(node,'id','')=='X1':return None
            return FiniteSource(-2,2,-24,0)
        session=NS(value_kind=lambda node:'half',bounds=bound)
        model=NS(compiler=compiler,domains=domains,conversions=session,
            shape=lambda name:[1,2],weight=lambda name,row,column:repr((0.0,1.0)[column]),
            linear_frames={'first':{'name':'w','row':0,'operands':('CompileValue0()','CompileValue1()'),'context':context},
                           'second':{'name':'w','row':0,'operands':(None,'CompileValue2()'),'context':context}})
        registry=NS(model=model,names={'first':'CompileValue2()','second':'CompileValue3()'})
        result=projection_branch_bounds(registry,bounds)
        self.assertEqual((result[2].minimum,result[2].maximum),(1,1))
        self.assertEqual((result[3].minimum,result[3].maximum),(1,1))
        self.assertEqual(projection_branch_bounds(registry,bounds,max_products=1),bounds)
        model.linear_frames['first']['context']=('wrong',)
        self.assertNotIn(2,projection_branch_bounds(registry,bounds))
        self.assertEqual(projection_branch_bounds(registry,{}),{})

    def test_only_whole_projection_producers_receive_a_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory,'config.json').write_text(json.dumps({'model_type':'llama','hidden_size':2,'num_hidden_layers':1}))
            model=CheckpointStrings(directory,StringCompiler(),lower_conversions=False)
            model.shape=lambda name:[1,2];model.weight=lambda name,row,col:'1.0'
            model.producer('projection',lambda:model.linear('w',0,lambda col:'X'+str(col+1)))
            self.assertIn('projection',model.linear_frames)
            model.producer('residual',lambda:'('+model.linear('w',0,lambda col:'X'+str(col+1))+') + X1')
            self.assertNotIn('residual',model.linear_frames)
            def failure():
                model.linear('w',0,lambda col:'X'+str(col+1));raise ValueError('failed')
            with self.assertRaisesRegex(ValueError,'failed'):model.producer('failure',failure)
            self.assertEqual(model._projection_capture,[])
            self.assertNotIn('failure',model.linear_frames)

    @unittest.skipUnless(os.environ.get('LLM_INNER_DIRECT_JSON_CHECKPOINT'),'Checkpoint required')
    def test_original_checkpoint_projections_on_all_tiny_branch_pairs(self):
        with CheckpointStrings(os.environ['LLM_INNER_DIRECT_JSON_CHECKPOINT'],StringCompiler()) as model:
            name='model.layers.0.self_attn.'
            w=[[float(model.weight(name+'v_proj.weight',i,j)) for j in range(2)] for i in range(2)]
            o=[[float(model.weight(name+'o_proj.weight',i,j)) for j in range(2)] for i in range(2)]
            h=[float(model.weight('lm_head.weight',2,j)) for j in range(2)]
            z=[(-2**-14,-2**-24),(1.4140625,1.4140625)]
            v=[dot_interval(model,name+'v_proj.weight',i,z) for i in range(2)]
            out=[dot_interval(model,name+'o_proj.weight',i,v) for i in range(2)]
            head=dot_interval(model,'lm_head.weight',2,z)
            for pair in v+out+[head]:self.assertEqual(pair[0],pair[1])
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'proof.cpp';binary=root/'proof'
            source.write_text('''#include <cmath>
#include <cstdint>
#include <cstring>
#include <cfenv>
#include <cstdio>
template<class T,class U>T word(U x){T y;std::memcpy(&y,&x,sizeof(y));return y;}
'''+f'double w[2][2]={{{{{w[0][0]},{w[0][1]}}},{{{w[1][0]},{w[1][1]}}}}};\n'+
                f'double o[2][2]={{{{{o[0][0]},{o[0][1]}}},{{{o[1][0]},{o[1][1]}}}}};\n'+
                f'double h[2]={{{h[0]},{h[1]}}};double expectV[2]={{{v[0][0]},{v[1][0]}}};double expectO[2]={{{out[0][0]},{out[1][0]}}};double expectH={head[0]};\n'+r'''
double dot(double x,double y,double* w){float a=float(0.0f+float(x*w[0])),b=float(0.0f+float(y*w[1]));return double(_Float16(float(float(a+b)+float(0.0f+0.0f))));}
int main(){std::fesetround(FE_TONEAREST);unsigned long long pairs=0,selected=0,violations=0;
for(unsigned i=10240;i<=31743;i++)for(unsigned j=10240;j<=31743;j++){
 double x=-double(word<_Float16>(uint16_t(i))),y=double(word<_Float16>(uint16_t(j)));pairs++;
 float mean=float(float(float(x*x)+float(y*y))/2.0f+1e-6f);
 if(!(x*x<double(mean)*0x1p-28))continue;selected++;
 float inv=float(1.0/double(float(std::sqrt(double(mean)))));
 double z0=double(_Float16(float(x*inv))),z1=double(_Float16(float(y*inv)));
 double v0=dot(z0,z1,w[0]),v1=dot(z0,z1,w[1]);
 violations+=v0!=expectV[0]||v1!=expectV[1]||dot(v0,v1,o[0])!=expectO[0]||dot(v0,v1,o[1])!=expectO[1]||dot(z0,z1,h)!=expectH;
}std::printf("Branch projection cells: pairs=%llu selected=%llu violations=%llu\n",pairs,selected,violations);return violations?1:0;}
''')
            built=subprocess.run(['clang++','-O3','-ffp-contract=off','-std=c++17',str(source),'-o',str(binary)],capture_output=True,text=True)
            self.assertEqual(built.returncode,0,built.stderr)
            run=subprocess.run([str(binary)],capture_output=True,text=True,timeout=120)
            self.assertEqual(run.returncode,0,run.stdout+run.stderr)
            self.assertIn('pairs=462422016 selected=22147398 violations=0',run.stdout)
            print(run.stdout,end='')

if __name__=='__main__':unittest.main()
