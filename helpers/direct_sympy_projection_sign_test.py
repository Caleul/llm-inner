"""Certify cheaper projection signs beyond every original numerical error."""
import json,math,os,struct,subprocess,tempfile,unittest
from fractions import Fraction as F
from pathlib import Path
from types import SimpleNamespace as NS
from direct_sympy_strings import StringCompiler,Domain,syntax
from direct_sympy_conversions import ConversionSession,FiniteSource
from direct_sympy_projection_constraints import rms_projection_sign_source,norm_error_bound
from direct_sympy_conversions_test import cpp

def checkpoint_sign_parity():
  """Exercise the proof in selected, emitted checkpoint expressions too."""
  from direct_sympy_checkpoint import CheckpointStrings
  from direct_sympy_streaming_literals import streaming_literals
  from direct_sympy_coherent_paths import CoherentPaths
  from direct_sympy_input_partitions import interval
  domains={name:interval(-1/128,1/128)for name in ('X1','X2')}
  with tempfile.TemporaryDirectory()as directory:
    root=Path(directory);path=Path(os.environ.get('LLM_INNER_DIRECT_PROJECTION_SIGN_OUTPUT',str(root/'selected.expr')))
    with CheckpointStrings(os.environ['LLM_INNER_DIRECT_JSON_CHECKPOINT'],StringCompiler(max_characters=1024**2),input_domains=domains)as model:
      with streaming_literals(model)as registry:
        def component(i):return model.norm('sign-test','model.layers.0.input_layernorm.weight',i,lambda j:f'X{j+1}')
        projection=model.producer('sign-test-projection',lambda:model.linear('model.layers.0.mlp.gate_proj.weight',0,component))
        threshold=struct.unpack('Q',struct.pack('d',2**-14))[0]
        expression=f'Piecewise((Float64(U64And(Bits64({projection}),9223372036854775808)),U64And(Bits64({projection}),9223372036854775807)>={threshold}),(1.0,True))'
        plan=CoherentPaths(registry,max_paths=64);arms=list(plan.arms(expression))
        assert plan.stats.get('selectedSignMaskEliminations',0)>0,plan.stats
        # Both the admitted projection and its non-admitted sibling must
        # survive: a proved sign must never replace a magnitude or escape
        # into a sibling where the error margin has not been established.
        admitted=min(a.view.size(a.expression)for a in arms if 'X1'in a.expression and 'CompileValue'not in a.expression)
        assert admitted<200,admitted
        assert any(a.expression=='1'for a in arms)
        fallback=max(a.view.size(a.expression)for a in arms)
        assert fallback>200,fallback
        report=plan.write(path,expression,max_characters=4*1024**2)
        weights=[float(model.weight('model.layers.0.mlp.gate_proj.weight',0,i))for i in range(2)]
        gamma=[float(model.weight('model.layers.0.input_layernorm.weight',i))for i in range(2)]
        epsilon=repr(float(model.config['rms_norm_eps']))
    text=path.read_text();assert 'CompileValue'not in text
    native=root/'selected.cpp';binary=root/'selected'
    native.write_text('''#include <cstdint>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <cfenv>
#include <initializer_list>
template<class T,class U>T word(U x){T y;std::memcpy(&y,&x,sizeof(y));return y;}
double candidate(double X1,double X2){return '''+cpp(syntax(text))+''';}
int main(){if(std::fesetround(FE_TONEAREST))return 2;unsigned cases=0,mismatches=0,accepted=0;
auto test=[&](uint16_t a,uint16_t b){double x=word<_Float16>(a),y=word<_Float16>(b);
float sum=float(x*x+y*y);float mean=float(double(float(double(sum)/2))+'''+epsilon+''');
float inverse=float(1.0/double(float(std::sqrt(double(mean)))));
double h0=double(_Float16(float(double(_Float16(float(x*inverse)))*'''+repr(gamma[0])+''')));
double h1=double(_Float16(float(double(_Float16(float(y*inverse)))*'''+repr(gamma[1])+''')));
float p0=float(0.0+h0*'''+repr(weights[0])+'''),p1=float(0.0+h1*'''+repr(weights[1])+''');
double projected=double(_Float16(float(double(float(double(p0)+p1))+float(0.0+0.0))));
bool chosen=std::abs(projected)>=0x1p-14;accepted+=chosen;cases++;
double expected=chosen?word<double>(word<uint64_t>(projected)&(uint64_t(1)<<63)):1.0;
mismatches+=word<uint64_t>(candidate(x,y))!=word<uint64_t>(expected);
};
for(unsigned a=0;a<=0x2000;a++)for(unsigned sa:{0u,0x8000u})for(unsigned b:{0u,0x8000u,1u,0x8001u,0x0400u,0x8400u,0x2000u,0xa000u})test(uint16_t(a|sa),uint16_t(b));
for(unsigned i=0;i<4096;i++)test(uint16_t(((i*997u)%0x2001u)|((i&1)?0x8000u:0)),uint16_t(((i*577u)%0x2001u)|((i&2)?0x8000u:0)));
std::printf("Selected checkpoint projection parity: cases=%u accepted=%u mismatches=%u\\n",cases,accepted,mismatches);return mismatches||!accepted?1:0;}
''')
    subprocess.run(['clang++','-O3','-ffp-contract=off','-std=c++17',str(native),'-o',str(binary)],check=True,capture_output=True,text=True)
    result=subprocess.run([str(binary)],check=True,capture_output=True,text=True,timeout=120)
    assert 'cases=135184'in result.stdout and 'mismatches=0'in result.stdout,result.stdout
    print(result.stdout,end='');print(f'Selected projection replacement: admittedCharacters={admitted} fallbackCharacters={fallback} emittedCharacters={report["characters"]}')
    if os.environ.get('LLM_INNER_DIRECT_PROJECTION_SIGN_OUTPUT'):
      from direct_sympy_cover_regions import live_identity
      from direct_sympy_savepoints import digest_file
      from direct_sympy_partition_run import encode
      evidence={'scope':'Checkpoint input RMS, gate projection and selected sign decision; not complete Llama output coordinate',
        'compilerIdentity':live_identity(os.environ['LLM_INNER_DIRECT_JSON_CHECKPOINT'],2),
        'artifact':{'path':str(path.resolve()),'characters':len(text),'sha256':digest_file(path),'compilerAliases':0},
        'inputDomains':encode(domains),'admittedBodyCharacters':admitted,'nonAdmittedBodyCharacters':fallback,
        'nativeCases':135184,'nativeAcceptedCases':117379,'nativeBitMismatches':0,'stats':plan.stats}
      path.with_suffix('.json').write_text(json.dumps(evidence,indent=2)+'\n')

def fixture():
 compiler=StringCompiler();domains={name:Domain(F(-65504),F(65504),-24,False)for name in ('X1','X2')}
 epsilon=F(struct.unpack('f',struct.pack('f',1e-6))[0])
 recipe='R16(R32(R32(R32(0.0+CompileValue0()*0.01171875)+R32(0.0+CompileValue1()*-0.009765625))+R32(0.0+0.0)))'
 vector={'width':2,'sources':('X1','X2'),'sourceBounds':(FiniteSource(-65504,65504,-24),)*2,
  'components':{0:'CompileValue0()',1:'CompileValue1()'},'magnitudes':(F(2),F(2)),
  'gamma':(F(-1),F(1)),'epsilon':epsilon,'context':compiler.context(domains),'roundingError':norm_error_bound(2,epsilon)}
 model=NS(compiler=compiler,domains=domains,norm_vectors={'test':vector},conversions=ConversionSession(compiler,domains,input_dtype='f16'))
 registry=NS(model=model,definitions=['X1','X2',recipe],definition_recipes={2:recipe})
 return registry,vector,epsilon

class ProjectionSignTests(unittest.TestCase):
 def test_sign_agrees_with_original_projection_for_every_central_half_pair(self):
  registry,vector,epsilon=fixture()
  source=rms_projection_sign_source(registry,2,FiniteSource(-.06,.06,-24,2**-14))
  self.assertIsNotNone(source)
  self.assertNotIn('CompileValue',source);self.assertNotIn('sqrt',source)
  loss=registry._rms_projection_signs[1][2][0][0]
  self.assertLess(loss,F(2)**-14)
  self.assertIsNone(rms_projection_sign_source(registry,2,FiniteSource(-.06,.06,-24,loss)))
  above=math.nextafter(float(loss),math.inf)
  self.assertIsNotNone(rms_projection_sign_source(registry,2,FiniteSource(-.06,.06,-24,above)))
  self.assertIsNone(rms_projection_sign_source(registry,2,FiniteSource(-.06,.06,-24)))
  anchor=syntax('0.0+CompileValue0()*0.01171875+CompileValue1()*-0.009765625')
  selected=FiniteSource(-.06,.06,-24)
  self.assertEqual(rms_projection_sign_source(registry,2,selected,anchors=((anchor,FiniteSource(-.06,.06,-55,2**-14)),)),source)
  text='''#include <cstdint>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <cfenv>
#include <initializer_list>
template<class T,class U>T word(U x){T y;std::memcpy(&y,&x,sizeof(y));return y;}
double sign_source(double X1,double X2){return SOURCE;}
int main(){if(std::fesetround(FE_TONEAREST))return 2;uint64_t cases=0,selected=0,failures=0;
auto test=[&](uint16_t a,uint16_t b){double x=word<_Float16>(a),y=word<_Float16>(b);
 float sum=float(x*x+y*y);float mean=float(double(float(double(sum)/2))+EPS);
 float inverse=float(1.0/double(float(std::sqrt(double(mean)))));
 double h0=-double(_Float16(float(x*inverse))),h1=double(_Float16(float(y*inverse)));
 float p0=float(0.0+h0*0.01171875),p1=float(0.0+h1*-0.009765625);
 double value=double(_Float16(float(double(float(double(p0)+p1))+float(0.0+0.0))));cases++;
 double anchor=0.0+h0*0.01171875+h1*-0.009765625;
 if(std::abs(value)>=0x1p-14||std::abs(anchor)>=0x1p-14){selected++;double raw=sign_source(x,y);
  failures+=raw==0.0||((word<uint64_t>(raw)^word<uint64_t>(value))>>63)!=0;}
};
for(unsigned a=0;a<=0xc00;a++)for(unsigned sa:{0u,0x8000u})for(unsigned b=0;b<=0xc00;b++)for(unsigned sb:{0u,0x8000u})test(uint16_t(a|sa),uint16_t(b|sb));
for(unsigned a=0;a<256;a++)for(unsigned b=0;b<256;b++)test(uint16_t(((a*997u)%0x7c00u)|((a&1)?0x8000u:0)),uint16_t(((b*577u)%0x7c00u)|((b&1)?0x8000u:0)));
std::printf("RMS projection sign parity: cases=%llu selected=%llu mismatches=%llu\\n",(unsigned long long)cases,(unsigned long long)selected,(unsigned long long)failures);return failures||!selected?1:0;}
'''.replace('SOURCE',cpp(syntax(source))).replace('EPS',repr(float(epsilon)))
  with tempfile.TemporaryDirectory()as directory:
   root=Path(directory);native=root/'sign.cpp';binary=root/'sign';native.write_text(text)
   subprocess.run(['clang++','-O3','-ffp-contract=off','-std=c++17',str(native),'-o',str(binary)],check=True,capture_output=True,text=True)
   result=subprocess.run([str(binary)],check=True,capture_output=True,text=True,timeout=120)
   self.assertIn('cases=37838852',result.stdout);self.assertIn('mismatches=0',result.stdout);print(result.stdout,end='')
  if os.environ.get('LLM_INNER_DIRECT_JSON_CHECKPOINT'):checkpoint_sign_parity()

 def test_unsupported_contexts_and_zero_margins_keep_original_signs(self):
  bound=FiniteSource(-.06,.06,-24,2**-14)
  for kind in ('gamma','context','bias','unknown','forward'):
   registry,vector,_=fixture()
   if kind=='gamma':vector['gamma']=(F(2),F(1))
   elif kind=='context':vector['context']='foreign'
   elif kind=='bias':registry.definition_recipes[2]='R16(CompileValue0()*0.01171875+0.001)'
   elif kind=='unknown':registry.definition_recipes[2]='R16(Silu16(CompileValue0()))'
   else:vector['sources']=('CompileValue2()','X2')
   self.assertIsNone(rms_projection_sign_source(registry,2,bound),kind)
  registry,vector,_=fixture()
  self.assertIsNotNone(rms_projection_sign_source(registry,2,bound))
  vector['gamma']=(F(2),F(1))
  self.assertIsNone(rms_projection_sign_source(registry,2,bound))
  vector['gamma']=(F(-1),F(1));vector['epsilon']=F(2)**-150
  self.assertIsNone(rms_projection_sign_source(registry,2,bound))

if __name__=='__main__':unittest.main()
