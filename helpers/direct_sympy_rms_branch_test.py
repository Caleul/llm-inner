"""Branch-local RMS cell proofs retain the original mean and rounding."""
import os,struct,subprocess,tempfile,unittest
from fractions import Fraction as F
from pathlib import Path
from types import SimpleNamespace as NS
from direct_sympy_conversions import FiniteSource
from direct_sympy_projection_constraints import rms_branch_bounds,rms_source_branch_bounds
from direct_sympy_strings import Domain,StringCompiler,syntax

def fixture():
    compiler=StringCompiler();domains={'X1':Domain(F(1,32),F(65504),-15,False),'X2':Domain(F(1,32),F(65504),-15,False)}
    eps=struct.unpack('f',struct.pack('f',1e-6))[0]
    vector={'width':2,'sources':('X1','X2'),'sourceBounds':(FiniteSource(.03125,65504,-15,.03125),)*2,
        'context':compiler.context(domains),'mean':'CompileValue2()','epsilon':F(eps),'gamma':(F(1),F(1)),
        'components':{0:'CompileValue3()',1:'CompileValue4()'}}
    return NS(model=NS(compiler=compiler,domains=domains,norm_vectors={'n':vector})),vector

class BranchCellTests(unittest.TestCase):
    def test_shared_tiny_mean_refines_sources_only_in_its_own_context(self):
        registry,vector=fixture()
        vector['sourceBounds']=(FiniteSource(-65504,65504,-24),)*2
        registry.model.domains={name:Domain(F(-65504),F(65504),-24,False) for name in registry.model.domains}
        vector['context']=registry.model.compiler.context(registry.model.domains)
        guards=[(syntax(name+'**2 < CompileValue2()*2**(-28)'),True) for name in ('X1','X2')]
        original=dict(registry.model.domains)
        refined=rms_source_branch_bounds(registry,guards)
        self.assertEqual(set(refined),{'X1','X2'})
        for bound in refined.values():self.assertEqual((bound.minimum,bound.maximum),(-2**-24,2**-24))
        self.assertEqual(registry.model.domains,original)
        self.assertEqual(rms_source_branch_bounds(registry,guards[:1]),{})
        self.assertEqual(rms_source_branch_bounds(registry,[guards[0],(guards[1][0],False)]),{})
        self.assertEqual(rms_source_branch_bounds(registry,[guards[0],(syntax('X2**2 < CompileValue9()*2**(-28)'),True)]),{})
        vector['context']=();self.assertEqual(rms_source_branch_bounds(registry,guards),{})

    def test_only_matching_mean_sources_gamma_and_context_admit_a_cell(self):
        registry,vector=fixture();guard=syntax('X1**2 < CompileValue2()*2**(-28)')
        bounds=rms_branch_bounds(registry,[(guard,True)])
        self.assertEqual((bounds[4].minimum,bounds[4].maximum),(1.4140625,1.4140625))
        self.assertEqual(bounds[3].maximum,2**-14)
        for expr,truth in (('X1**2 < CompileValue9()*2**(-28)',True),('X9**2 < CompileValue2()*2**(-28)',True),('X1**2 < CompileValue2()*2**(-14)',True)):
            self.assertEqual(rms_branch_bounds(registry,[(syntax(expr),truth)]),{})
        for key,value in (('context',()),('gamma',(F(2),F(1))),('width',3),('sourceBounds',())):
            old=vector[key];vector[key]=value
            self.assertEqual(rms_branch_bounds(registry,[(guard,True)]),{})
            vector[key]=old
        normal=rms_branch_bounds(registry,[(guard,False)])
        self.assertEqual(normal[3].minimum,2**-14)
        self.assertNotIn(4,normal)
        opposite=rms_branch_bounds(registry,[(syntax('X1**2 >= CompileValue2()*2**(-28)'),False)])
        self.assertEqual(opposite,bounds)
        vector['sourceBounds']=(FiniteSource(-65504,-.03125,-15,.03125),)*2
        registry.model.domains={name:Domain(F(-65504),F(-1,32),-15,False) for name in registry.model.domains}
        vector['context']=registry.model.compiler.context(registry.model.domains)
        vector['gamma']=(F(1),F(-1))
        signed=rms_branch_bounds(registry,[(guard,True)])
        self.assertEqual(signed[4].minimum,1.4140625)
        self.assertEqual(signed[3].minimum,-2**-14)

    @unittest.skipUnless(os.environ.get('LLM_INNER_DIRECT_JSON_CHECKPOINT'),'Checkpoint required')
    def test_zero_admitting_metadata_preserves_all_native_norm_cells(self):
        from direct_sympy_checkpoint import CheckpointStrings
        from direct_sympy_streaming_literals import streaming_literals
        from direct_sympy_conversions_test import cpp
        from direct_sympy_projection_constraints import impossible_projections
        with CheckpointStrings(os.environ['LLM_INNER_DIRECT_JSON_CHECKPOINT'],StringCompiler(max_characters=32*1024**2)) as model:
            with streaming_literals(model) as registry:
                values=[model.norm('pre','model.layers.0.input_layernorm.weight',i,lambda j:'X'+str(j+1)) for i in range(model.width)]
                vector=next(iter(model.norm_vectors.values()))
                self.assertEqual(vector['floor'],0);self.assertEqual(vector['sourceNormFloor'],0)
                guard=syntax('X1**2 < '+vector['mean']+'*2**(-28)')
                alias=int(vector['components'][0].removeprefix('CompileValue').removesuffix('()'))
                small=rms_branch_bounds(registry,[(guard,True)]);normal=rms_branch_bounds(registry,[(guard,False)])
                self.assertEqual((small[alias].minimum,small[alias].maximum),(-2**-14,2**-14))
                self.assertEqual(normal[alias].minimum_magnitude,2**-14)
                threshold=struct.unpack('Q',struct.pack('d',2**-14))[0]
                guards=[NS(expression=f'U64And(Bits64({v}),9223372036854775807) < {threshold}',truth=True)for v in values]
                self.assertFalse(impossible_projections(registry,guards))
                expressions=[cpp(syntax(''.join(registry.chunks(v))))for v in values]
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'proof.cpp';binary=root/'proof'
            source.write_text("""#include <cmath>
#include <cstdint>
#include <cstring>
#include <cfenv>
#include <cstdio>
template<class T,class U>T word(U x){T y;std::memcpy(&y,&x,sizeof(y));return y;}
"""+f'double first(double X1,double X2){{return {expressions[0]};}}\ndouble second(double X1,double X2){{return {expressions[1]};}}\n'+r'''
int main(){std::fesetround(FE_TONEAREST);unsigned long long cases=0,mismatches=0;
unsigned anchors[]={0,0x8000,1,0x8001,0x400,0x8400,0x1419,0x9419,0x3c00,0xbc00,0x7bff,0xfbff};
for(unsigned b=0;b<65536;b++){if((b&0x7c00)==0x7c00)continue;
 for(unsigned a:anchors)for(unsigned swap=0;swap<2;swap++){
 double x=word<_Float16>(uint16_t(swap?a:b)),y=word<_Float16>(uint16_t(swap?b:a));
float mean=float(float(float(x*x)+float(y*y))/2.0f+1e-6f);
if(x*x<double(mean)*0x1p-28 && y*y<double(mean)*0x1p-28)
 mismatches+=std::fabs(x)>0x1p-24 || std::fabs(y)>0x1p-24;
 float inverse=float(1.0/double(float(std::sqrt(double(mean)))));
 double expected0=double(_Float16(float(x*inverse))),expected1=double(_Float16(float(y*inverse)));
 mismatches+=word<uint64_t>(first(x,y))!=word<uint64_t>(expected0);cases++;
 mismatches+=word<uint64_t>(second(x,y))!=word<uint64_t>(expected1);cases++;
 }}std::printf("Zero-admitting RMS parity: cases=%llu mismatches=%llu\n",cases,mismatches);return mismatches?1:0;}
''')
            built=subprocess.run(['clang++','-O3','-ffp-contract=off','-fbracket-depth=4096','-std=c++17',str(source),'-o',str(binary)],capture_output=True,text=True)
            self.assertEqual(built.returncode,0,built.stderr)
            run=subprocess.run([str(binary)],capture_output=True,text=True,timeout=120)
            self.assertEqual(run.returncode,0,run.stdout+run.stderr)
            self.assertIn('cases=3047424 mismatches=0',run.stdout);print(run.stdout,end='')

    def test_every_positive_half_pair_retains_original_subnormal_branch_cells(self):
        registry,vector=fixture();bounds=rms_branch_bounds(registry,[(syntax('X1**2 < CompileValue2()*2**(-28)'),True)])
        normal=rms_branch_bounds(registry,[(syntax('X1**2 < CompileValue2()*2**(-28)'),False)])
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'proof.cpp';binary=root/'proof'
            source.write_text('''#include <cmath>
#include <cstdint>
#include <cstring>
#include <cfenv>
#include <cstdio>
template<class T,class U>T word(U x){T y;std::memcpy(&y,&x,sizeof(y));return y;}
'''+f'double tiny={bounds[3].maximum!r},other={bounds[4].minimum!r},normal={normal[3].minimum!r};\n'+r'''
int main(){std::fesetround(FE_TONEAREST);unsigned long long pairs=0,selected=0,violations=0;
 for(unsigned i=10240;i<=31743;i++)for(unsigned j=10240;j<=31743;j++){
  double x=word<_Float16>(uint16_t(i)),y=word<_Float16>(uint16_t(j));
  float mean=float(float(float(x*x)+float(y*y))/2.0f+1e-6f);pairs++;
  bool a=x*x<double(mean)*0x1p-28,b=y*y<double(mean)*0x1p-28;
  if(a||b)selected++;
  float inv=float(1.0/double(float(std::sqrt(double(mean)))));
  double z0=double(_Float16(float(x*inv))),z1=double(_Float16(float(y*inv)));
  if(!a)violations+=z0<normal;
  if(!b)violations+=z1<normal;
  if(a)violations+=z0>tiny||z1!=other;
  if(b)violations+=z1>tiny||z0!=other;
 }std::printf("RMS subnormal branch cells: pairs=%llu selected=%llu violations=%llu\n",pairs,selected,violations);
 return violations?1:0;}
''')
            built=subprocess.run(['clang++','-O3','-ffp-contract=off','-std=c++17',str(source),'-o',str(binary)],capture_output=True,text=True)
            self.assertEqual(built.returncode,0,built.stderr)
            tested=subprocess.run([str(binary)],capture_output=True,text=True,timeout=120)
            self.assertEqual(tested.returncode,0,tested.stdout+tested.stderr)
            self.assertRegex(tested.stdout,r'pairs=462422016 selected=[1-9][0-9]* violations=0')
            print(tested.stdout,end='')

if __name__=='__main__':unittest.main()
