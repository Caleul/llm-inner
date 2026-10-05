"""Branch-local RMS cell proofs retain the original mean and rounding."""
import struct,subprocess,tempfile,unittest
from fractions import Fraction as F
from pathlib import Path
from types import SimpleNamespace as NS
from direct_sympy_conversions import FiniteSource
from direct_sympy_projection_constraints import rms_branch_bounds
from direct_sympy_strings import Domain,StringCompiler,syntax

def fixture():
    compiler=StringCompiler();domains={'X1':Domain(F(1,32),F(65504),-15,False),'X2':Domain(F(1,32),F(65504),-15,False)}
    eps=struct.unpack('f',struct.pack('f',1e-6))[0]
    vector={'width':2,'sources':('X1','X2'),'sourceBounds':(FiniteSource(.03125,65504,-15,.03125),)*2,
        'context':compiler.context(domains),'mean':'CompileValue2()','epsilon':F(eps),'gamma':(F(1),F(1)),
        'components':{0:'CompileValue3()',1:'CompileValue4()'}}
    return NS(model=NS(compiler=compiler,domains=domains,norm_vectors={'n':vector})),vector

class BranchCellTests(unittest.TestCase):
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
