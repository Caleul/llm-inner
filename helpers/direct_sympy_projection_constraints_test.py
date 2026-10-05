"""Coupled-path exclusions must enclose the original rounded operations."""
import json,math,os,struct,subprocess,tempfile,unittest
from fractions import Fraction as F
from pathlib import Path
from types import SimpleNamespace as NS
from direct_sympy_conversions import FiniteSource
from direct_sympy_projection_constraints import inverse,LinearProof,norm_squared_floor,impossible_projections
from direct_sympy_strings import syntax,StringCompiler
from direct_sympy_checkpoint import CheckpointStrings


def guard(expression,threshold=2**-14,truth=True):
    limit=struct.unpack('Q',struct.pack('d',threshold))[0]
    return NS(expression=f'U64And(Bits64({expression}), 9223372036854775807) < {limit}',truth=truth)


def registry(floor=F(1)):
    vector={'width':2,'floor':floor,'components':{0:'CompileValue0()',1:'CompileValue1()'},'magnitudes':{0:F(2),1:F(2)}}
    recipes={2:'R16(R32(CompileValue0() * .5 + CompileValue1() * .25))',
        3:'R16(R32(CompileValue0() * .25 - CompileValue1() * .5))'}
    return NS(model=NS(norm_vectors={'test':vector}),definition_recipes=recipes),vector


class ProjectionConstraintTests(unittest.TestCase):
    def test_norm_floor_requires_nonzero_finite_sources_and_directed_roots(self):
        self.assertEqual(norm_squared_floor([FiniteSource(-1,1),FiniteSource(-0.,0.)],1e-6),0)
        for sources,epsilon in (([],1e-6),([None],1e-6),([FiniteSource(0,65505)],1e-6),([FiniteSource(1,1)],0)):
            self.assertEqual(norm_squared_floor(sources,epsilon),0)
        floor=norm_squared_floor([FiniteSource(.0625,2),FiniteSource(.0625,1)],1e-6)
        self.assertTrue(F(199,100)<floor<2)
        separated=norm_squared_floor([FiniteSource(-2,2,minimum_magnitude=.0625)]*2,1e-6)
        self.assertEqual(separated,floor)

    def test_exact_inverse_and_rounding_errors_are_not_discarded(self):
        self.assertEqual(inverse([[2,1],[1,1]]),[[F(1),F(-1)],[F(-1),F(2)]])
        self.assertIsNone(inverse([[1,1],[2,2]]));self.assertIsNone(inverse([[1,2]]))
        r,v=registry();p=LinearProof(r,v)
        exact=p.form(syntax('CompileValue0() * .5 + CompileValue1() * .25'))
        rounded=p.form(syntax('CompileValue2()'))
        self.assertEqual(exact[:2],rounded[:2]);self.assertGreater(rounded[2],exact[2])
        self.assertGreater(exact[2],0)
        for expr in ('CompileValue0() * CompileValue1()','Bits64(CompileValue0())',
            'CompileValue0() / 3','CompileValue0() * 1e308','CompileValue8()'):
            self.assertIsNone(p.form(syntax(expr)))

    def test_exclusion_needs_independent_constraints_and_positive_norm_floor(self):
        r,v=registry();both=[guard('CompileValue2()'),guard('CompileValue3()')]
        self.assertTrue(impossible_projections(r,both))
        self.assertFalse(impossible_projections(r,both[:1]))
        self.assertFalse(impossible_projections(r,[both[0],both[0]]))
        self.assertFalse(impossible_projections(r,both,max_systems=0))
        self.assertFalse(impossible_projections(r,[guard('CompileValue2()',truth=False),both[1]]))
        self.assertFalse(impossible_projections(r,[NS(expression='CompileValue2() > 0',truth=True),both[1]]))
        self.assertFalse(impossible_projections(r,[guard('CompileValue2()',1),guard('CompileValue3()',1)]))
        v['floor']=0;self.assertFalse(impossible_projections(r,both))
        v['floor']=1;r.definition_recipes[3]='R16(CompileValue0() * CompileValue1())'
        self.assertFalse(impossible_projections(r,both))
        r.definition_recipes[3]='CompileValue3()';self.assertFalse(impossible_projections(r,both))

    @unittest.skipUnless(os.environ.get('LLM_INNER_DIRECT_JSON_CHECKPOINT'),'Checkpoint required')
    def test_every_half_pair_in_central_region_obeys_original_rounded_constraints(self):
        with CheckpointStrings(os.environ['LLM_INNER_DIRECT_JSON_CHECKPOINT'],StringCompiler()) as m:
            self.assertEqual(m.width,2)
            eps=struct.unpack('f',struct.pack('f',m.config['rms_norm_eps']))[0]
            gamma=[float(m.weight('model.layers.0.input_layernorm.weight',i)) for i in range(2)]
            self.assertTrue(all(x in (-1,1) for x in gamma))
            weights=[float(m.weight('model.layers.0.self_attn.'+name+'.weight',i,j)) for name in ('v_proj','o_proj') for i in range(2) for j in range(2)]
        floor=norm_squared_floor([FiniteSource(.0625,2),FiniteSource(.0625,1)],eps)
        lower=float(floor)
        if F(lower)>floor:lower=math.nextafter(lower,-math.inf)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'constraints.cpp';binary=root/'constraints'
            source.write_text('''#include <cmath>
#include <cstdint>
#include <cstring>
#include <cfenv>
#include <cstdio>
template<class T,class U>T word(U x){T y;std::memcpy(&y,&x,sizeof(y));return y;}
'''+''.join('double '+name+'[]={'+','.join(map(repr,values))+'};\n' for name,values in [('w',weights),('gamma',gamma)])+
                'double eps='+repr(eps)+';double floor_bound='+repr(lower)+';\n'+r'''
double projection(double a,double b,int i){float lanes[4]={};
 lanes[0]=float(float(a*w[i])+0.0f);lanes[1]=float(float(b*w[i+1])+0.0f);
 return double(_Float16(float(float(lanes[0]+lanes[1])+float(lanes[2]+lanes[3]))));}
int main(){std::fesetround(FE_TONEAREST);unsigned cases=0,norm_fail=0,coupled_fail=0;
 for(unsigned i=0x2c00;i<=0x4000;i++)for(unsigned j=0x2c00;j<=0x3c00;j++){
  double x=word<_Float16>(uint16_t(i)),y=word<_Float16>(uint16_t(j));
  float mean=float(float(float(x*x)+float(y*y))/2.0f+float(eps));
  float reciprocal=float(1.0/double(float(std::sqrt(double(mean)))));
  double z0=double(_Float16(float(double(_Float16(float(x*reciprocal)))*gamma[0])));
  double z1=double(_Float16(float(double(_Float16(float(y*reciprocal)))*gamma[1])));
  double v0=projection(z0,z1,0),v1=projection(z0,z1,2);
  // Exactly the original raw F64 predicates; neither the reduction nor
  // the Half/F32 frontiers are replaced by reassociated coefficients.
  double p1=z0*w[2]+z1*w[3],p2=v0*w[4]+v1*w[5],p3=v0*w[6]+v1*w[7];
  unsigned tiny=(std::abs(p1)<=0x1p-14)+(std::abs(p2)<=0x1p-14)+(std::abs(p3)<=0x1p-14);
  norm_fail+=(z0*z0+z1*z1)<floor_bound;coupled_fail+=tiny>=2;cases++;
 }std::printf("Coupled projection proof: pairs=%u normViolations=%u exclusionViolations=%u\n",cases,norm_fail,coupled_fail);
 return norm_fail||coupled_fail?1:0;}
''')
            built=subprocess.run(['clang++','-O3','-ffp-contract=off','-std=c++17',str(source),'-o',str(binary)],capture_output=True,text=True)
            self.assertEqual(built.returncode,0,built.stderr)
            tested=subprocess.run([str(binary)],capture_output=True,text=True)
            self.assertEqual(tested.returncode,0,tested.stdout+tested.stderr)
            self.assertIn('pairs=20980737 normViolations=0 exclusionViolations=0',tested.stdout)
            print(tested.stdout,end='')


if __name__=='__main__':unittest.main()
