"""Cross-normalization constraints enclose perturbations and all roundings."""
import os,struct,subprocess,tempfile,unittest
from fractions import Fraction as F
from pathlib import Path
from types import SimpleNamespace as NS
from direct_sympy_projection_constraints import norm_distance,norm_error_bound,norm_links,impossible_projections
from direct_sympy_projection_constraints_test import guard
from direct_sympy_conversions import FiniteSource
from direct_sympy_checkpoint import CheckpointStrings
from direct_sympy_input_partitions import interval
from direct_sympy_streaming_literals import streaming_literals
from direct_sympy_strings import StringCompiler


def mock_registry():
    class Bounds:
        def bounds(self,node):return FiniteSource(-.001,.001) if getattr(getattr(node,'func',None),'id','') in ('CompileValue2','CompileValue3') else None
    common={'width':2,'floor':F(1),'epsilon':F(1e-6),'context':'same','gamma':(F(1),F(1)),
        'sourceNormFloor':F(1,128),'roundingError':norm_error_bound(2,1e-6)}
    old={**common,'sources':('X1','X2'),'inputMagnitudes':{0:F(2),1:F(1)},
        'components':{0:'CompileValue0()',1:'CompileValue1()'},'magnitudes':{0:F(1.5),1:F(1.5)}}
    new={**common,'sources':('CompileValue4()','CompileValue5()'),'inputMagnitudes':{0:F(3),1:F(2)},
        'components':{0:'CompileValue6()',1:'CompileValue7()'},'magnitudes':{0:F(1.5),1:F(1.5)}}
    r=NS(model=NS(norm_vectors={'old':old,'new':new},conversions=Bounds()),definition_recipes={
        4:'R16(R32(X1 + CompileValue2()))',5:'R16(R32(X2 + CompileValue3()))'})
    return r,old,new


class NormCorrelationTests(unittest.TestCase):
    def test_source_difference_includes_both_rounding_frontiers(self):
        r,a,b=mock_registry();bound=norm_distance(r,a,b)
        self.assertTrue(F(.03)<bound<F(.06))
        r.definition_recipes[4]='X1 + CompileValue2()';r.definition_recipes[5]='X2 + CompileValue3()'
        without_storage=norm_distance(r,a,b)
        self.assertLess(without_storage,bound)
        self.assertGreater(without_storage,2*a['roundingError'])

    def test_missing_large_or_foreign_context_relations_remain_unproved(self):
        for change in ('zero','epsilon','context','nonlinear','large','no_bounds'):
            r,a,b=mock_registry()
            if change=='zero':a['sourceNormFloor']=0
            if change=='epsilon':b['epsilon']=F(1)
            if change=='context':b['context']='foreign'
            if change=='nonlinear':r.definition_recipes[4]='R16(R32(X1 * X2))'
            if change=='large':r.definition_recipes[4]='R16(R32(X1 + 1))'
            if change=='no_bounds':r.model.conversions=None
            self.assertIsNone(norm_distance(r,a,b),change)

    def test_cross_constraints_require_proved_links_and_preserve_gamma_signs(self):
        r,a,b=mock_registry()
        guards=[guard('CompileValue0() * .0006 - CompileValue1() * .00025'),
            guard('CompileValue6() * -.007 - CompileValue7() * .0025')]
        self.assertFalse(impossible_projections(r,guards,max_links=0))
        self.assertTrue(impossible_projections(r,guards))
        r,a,b=mock_registry();b['gamma']=(F(-1),F(1))
        related=norm_links(r)[id(a)]
        self.assertEqual(related['CompileValue6()'][0],(F(-1),F(0)))
        self.assertEqual(related['CompileValue7()'][0],(F(0),F(1)))

    def test_inlined_scalar_matches_only_its_exact_frozen_definition(self):
        r,a,b=mock_registry();r.definition_recipes[8]='CompileValue0()'
        word='Float64(Bits64(CompileValue0()))'
        first=guard(word+' * .0006 - CompileValue1() * .00025')
        second=guard('CompileValue6() * -.007 - CompileValue7() * .0025')
        self.assertFalse(impossible_projections(r,[first,second]))
        first.view=NS(definitions=tuple(['0.0']*8+[word]))
        self.assertTrue(impossible_projections(r,[first,second]))
        altered=guard('Float64(U64Add(Bits64(CompileValue0()), 1)) * .0006 - CompileValue1() * .00025')
        altered.view=first.view
        self.assertFalse(impossible_projections(r,[altered,second]))

    def test_emitted_integer_grid_composition_keeps_rounding_error_bounds(self):
        r,a,b=mock_registry()
        raw='CompileValue0() * .0006 - CompileValue1() * .00025'
        word='Float64(U64FromF64(F64FromU64(Bits64('+raw+')) + 2**81 - 2**81 + 2**94 - 2**94))'
        second=guard('CompileValue6() * -.007 - CompileValue7() * .0025')
        self.assertTrue(impossible_projections(r,[guard(word),second]))
        self.assertFalse(impossible_projections(r,[guard(word.replace('2**94','2**95')),second]))

    @unittest.skipUnless(os.environ.get('LLM_INNER_DIRECT_JSON_CHECKPOINT'),'Checkpoint required')
    def test_original_rounded_norm_distance_and_cross_exclusions_on_all_central_signs(self):
        with CheckpointStrings(os.environ['LLM_INNER_DIRECT_JSON_CHECKPOINT'],StringCompiler(max_characters=8388608),
            input_domains={'X1':interval(-2,-.0625),'X2':interval(.0625,1)}) as m:
            with streaming_literals(m) as registry:
                m.coordinate(2);a,b=list(m.norm_vectors.values());distance=norm_distance(registry,a,b)
                self.assertIsNotNone(distance)
            eps=float(a['epsilon'])
            weights=[float(m.weight('model.layers.0.self_attn.'+name+'.weight',i,j)) for name in ('v_proj','o_proj') for i in range(2) for j in range(2)]
            head=[float(m.weight('lm_head.weight',2,i)) for i in range(2)]
            self.assertEqual(a['gamma'],(1,1));self.assertEqual(b['gamma'],(1,1))
        limit=float(distance*distance)
        if F(limit)<distance*distance:
            import math
            limit=math.nextafter(limit,math.inf)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'distance.cpp';binary=root/'distance'
            source.write_text('''#include <cmath>
#include <cstdint>
#include <cstring>
#include <cfenv>
#include <cstdio>
template<class T,class U>T word(U x){T y;std::memcpy(&y,&x,sizeof(y));return y;}
'''+''.join('double '+name+'[]={'+','.join(map(repr,values))+'};\n' for name,values in [('w',weights),('head',head)])+
                'double eps='+repr(eps)+';double distance_squared='+repr(limit)+';\n'+r'''
void norm(double x,double y,double &a,double &b){
 float mean=float(float(float(x*x)+float(y*y))/2.0f+float(eps));
 float reciprocal=float(1.0/double(float(std::sqrt(double(mean)))));
 a=double(_Float16(float(x*reciprocal)));b=double(_Float16(float(y*reciprocal)));}
double projection(double a,double b,int i){float lanes[4]={};
 lanes[0]=float(float(a*w[i])+0.0f);lanes[1]=float(float(b*w[i+1])+0.0f);
 return double(_Float16(float(float(lanes[0]+lanes[1])+float(lanes[2]+lanes[3]))));}
int main(){std::fesetround(FE_TONEAREST);unsigned cases=0,distance_fail=0,cross_fail=0;
 for(unsigned i=0x2c00;i<=0x4000;i++)for(unsigned j=0x2c00;j<=0x3c00;j++)for(int signs=0;signs<4;signs++){
  double x=word<_Float16>(uint16_t(i|((signs&1)?0x8000:0))),y=word<_Float16>(uint16_t(j|((signs&2)?0x8000:0)));
  double a,b;norm(x,y,a,b);double v0=projection(a,b,0),v1=projection(a,b,2);
  double o0=projection(v0,v1,4),o1=projection(v0,v1,6);
  double r0=double(_Float16(float(x+o0))),r1=double(_Float16(float(y+o1)));
  double c,d;norm(r0,r1,c,d);
  double squared=(a-c)*(a-c)+(b-d)*(b-d);
  double raw0=v0*w[4]+v1*w[5],raw1=v0*w[6]+v1*w[7],raw_head=c*head[0]+d*head[1];
  distance_fail+=squared>distance_squared;
  cross_fail+=std::abs(raw_head)<=0x1p-14&&(std::abs(raw0)<=0x1p-14||std::abs(raw1)<=0x1p-14);cases++;
 }std::printf("Cross norm proof: pairs=%u distanceViolations=%u exclusionViolations=%u\n",cases,distance_fail,cross_fail);
 return distance_fail||cross_fail?1:0;}
''')
            built=subprocess.run(['clang++','-O3','-ffp-contract=off','-std=c++17',str(source),'-o',str(binary)],capture_output=True,text=True)
            self.assertEqual(built.returncode,0,built.stderr)
            tested=subprocess.run([str(binary)],capture_output=True,text=True)
            self.assertEqual(tested.returncode,0,tested.stdout+tested.stderr)
            self.assertIn('pairs=83922948 distanceViolations=0 exclusionViolations=0',tested.stdout)
            print(tested.stdout,end='')


if __name__=='__main__':unittest.main()
