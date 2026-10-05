"""Inverse certificates must enclose the actual ordered stored-Half pipeline."""
import struct,subprocess,tempfile,unittest
from fractions import Fraction as F
from pathlib import Path
from types import SimpleNamespace as NS
from direct_sympy_conversions import FiniteSource,ConversionSession
from direct_sympy_strings import Domain,StringCompiler,syntax
from direct_sympy_recipe_bounds import conversion_preimage,propagate_recipe_bounds


def fixture():
    domains={'X1':Domain(F(-2**-10),F(2**-10),-24,False)};compiler=StringCompiler()
    model=NS(compiler=compiler,domains=domains,conversions=ConversionSession(compiler,domains,input_dtype='f16'))
    bounds=[(-2**-10,2**-10),(-1,1),(-1/64,1/64),(0,2**-12),(0,2**-18),(-2**-9,2**-9)]
    return NS(model=model,definition_proofs=[(FiniteSource(a,b,-24),'half',False,None)for a,b in bounds],
        definition_recipes={0:'X1',1:'R16(R32(CompileValue0()*1000.0))',
        2:'R16(R32(CompileValue1()*0.015625))',3:'R16(R32(CompileValue2()*CompileValue2()))',
        4:'R16(R32(CompileValue3()*0.015625))',5:'R16(R32(CompileValue0()+CompileValue4()))'})


class RecipeBoundsTests(unittest.TestCase):
    def test_raw_projection_guard_restricts_sources_through_a_varying_positive_inverse(self):
        r=fixture();r.definition_proofs=[(FiniteSource(1,2000,-149),'f32',True,None),
            (FiniteSource(-1,1,-24),'half',False,None)]
        r.definition_recipes={0:'R32(1.0/R32(sqrt(R32(R32(X1**2)+1e-6))))',
            1:'R16(R32(X1*CompileValue0()))'}
        limit=struct.unpack('Q',struct.pack('d',2**-14))[0]
        condition=syntax(f'U64And(Bits64(CompileValue1()*0.015625),9223372036854775807)<{limit}')
        result=propagate_recipe_bounds(r,{},assumptions=[(condition,True)])
        self.assertIsNotNone(result);self.assertLess(result['X1'].maximum,2**-17)
        with tempfile.TemporaryDirectory()as directory:
            root=Path(directory);source=root/'varying.cpp';binary=root/'varying'
            source.write_text('#include <cstdint>\n#include <cstring>\n#include <cmath>\n#include <cstdio>\n#include <cfenv>\n#include <initializer_list>\ntemplate<class T,class U>T word(U x){T y;std::memcpy(&y,&x,sizeof(y));return y;}\nint main(){if(std::fesetround(FE_TONEAREST))return 2;unsigned cases=0,selected=0,failures=0;for(unsigned i=0;i<=0x1400;i++)for(unsigned sign:{0u,0x8000u}){double x=word<_Float16>(uint16_t(i|sign));float m=float(double(float(x*x))+1e-6);float inverse=float(1.0/double(float(std::sqrt(double(m)))));double h=double(_Float16(float(x*inverse)));cases++;if(std::abs(h*0.015625)<0x1p-14){selected++;failures+=x<'+repr(result['X1'].minimum)+'||x>'+repr(result['X1'].maximum)+';}}std::printf("Backward varying inverse certificate: cases=%u selected=%u violations=%u\\n",cases,selected,failures);return failures||!selected?1:0;}')
            subprocess.run(['clang++','-O3','-ffp-contract=off','-std=c++17',str(source),'-o',str(binary)],check=True,capture_output=True,text=True)
            checked=subprocess.run([str(binary)],check=True,capture_output=True,text=True)
            self.assertIn('cases=10242 selected=132 violations=0',checked.stdout);print(checked.stdout,end='')

    def test_inverse_cells_retain_ties_subnormals_signs_and_finite_overflow_edges(self):
        for kind,unit in (('R16',2**-24),('R32',2**-149)):
            b=conversion_preimage(FiniteSource(-0.,0.),kind)
            self.assertEqual((b.minimum,b.maximum),(-unit/2,unit/2))
            b=conversion_preimage(FiniteSource(unit,unit),kind)
            self.assertEqual((b.minimum,b.maximum),(unit/2,1.5*unit))
            b=conversion_preimage(FiniteSource(-unit,-unit),kind)
            self.assertEqual((b.minimum,b.maximum),(-1.5*unit,-unit/2))
        b=conversion_preimage(FiniteSource(1,1),'R16')
        self.assertEqual((b.minimum,b.maximum),(1-2**-12,1+2**-11))
        self.assertEqual(conversion_preimage(FiniteSource(65504,65504),'R16').maximum,65520)
        self.assertIsNone(conversion_preimage(FiniteSource(-1e308,1e308),'R32'))

    def test_backward_residual_then_forward_storage_eliminates_only_proved_magnitudes(self):
        r=fixture();stats={};unit=2**-24
        result=propagate_recipe_bounds(r,{5:FiniteSource(-unit,unit,-24)},stats)
        self.assertIsNotNone(result)
        self.assertEqual((result['X1'].minimum,result['X1'].maximum),(-unit,unit))
        for key in (3,4):self.assertEqual((result[key].minimum,result[key].maximum),(0.,0.))
        self.assertGreater(stats['recipeConstraintPasses'],1)
        self.assertGreater(stats['backwardRoundingSteps'],0)
        self.assertEqual(r.model.domains['X1'].maximum,F(2**-10))
        with tempfile.TemporaryDirectory()as directory:
            root=Path(directory);source=root/'inverse.cpp';binary=root/'inverse'
            source.write_text('#include <cstdint>\n#include <cstring>\n#include <cmath>\n#include <cstdio>\n#include <cfenv>\n#include <initializer_list>\ntemplate<class T,class U>T word(U x){T y;std::memcpy(&y,&x,sizeof(y));return y;}\nint main(){if(std::fesetround(FE_TONEAREST))return 2;unsigned cases=0,selected=0,failures=0;for(unsigned i=0;i<=0x1400;i++)for(unsigned sign:{0u,0x8000u}){double x=word<_Float16>(uint16_t(i|sign));double a=double(_Float16(float(x*1000.0)));double b=double(_Float16(float(a*0.015625)));double c=double(_Float16(float(b*b)));double d=double(_Float16(float(c*0.015625)));double y=double(_Float16(float(x+d)));cases++;if(std::abs(y)<=0x1p-24){selected++;failures+=x<'+repr(result['X1'].minimum)+'||x>'+repr(result['X1'].maximum)+'||c!=0.0||d!=0.0;}}std::printf("Backward residual certificate: cases=%u selected=%u violations=%u\\n",cases,selected,failures);return failures||!selected?1:0;}')
            subprocess.run(['clang++','-O3','-ffp-contract=off','-std=c++17',str(source),'-o',str(binary)],check=True,capture_output=True,text=True)
            checked=subprocess.run([str(binary)],check=True,capture_output=True,text=True)
            self.assertIn('cases=10242 selected=4 violations=0',checked.stdout);print(checked.stdout,end='')

    def test_contradictions_require_proof_and_work_limits_keep_reachable_domains(self):
        r=fixture();r.model.domains={'X1':Domain(F(2**-20),F(2**-19),-24,False)}
        self.assertIsNone(propagate_recipe_bounds(r,{5:FiniteSource(-2**-24,2**-24,-24)}))
        for limits in ({'max_passes':0},{'max_nodes':0}):
            r=fixture();stats={};result=propagate_recipe_bounds(r,{5:FiniteSource(-2**-24,2**-24,-24)},stats,**limits)
            self.assertIsNotNone(result);self.assertGreater(result.get('X1',FiniteSource(-2**-10,2**-10)).maximum,2**-24)
            self.assertGreater(stats['recipeConstraintBudgetStops'],0)
        r=fixture();r.definition_recipes={5:'unknown(CompileValue0())'}
        result=propagate_recipe_bounds(r,{5:FiniteSource(-2**-24,2**-24,-24)})
        self.assertNotIn('X1',result)


if __name__=='__main__':unittest.main()
