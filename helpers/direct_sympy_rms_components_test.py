"""Correlated numerical bounds; never replace the reference RMS calculation."""
from fractions import Fraction as F
import json,os,struct,subprocess,tempfile,unittest
from pathlib import Path
from direct_sympy_checkpoint import rms_component_enclosures,f32,CheckpointStrings,dominant_half_rms_source
from direct_sympy_conversions import FiniteSource
from direct_sympy_strings import StringCompiler,syntax
from direct_sympy_input_partitions import interval
from direct_sympy_conversions_test import cpp


class ComponentTests(unittest.TestCase):
    def test_dominant_square_requires_strict_original_rounding_cells(self):
        eps=f32(1e-6)
        small=FiniteSource(-0.0625,0.0625,-24)
        large=FiniteSource(-65504,-512,-24)
        self.assertEqual(dominant_half_rms_source([small,large],eps),1)
        self.assertEqual(dominant_half_rms_source([large,small],eps),0)
        self.assertIsNone(dominant_half_rms_source([small,FiniteSource(-65504,-32,-24)],eps))
        self.assertIsNone(dominant_half_rms_source([small,small],eps))
        self.assertIsNone(dominant_half_rms_source([large]*4,eps))
        self.assertIsNone(dominant_half_rms_source([None,large],eps))
        self.assertIsNone(dominant_half_rms_source([large],0))
        # Exactly a half-cell is a possible tie, never an invisible update.
        self.assertIsNone(dominant_half_rms_source([FiniteSource(1,1,-24)],2**-25))

    @unittest.skipUnless(os.environ.get('LLM_INNER_DIRECT_JSON_CHECKPOINT'),'Checkpoint fixture required')
    def test_dominant_square_kernel_and_compiled_norm_have_native_parity(self):
        with CheckpointStrings(os.environ['LLM_INNER_DIRECT_JSON_CHECKPOINT'],StringCompiler(),input_domains={'X1':interval(-0.0625,0.0625),'X2':interval(-65504,-512)}) as model:
            result=model.norm('dominant','model.layers.0.input_layernorm.weight',0,lambda i:'X'+str(i+1))
            weight=model.weight('model.layers.0.input_layernorm.weight',0)
            self.assertEqual(model.rms_dominant_squares,1)
            self.assertNotIn('sqrt(',result)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'square.cpp';binary=root/'square'
            source.write_text('#include <cmath>\n#include <cstdint>\n#include <cstring>\n#include <cstdio>\n#include <initializer_list>\n#include <cfenv>\ntemplate<class T,class U>T word(U x){T y;std::memcpy(&y,&x,sizeof(y));return y;}\ndouble candidate(double X1,double X2){return '+cpp(syntax(result))+';}\n'+r'''
int main(){std::fesetround(FE_TONEAREST);unsigned kernels=0,norms=0,fail=0;
for(unsigned b=0;b<65536;b++){if((b&0x7c00)==0x7c00 || !(b&0x7fff))continue;
 double x=word<_Float16>(uint16_t(b));
 for(int width:{1,2}){float mean=float(float(x*x)/width);float expected=float(std::sqrt(double(mean)));float actual=float(std::fabs(x)*(width==1?1.0:0.7071067811865476));fail+=word<uint32_t>(expected)!=word<uint32_t>(actual);kernels++;}
 if(x < -0.0625 || x > 0.0625)continue;
 for(double y:{-512.0,-640.0,-4604.0,-65504.0}){
  float mean=float(float(float(x*x)+float(y*y))/2.0f+1e-6f);
  float inverse=float(1.0/double(float(std::sqrt(double(mean)))));
  double normalized=double(_Float16(float(x*double(inverse))));
  double expected=double(_Float16(float(normalized*WEIGHT)));
  fail+=word<uint64_t>(candidate(x,y))!=word<uint64_t>(expected);norms++;
 }
}
std::printf("Dominant RMS square: kernels=%u normCases=%u mismatches=%u\n",kernels,norms,fail);return fail?1:0;}
'''.replace('WEIGHT',weight))
            subprocess.run(['clang++','-O3','-ffp-contract=off','-std=c++17',str(source),'-o',str(binary)],check=True,capture_output=True,text=True)
            actual=subprocess.run([str(binary)],check=True,capture_output=True,text=True)
            self.assertIn('kernels=126972',actual.stdout);self.assertIn('mismatches=0',actual.stdout);print(actual.stdout,end='')

    def test_sign_gap_zeros_and_unsafe_certificates(self):
        eps=f32(1e-6)
        negative=[FiniteSource(-1,1,-24),FiniteSource(-65504,-256,-2)]
        raw,half=rms_component_enclosures(negative,1,eps)
        self.assertEqual(half[0],(-1.4140625,-1.4140625))
        self.assertGreater(raw[1],1.4)
        union=rms_component_enclosures([FiniteSource(-2,2,-24,1),FiniteSource(2,4,-9)],0,eps)
        self.assertGreater(union[0][1],0)
        zeros=rms_component_enclosures([FiniteSource(-0.0,0.0,-24)],0,eps)
        self.assertNotEqual(struct.pack('e',zeros[1][0][0]),struct.pack('e',zeros[1][0][1]))
        for own in (FiniteSource(0,1,-24),FiniteSource(-1,0,-24)):
            enclosure=rms_component_enclosures([own],0,f32(1.7014117331926443e38))
            self.assertNotEqual(struct.pack('e',enclosure[1][0][0]),struct.pack('e',enclosure[1][0][1]))
        for epsilon in (0,2**-127,1.8e38):self.assertIsNone(rms_component_enclosures(negative,1,epsilon))
        self.assertIsNone(rms_component_enclosures([],0,eps))
        self.assertIsNone(rms_component_enclosures(negative,2,eps))
        self.assertIsNone(rms_component_enclosures([FiniteSource(-65505,65505,-24)],0,eps))

    def test_enclosures_cover_ordered_native_rms_for_all_finite_half_inputs(self):
        records=[]
        anchors=[0.0,2**-24,-2**-24,1.0,-1.0,256.0,65504.0]
        for width in (1,2,4,8):
            for eps in (2**-126,f32(1e-6),f32(1.7014117331926443e38)):
                for anchor in anchors:
                    sources=[FiniteSource(-65504,65504,-24)]+[FiniteSource(anchor,anchor,-24)]*(width-1)
                    raw,half=rms_component_enclosures(sources,0,eps)
                    records.append('{'+','.join([str(width),repr(eps),repr(anchor),*[repr(v) for v in raw[0]],*[repr(v) for v in half[0]]])+'}')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'bounds.cpp';binary=root/'bounds'
            source.write_text('#include <cmath>\n#include <cstdint>\n#include <cstring>\n#include <cstdio>\n#include <cfenv>\ntemplate<class T,class U>T word(U x){T y;std::memcpy(&y,&x,sizeof(y));return y;}\nstruct Bound{int n;double eps,anchor,lo,hi,hlo,hhi;};\nBound bounds[]={'+','.join(records)+'};\n'+r'''
int main(){if(std::fesetround(FE_TONEAREST))return 2;unsigned long long cases=0,fail=0;
for(auto b:bounds)for(unsigned bits=0;bits<65536;bits++){
 if((bits&0x7c00)==0x7c00)continue;
 double x=word<_Float16>(uint16_t(bits));float sum=0;
 if(b.n<4){for(int j=0;j<b.n;j++){double v=j?b.anchor:x;sum=float(sum+float(v*v));}}
 else {float lanes[4]={};for(int l=0;l<4;l++)for(int j=l;j<b.n;j+=4){double v=j?b.anchor:x;lanes[l]=float(lanes[l]+float(v*v));}sum=lanes[0];for(int l=1;l<4;l++)sum=float(sum+lanes[l]);}
 float mean=float(float(sum/float(b.n))+float(b.eps));float inverse=float(1.0f/std::sqrt(mean));
 double raw=x*double(inverse),rounded=float(raw),half=double(_Float16(rounded));
 fail+=raw<b.lo||raw>b.hi;fail+=rounded<b.lo||rounded>b.hi;fail+=half<b.hlo||half>b.hhi;cases++;
}std::printf("Correlated RMS enclosures: cases=%llu violations=%llu\n",cases,fail);return fail?1:0;}
''')
            subprocess.run(['clang++','-O3','-ffp-contract=off','-std=c++17',str(source),'-o',str(binary)],check=True,capture_output=True,text=True)
            result=subprocess.run([str(binary)],check=True,capture_output=True,text=True)
            self.assertIn('cases=5332992 violations=0',result.stdout);print(result.stdout,end='')

    @unittest.skipUnless(os.environ.get('LLM_INNER_DIRECT_JSON_CHECKPOINT'),'Checkpoint fixture required')
    def test_constant_component_elides_mean_and_inverse_before_expansion_with_native_parity(self):
        with CheckpointStrings(os.environ['LLM_INNER_DIRECT_JSON_CHECKPOINT'],StringCompiler(),input_domains={'X1':interval(-1,1),'X2':interval(-65504,-256)}) as model:
            result=model.norm('component','model.layers.0.input_layernorm.weight',1,lambda i:'X'+str(i+1))
            self.assertEqual(model.rms_constant_components,1)
            self.assertEqual(set(model.memo),{'component:1'})
            self.assertNotIn('X',result)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'constant.cpp';binary=root/'constant'
            source.write_text('#include <cmath>\n#include <cstdint>\n#include <cstring>\n#include <cstdio>\n#include <initializer_list>\n#include <cfenv>\ntemplate<class T,class U>T word(U x){T y;std::memcpy(&y,&x,sizeof(y));return y;}\ndouble candidate(double X1,double X2){return '+cpp(syntax(result))+';}\n'+r'''
int main(){if(std::fesetround(FE_TONEAREST))return 2;unsigned cases=0,fail=0;for(unsigned bits=0;bits<65536;bits++){if((bits&0x7c00)==0x7c00)continue;double x=word<_Float16>(uint16_t(bits));if(x < -1 || x > 1)continue;for(double y:{-256.0,-320.0,-4604.0,-65504.0}){float sum=float(float(x*x)+float(y*y));float mean=float(float(sum/2.0f)+1e-6f);float inverse=1.0f/std::sqrt(mean);double expected=double(_Float16(float(y*double(inverse))));fail+=word<uint64_t>(candidate(x,y))!=word<uint64_t>(expected);cases++;}}std::printf("Early RMS constant component: cases=%u mismatches=%u\n",cases,fail);return fail?1:0;}
''')
            subprocess.run(['clang++','-O3','-ffp-contract=off','-std=c++17',str(source),'-o',str(binary)],check=True,capture_output=True,text=True)
            result=subprocess.run([str(binary)],check=True,capture_output=True,text=True)
            self.assertIn('cases=122888 mismatches=0',result.stdout);print(result.stdout,end='')


if __name__=='__main__':unittest.main()
