"""RMS dispatch proof boundaries; these are not full-coordinate parity."""
import os
import subprocess
import tempfile
from pathlib import Path
from fractions import Fraction as F
import unittest
from direct_sympy_conversions import FiniteSource
from direct_sympy_conversions_test import cpp
from direct_sympy_strings import StringCompiler,syntax,Domain
from direct_sympy_tandem import lower_tandem


class RmsGuardTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get('LLM_INNER_DIRECT_JSON_CHECKPOINT'),'Checkpoint validation fixture not configured')
    def test_resume_rebuilds_rms_certificate_without_reusing_stale_guard_context(self):
        from direct_sympy_checkpoint import CheckpointStrings
        from direct_sympy_savepoints import ProducerSavepoints
        checkpoint=os.environ['LLM_INNER_DIRECT_JSON_CHECKPOINT']
        with tempfile.TemporaryDirectory() as directory:
            with CheckpointStrings(checkpoint,StringCompiler()) as original:
                with ProducerSavepoints(directory,original,2,compressed=True) as store:
                    def stop_after_inverse(model):
                        store.save(model)
                        if 'proof:inverse' in model.memo:raise TimeoutError('saved inverse')
                    original.on_completed=stop_after_inverse
                    with self.assertRaisesRegex(TimeoutError,'saved inverse'):
                        original.norm('proof','model.layers.0.input_layernorm.weight',0,lambda i:f'X{i+1}')
                    self.assertEqual(set(original.memo),{'proof:mean','proof:inverse'})
                    self.assertFalse(original.conversions.rms_guards)
            with CheckpointStrings(checkpoint,StringCompiler()) as resumed:
                with ProducerSavepoints(directory,resumed,2,compressed=True) as store:
                    self.assertEqual(store.restore(resumed),2)
                    self.assertFalse(resumed.conversions.rms_guards)
                    actual=resumed.norm('proof','model.layers.0.input_layernorm.weight',0,lambda i:f'X{i+1}')
                    self.assertTrue(resumed.conversions.rms_guards)
            with CheckpointStrings(checkpoint,StringCompiler()) as fresh:
                expected=fresh.norm('proof','model.layers.0.input_layernorm.weight',0,lambda i:f'X{i+1}')
                self.assertEqual(actual,expected)

    def test_dispatch_overlap_and_ordered_inverse_at_f32_boundaries(self):
        domains={'X1':Domain(-F(65504),F(65504),-24,False),
                 'X2':Domain(F(2)**-149,F(2)**128-F(2)**104,-149,True),
                 'X3':Domain(F(2)**-64,F(2)**75,-87,True)}
        certificate=FiniteSource(-F(2)**91,F(2)**91,-111)
        compiler=StringCompiler()
        result=lower_tandem('X1*X3',certificate,compiler,domains,
                            small_condition='X1**2 < X2*2**(-28)')
        baseline=lower_tandem('X1*X3',certificate,compiler,domains)
        header='#include <cstdint>\n#include <cstring>\n#include <cmath>\n#include <cstdio>\n#include <initializer_list>\ntemplate<class T,class U>T word(U v){T r;std::memcpy(&r,&v,sizeof(r));return r;}\n'
        functions=''.join('double '+name+'(double X1,double X2,double X3){return '+cpp(syntax(text))+';}\n' for name,text in [('candidate',result),('baseline',baseline)])
        body=r'''int main(){uint64_t cases=0,mismatches=0,disagreements=0;auto check=[&](double x,float mean){if(!(mean>0) || !std::isfinite(mean))return;float inverse=1.0f/std::sqrt(mean);double p=x*double(inverse);double expected=double(_Float16(float(p)));mismatches+=word<uint64_t>(candidate(x,mean,inverse))!=word<uint64_t>(expected);mismatches+=word<uint64_t>(baseline(x,mean,inverse))!=word<uint64_t>(expected);disagreements+=(x*x<double(mean)*0x1p-28)!=(std::abs(float(p))<0x1p-14);cases++;};for(unsigned bits=0;bits<65536;bits++){if((bits&0x7c00)==0x7c00)continue;double x=word<_Float16>(uint16_t(bits));for(uint32_t m:{1u,2u,0x007fffffu,0x00800000u,0x3f800000u,0x5b800000u,0x7f7fffffu})check(x,word<float>(m));float center=float(x*x*0x1p28);check(x,center);check(x,std::nextafter(center,0.0f));check(x,std::nextafter(center,INFINITY));}std::printf("RMS guard boundary parity: cases=%llu mismatches=%llu guardDisagreements=%llu\n",(unsigned long long)cases,(unsigned long long)mismatches,(unsigned long long)disagreements);return mismatches||!disagreements?1:0;}'''
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'rms.cpp';binary=root/'rms'
            source.write_text(header+functions+body)
            build=subprocess.run(['clang++','-O3','-ffp-contract=off','-std=c++17',str(source),'-o',str(binary)],capture_output=True,text=True)
            self.assertEqual(build.returncode,0,build.stderr)
            run=subprocess.run([str(binary)],capture_output=True,text=True,timeout=120)
            self.assertEqual(run.returncode,0,run.stdout+run.stderr);print(run.stdout,end='')


if __name__=='__main__':unittest.main()
