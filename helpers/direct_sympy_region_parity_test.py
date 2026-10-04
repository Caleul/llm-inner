"""Mixed signed zeros belong to the native corpus, not only all-zero vectors."""
import struct,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from direct_sympy_region_parity import sample_bits,verify_region
from direct_sympy_input_partitions import interval

def half(value):return struct.unpack('>H',struct.pack('>e',value))[0]

class CorpusTests(unittest.TestCase):
    def test_native_verifier_detects_an_error_visible_only_on_mixed_negative_zero(self):
        class Reference:
            config=SimpleNamespace(hidden_size=2,vocab_size=1)
            def eval(self):return self
            def __call__(self,inputs_embeds,**kwargs):return SimpleNamespace(logits=inputs_embeds[...,1:2])
        domains={'X1':interval(-1,-.5),'X2':interval(-1,1)}
        with tempfile.TemporaryDirectory() as directory,patch('direct_sympy_region_parity.AutoModelForCausalLM.from_pretrained',return_value=Reference()):
            artifact=Path(directory)/'coordinate.expr';artifact.write_text('X2')
            correct=verify_region('test-reference',0,artifact,domains,random_cases=0)
            self.assertEqual(correct['cases'],8);self.assertEqual(correct['mismatches'],0)
            self.assertEqual(correct['inputCorpusVersion'],2)
            artifact.write_text('0.0 + X2')
            wrong=verify_region('test-reference',0,artifact,domains,random_cases=0)
            self.assertEqual(wrong['mismatches'],2)
            self.assertTrue(all(row['inputBits'][1]==0x8000 for row in wrong['firstMismatches']))

    def test_zero_signs_are_exercised_on_one_axis_next_to_nonzero_boundaries(self):
        names,bits=sample_bits({'X2':interval(-1,1),'X1':interval(-1,-.5)},0)
        self.assertEqual(names,['X1','X2']);self.assertEqual(len(bits),8)
        for anchor in (-1,-.5):
            for sign in (0,0x8000):self.assertIn((half(anchor),sign),bits)
        self.assertEqual(sample_bits({'X1':interval(1,2)},0)[1],[(half(1),),(half(2),)])

    def test_multiple_zero_axes_include_mixed_signs_and_nonzero_neighbors(self):
        domains={'X1':interval(-1,1),'X2':interval(-1,1)}
        names,bits=sample_bits(domains,0);self.assertEqual(len(bits),16)
        for a in (0,0x8000):
            for b in (0,0x8000):self.assertIn((a,b),bits)
        for anchor in (-1,1):
            for sign in (0,0x8000):
                self.assertIn((half(anchor),sign),bits);self.assertIn((sign,half(anchor)),bits)
        self.assertEqual(sample_bits(domains,37),sample_bits(domains,37))

    def test_high_dimension_zero_coverage_is_bounded_and_invalid_counts_are_rejected(self):
        domains={'X'+str(i):interval(-1,1) for i in range(1,12)}
        names,bits=sample_bits(domains,0)
        self.assertLessEqual(len(bits),5*len(names)+4)
        for axis in range(len(names)):
            for sign in (0,0x8000):
                row=[half(-1)]*len(names);row[axis]=sign;self.assertIn(tuple(row),bits)
        for count in (-1,1.5,True):
            with self.assertRaises(ValueError):sample_bits(domains,count)
        with self.assertRaises(ValueError):sample_bits({},0)

if __name__=='__main__':unittest.main()
