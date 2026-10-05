"""Real worker isolation and first-coordinate admission, using disposable fixtures."""
import argparse,json,os,tempfile,unittest
from pathlib import Path
from direct_sympy_logits_run import run,DEFAULT_EXPRESSION_BYTES

class LogitControllerTests(unittest.TestCase):
    def test_default_cap_is_512_mib_and_absent_h100_does_not_dispatch(self):
        import torch
        self.assertEqual(DEFAULT_EXPRESSION_BYTES,512*1024**2)
        if torch.cuda.is_available():self.skipTest('This test requires the CPU-only local runtime')
        with tempfile.TemporaryDirectory()as d:
            args=argparse.Namespace(output=str(Path(d)/'job'),require_gpu='H100',memory_mib=1024,expression_bytes=DEFAULT_EXPRESSION_BYTES,workers=4)
            self.assertEqual(run(args),1)
            report=json.loads((Path(args.output)/'run.json').read_text())
            self.assertEqual(report['state'],'BLOCKED');self.assertFalse(report['runs']);self.assertEqual(report['maxWorkersLive'],0)

    @unittest.skipUnless(os.environ.get('LLM_INNER_DIRECT_JSON_CHECKPOINT'),'Checkpoint fixture not configured')
    def test_complete_coordinate_parity_precedes_real_parallel_remaining_logits(self):
        import torch
        from safetensors.torch import load_file,save_file
        checkpoint=Path(os.environ['LLM_INNER_DIRECT_JSON_CHECKPOINT'])
        with tempfile.TemporaryDirectory()as d:
            root=Path(d);derived=root/'checkpoint';derived.mkdir()
            (derived/'config.json').write_bytes((checkpoint/'config.json').read_bytes())
            weights=load_file(str(checkpoint/'model.safetensors'))
            # Test dead-dependency semantics, not a replacement of the real
            # checkpoint: four zero output rows make every output constant.
            weights['lm_head.weight']=torch.zeros_like(weights['lm_head.weight'])
            save_file(weights,str(derived/'model.safetensors'))
            args=argparse.Namespace(checkpoint=str(derived),output=str(root/'job'),require_gpu='none',memory_mib=3072,
                expression_bytes=1024,workers=2,dimension=2,max_paths=16,parity_cases=8,worker_seconds=120)
            self.assertEqual(run(args),0)
            report=json.loads((Path(args.output)/'run.json').read_text())
            self.assertTrue(report['firstCoordinateValidated']);self.assertTrue(report['vectorEmitted']);self.assertFalse(report['multipleTokenParity'])
            self.assertEqual(report['runs'][0]['dimension'],2)
            self.assertEqual({r['dimension']for r in report['runs']},{0,1,2,3})
            self.assertEqual(report['maxWorkersLive'],2)
            self.assertLessEqual(report['peakObservedRSSBytes'],report['memoryBudgetBytes'])
            for r in report['runs']:
                self.assertTrue(r['complete']);self.assertTrue(r['parityVerified']);self.assertEqual(r['parity']['mismatches'],0)
            self.assertLessEqual(sum(json.loads((Path(args.output)/'artifact-budget.json').read_text())['claims'].values()),1024)
            text=(Path(args.output)/'vector.expr').read_text();self.assertNotIn('CompileValue',text)
            self.assertEqual(len(text),report['vectorCharacters'])
            print(f'Parallel logit controller: completeCoordinates=4 maxWorkersLive=2 bitMismatches=0 vectorCharacters={len(text)}')
if __name__=='__main__':unittest.main()
