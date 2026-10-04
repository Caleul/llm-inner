"""Exercise actual budget stops, persistent continuation and identity rejection."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class CheckpointRunTests(unittest.TestCase):
    @unittest.skipUnless(sys.platform.startswith('linux'),'Linux mapped-address-space admission')
    def test_memory_cap_below_imported_libraries_is_rejected_before_creating_state(self):
        helper=Path(__file__).with_name('direct_sympy_checkpoint_run.py')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            process=subprocess.run([sys.executable,str(helper),'unused-checkpoint',str(root/'state'),
                str(root/'report.json'),'--max-address-space-mib','1'],capture_output=True,text=True,timeout=60)
            self.assertEqual(process.returncode,2,process.stdout+process.stderr)
            self.assertIn('must exceed already mapped compiler libraries',process.stderr)
            self.assertFalse((root/'state').exists())

    @unittest.skipUnless(os.environ.get('LLM_INNER_DIRECT_JSON_CHECKPOINT'),'Checkpoint validation fixture not configured')
    def test_real_checkpoint_continues_after_budget_stop_and_rejects_changed_dimension(self):
        checkpoint=os.environ['LLM_INNER_DIRECT_JSON_CHECKPOINT']
        helper=Path(__file__).with_name('direct_sympy_checkpoint_run.py')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);state=root/'state';report=root/'report.json'
            def run(*arguments):
                process=subprocess.run([sys.executable,str(helper),checkpoint,str(state),str(report),
                    '--max-seconds','60',*arguments],capture_output=True,text=True,timeout=90)
                self.assertEqual(process.returncode,1,process.stdout+process.stderr)
                return json.loads(report.read_text())
            first=run('--max-characters',str(1024**2))
            self.assertFalse(first['coordinateComplete']);self.assertFalse(first['parityVerified'])
            self.assertEqual(first['persistedDependencies'],10)
            self.assertEqual(first['completedDependencies'],first['persistedDependencies'])
            manifest=state/'frontier.json';initial=manifest.read_bytes()
            self.assertNotIn('output:0:2',[r['name'] for r in json.loads(initial)['payload']['records']])
            rejected=run('--resume','--dimension','1','--max-characters',str(4*1024**2))
            self.assertFalse(rejected['resumeCompatible']);self.assertIn('Incompatible',rejected['stop'])
            self.assertEqual(manifest.read_bytes(),initial)
            resumed=run('--resume','--max-characters',str(4*1024**2))
            self.assertTrue(resumed['resumeCompatible'])
            self.assertEqual(resumed['restoredDependencies'],10)
            self.assertGreater(resumed['persistedDependencies'],10)
            self.assertEqual(resumed['completedDependencies'],resumed['persistedDependencies'])
            self.assertFalse(resumed['coordinateComplete']);self.assertFalse(resumed['parityVerified'])
            old_records=json.loads(initial)['payload']['records']
            new_records=json.loads(manifest.read_text())['payload']['records']
            self.assertEqual(new_records[:len(old_records)],old_records)
            self.assertGreater(resumed['peakRSSBytes'],0)
            print('Persistent checkpoint run: '+json.dumps({'restored':10,'persisted':len(new_records),
                'identityRejectionBeforeMutation':True,'coordinateComplete':False}))


if __name__=='__main__':unittest.main()
