import argparse,json,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from direct_sympy_partition_parallel import compile_wave,run
from direct_sympy_checkpoint import CheckpointStrings
from direct_sympy_cover_regions import live_identity
from direct_sympy_partition_run import encode,seed_update_cells,audit_tree
from direct_sympy_savepoints import atomic,canonical,digest_file
from direct_sympy_strings import StringCompiler


class ParallelRegionTests(unittest.TestCase):
    def test_memory_admission_rejects_before_starting_a_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'out.expr'
            with patch('direct_sympy_partition_parallel.rss_bytes',return_value=100),patch('direct_sympy_partition_parallel.subprocess.Popen',side_effect=AssertionError('must not start')):
                with self.assertRaisesRegex(ValueError,'cannot admit'):
                    compile_wave('unused',2,[('',{'X1':[1,2]},path)],workers=2,memory_bytes=200,cas_characters=1000,max_characters=1000,max_paths=2,max_seconds=1)
            self.assertFalse(path.exists())

    def args(self,checkpoint,directory,workers):
        return argparse.Namespace(checkpoint=checkpoint,state=directory,dimension=2,workers=workers,memory_bytes=3*1024**3,
            max_attempts=4,region_seconds=20,random_cases=64,max_paths=128,max_characters=1048576,
            cas_characters=8388608,min_values=32,total_characters=67108864,frontier_order=None)

    def seed(self,directory):
        checkpoint=os.environ['LLM_INNER_DIRECT_JSON_CHECKPOINT']
        with CheckpointStrings(checkpoint,StringCompiler()) as model:
            root=encode(model.domains);tree,geometry=seed_update_cells(model,root,81)
        state={'identity':live_identity(checkpoint,2),'root':root,'tree':tree,'updateCellGeometry':geometry,'attempts':[],
            'coveredInputPatterns':0,'unfinishedInputPatterns':63488**2,'totalInputPatterns':63488**2,
            'finalArtifactEmitted':False,'finalParity':False}
        atomic(Path(directory)/'frontier.json',canonical(state))
        return checkpoint

    def completed_seed(self,directory,body):
        checkpoint=self.seed(directory);manifest=Path(directory)/'frontier.json'
        state=json.loads(manifest.read_text());path=Path(directory)/'body.expr';atomic(path,body.encode())
        for node in state['tree'].values():
            if node['status']=='pending':
                node.update(status='complete',artifact={'file':path.name,'characters':len(body),'sha256':digest_file(path)})
        covered,pending=audit_tree(state['tree'],state['root'])
        state.update(coveredInputPatterns=covered,unfinishedInputPatterns=pending)
        atomic(manifest,canonical(state));return checkpoint,state

    @unittest.skipUnless(os.environ.get('LLM_INNER_DIRECT_JSON_CHECKPOINT'),'Checkpoint required')
    def test_final_publication_coalesces_and_verifies_exact_candidate_before_admission(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);checkpoint,state=self.completed_seed(root,'X1 * 0.5 + X2 * 0.25')
            final=root/'coordinate.expr';final.write_text('previous')
            checked=[]
            def verify(checkpoint,dimension,candidate,domains,**kwargs):
                self.assertEqual(final.read_text(),'previous')
                self.assertEqual(encode(domains),state['root'])
                text=candidate.read_text();self.assertEqual(text.count('X1'),1);self.assertEqual(text.count('X2'),1)
                checked.append(candidate.read_bytes())
                return {'cases':123,'mismatches':0,'artifact':str(candidate),'sha256':digest_file(candidate)}
            with (patch('direct_sympy_partition_parallel.compile_wave',side_effect=AssertionError('already complete')),
                  patch('direct_sympy_partition_parallel.verify_region',side_effect=verify)):
                report=run(self.args(checkpoint,root,2))
            saved=json.loads((root/'frontier.json').read_text())
            self.assertEqual(final.read_bytes(),checked[0]);self.assertFalse(saved['finalParity'])
            self.assertTrue(saved['positionZeroArtifactParityVerified'])
            self.assertEqual(saved['finalParityEvidence']['cases'],123)
            self.assertEqual(saved['finalParityEvidence']['artifact'],str(final))
            self.assertEqual(saved['finalParityScope'],'position-zero-native-corpus')
            self.assertEqual(saved['finalCoalescing']['completeRegions'],9)
            self.assertEqual(saved['finalCoalescing']['distinctExpressions'],1)
            self.assertEqual(saved['finalArtifact']['sha256'],digest_file(final))
            self.assertFalse(report['fullCoordinateParity']);self.assertTrue(report['positionZeroArtifactParityVerified'])
            self.assertFalse(list(root.glob('.coordinate-*.expr')))

    @unittest.skipUnless(os.environ.get('LLM_INNER_DIRECT_JSON_CHECKPOINT'),'Checkpoint required')
    def test_final_candidate_modified_during_parity_is_not_published(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);checkpoint,_=self.completed_seed(root,'X1')
            manifest=root/'frontier.json';before=manifest.read_bytes()
            final=root/'coordinate.expr';final.write_text('previous')
            def tamper(checkpoint,dimension,candidate,domains,**kwargs):
                digest=digest_file(candidate);candidate.write_text('1.0')
                return {'cases':123,'mismatches':0,'sha256':digest}
            with patch('direct_sympy_partition_parallel.verify_region',side_effect=tamper):
                with self.assertRaisesRegex(ValueError,'integrity changed'):run(self.args(checkpoint,root,2))
            self.assertEqual(manifest.read_bytes(),before);self.assertEqual(final.read_text(),'previous')
            self.assertFalse(list(root.glob('.coordinate-*.expr')))

    @unittest.skipUnless(os.environ.get('LLM_INNER_DIRECT_JSON_CHECKPOINT'),'Checkpoint required')
    def test_real_final_parity_failure_preserves_previous_artifact_and_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);checkpoint,_=self.completed_seed(root,'0.0')
            manifest=root/'frontier.json';before=manifest.read_bytes()
            final=root/'coordinate.expr';final.write_text('previous')
            with self.assertRaisesRegex(ValueError,'Combined coordinate parity failed'):
                run(self.args(checkpoint,root,2))
            self.assertEqual(manifest.read_bytes(),before);self.assertEqual(final.read_text(),'previous')
            self.assertFalse(list(root.glob('.coordinate-*.expr')))
        print('Final publication proof: coalescedBody=true actualCheckpointRejectsWrongOutput=true atomic=true')

    @unittest.skipUnless(os.environ.get('LLM_INNER_DIRECT_JSON_CHECKPOINT'),'Checkpoint required')
    def test_timeout_retries_same_domain_before_real_size_failure_subdivision(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint=self.seed(directory);args=self.args(checkpoint,directory,1)
            args.max_attempts=1;args.region_seconds=10
            before=json.loads((Path(directory)/'frontier.json').read_text())
            def timeout(checkpoint,dimension,jobs,**budgets):
                self.assertEqual(budgets['job_seconds'],[10])
                return [{'complete':False,'stop':'Region worker wall-clock budget exceeded','inputDomains':jobs[0][1]}],{}
            with patch('direct_sympy_partition_parallel.compile_wave',side_effect=timeout):run(args)
            first=json.loads((Path(directory)/'frontier.json').read_text())
            self.assertEqual(first['tree']['00']['domains'],before['tree']['00']['domains'])
            self.assertEqual(first['tree']['00']['status'],'pending')
            self.assertEqual(first['tree']['00']['timeoutRetries'],1)
            self.assertNotIn('000',first['tree'])
            def size_failure(checkpoint,dimension,jobs,**budgets):
                self.assertEqual(budgets['job_seconds'],[30])
                self.assertEqual(jobs[0][1],before['tree']['00']['domains'])
                return [{'complete':False,'stop':'Flat artifact budget exceeded; no complete result','inputDomains':jobs[0][1]}],{}
            with patch('direct_sympy_partition_parallel.compile_wave',side_effect=size_failure):run(args)
            second=json.loads((Path(directory)/'frontier.json').read_text())
            self.assertEqual(second['tree']['00']['status'],'split')
            self.assertEqual(audit_tree(second['tree'],second['root']),(0,63488**2))
        print('Timeout retry proof: sameDomain=true deadlines=10,30 sizeFailurePreservesFullRoot=true')

    @unittest.skipUnless(os.environ.get('LLM_INNER_DIRECT_JSON_CHECKPOINT'),'Checkpoint required')
    def test_source_incompatibility_and_memory_failure_preserve_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint=self.seed(directory);manifest=Path(directory)/'frontier.json'
            state=json.loads(manifest.read_text());state['identity']['version']=-1
            atomic(manifest,canonical(state));before=manifest.read_bytes()
            with patch('direct_sympy_partition_parallel.compile_wave',side_effect=AssertionError('must reject first')):
                with self.assertRaisesRegex(ValueError,'Incompatible'):run(self.args(checkpoint,directory,2))
            self.assertEqual(manifest.read_bytes(),before)
            checkpoint=self.seed(directory);before=manifest.read_bytes()
            args=self.args(checkpoint,directory,2);args.memory_bytes=1
            with self.assertRaisesRegex(ValueError,'memory'):run(args)
            self.assertEqual(manifest.read_bytes(),before)
            self.assertEqual(list(Path(directory).glob('.parallel-*.expr')),[])

    @unittest.skipUnless(os.environ.get('LLM_INNER_DIRECT_JSON_CHECKPOINT'),'Checkpoint required')
    def test_one_and_two_workers_emit_identical_complete_regions_and_exact_coverage(self):
        reports=[];states=[]
        with tempfile.TemporaryDirectory() as directory:
            for workers in (1,2):
                path=Path(directory)/str(workers);path.mkdir();checkpoint=self.seed(path)
                report=run(self.args(checkpoint,path,workers));state=json.loads((path/'frontier.json').read_text())
                self.assertEqual(report['nativeMismatches'],0)
                self.assertEqual(report['attempts'],4)
                self.assertEqual(report['coveredInputPatterns'],553648128)
                self.assertEqual(audit_tree(state['tree'],state['root']),(553648128,3477078016))
                self.assertEqual(max(w['maxWorkersLive'] for w in report['waves']),workers)
                self.assertTrue(all(w['peakObservedRSSBytes']<=report['memoryBudgetBytes'] for w in report['waves']))
                self.assertFalse(state['finalArtifactEmitted']);self.assertFalse(state['finalParity'])
                reports.append(report);states.append(state)
            rows=lambda state:{key:(node['domains'],node['artifact']['sha256']) for key,node in state['tree'].items() if node['status']=='complete'}
            self.assertEqual(rows(states[0]),rows(states[1]))
        result={'sequential':reports[0],'parallel':reports[1],
            'identicalDomainsAndArtifactHashes':True,'coveredPatterns':553648128,
            'speedRatio':reports[0]['seconds']/reports[1]['seconds'],'fullCoordinateParity':False}
        if os.environ.get('LLM_INNER_PARTITION_BENCHMARK_REPORT'):
            Path(os.environ['LLM_INNER_PARTITION_BENCHMARK_REPORT']).write_text(json.dumps(result,indent=2)+'\n')
        print('Parallel partition proof: workers=1,2 patterns=553648128 identicalHashes=true mismatches=0')


if __name__=='__main__':unittest.main()
