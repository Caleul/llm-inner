import json,multiprocessing,os,tempfile,unittest
from pathlib import Path
from direct_sympy_artifact_budget import ArtifactBudget
from direct_sympy_coherent_paths import CoherentPaths
from direct_sympy_coherent_paths_test import fixture

def claim(path,ready,start,queue,name):
    budget=ArtifactBudget(path,100)
    ready.put(name);start.wait()
    try:budget.change(name,60);queue.put(True)
    except ValueError:queue.put(False)

class ArtifactBudgetTests(unittest.TestCase):
    def test_concurrent_claims_never_exceed_shared_limit(self):
        with tempfile.TemporaryDirectory()as d:
            path=Path(d)/'budget.json';ArtifactBudget(path,100,create=True)
            context=multiprocessing.get_context('spawn');ready=context.Queue();start=context.Event();result=context.Queue()
            jobs=[context.Process(target=claim,args=(path,ready,start,result,str(i)))for i in range(2)]
            for job in jobs:job.start()
            for _ in jobs:ready.get(timeout=30)
            start.set();values=[result.get(timeout=30)for _ in jobs]
            for job in jobs:job.join(30);self.assertEqual(job.exitcode,0)
            self.assertEqual(sorted(values),[False,True]);self.assertEqual(sum(json.loads(path.read_text())['claims'].values()),60)
    def test_failure_releases_only_its_lease_and_preserves_existing_artifact(self):
        with tempfile.TemporaryDirectory()as d:
            budget=ArtifactBudget(Path(d)/'budget.json',100,create=True)
            with budget.lease('prior')as lease:lease.claim(70);lease.commit(70)
            path=Path(d)/'value.expr';path.write_text('previous')
            with self.assertRaisesRegex(ValueError,'Accumulated'):
                with budget.lease('next')as lease:
                    CoherentPaths(fixture(['Piecewise((X1,X1>0.0),(-X1,True))'])).write(path,'CompileValue0()',max_characters=65536,artifact_lease=lease)
            self.assertEqual(path.read_text(),'previous');self.assertEqual(json.loads(budget.path.read_text())['claims'],{'prior':70})
    def test_lease_accounts_complete_expression_conditions_and_final_combination(self):
        with tempfile.TemporaryDirectory()as d:
            budget=ArtifactBudget(Path(d)/'budget.json',65536,create=True);path=Path(d)/'value.expr'
            with budget.lease('coordinate')as lease:
                report=CoherentPaths(fixture(['Piecewise((X1,X1>0.0),(-X1,True))'])).write(path,'CompileValue0()',max_characters=65536,artifact_lease=lease)
            state=json.loads(budget.path.read_text());self.assertEqual(state['claims']['coordinate'],len(path.read_text()))
            self.assertEqual(report['characters'],state['claims']['coordinate']);self.assertGreater(state['peakClaimedCharacters'],report['characters'])
        if os.environ.get('LLM_INNER_DIRECT_JSON_CHECKPOINT'):
            from direct_sympy_checkpoint import CheckpointStrings
            from direct_sympy_streaming_literals import streaming_literals
            from direct_sympy_strings import StringCompiler
            with tempfile.TemporaryDirectory()as d:
                root=Path(d);budget=ArtifactBudget(root/'budget.json',65536,create=True)
                with CheckpointStrings(os.environ['LLM_INNER_DIRECT_JSON_CHECKPOINT'],StringCompiler(max_characters=65536))as model:
                    with streaming_literals(model)as registry:
                        expression=model.norm('budget-test','model.layers.0.input_layernorm.weight',0,lambda i:f'X{i+1}')
                        plain=CoherentPaths(registry).write(root/'plain.expr',expression,max_characters=65536)
                        with budget.lease('normalization')as lease:
                            shared=CoherentPaths(registry).write(root/'shared.expr',expression,max_characters=65536,artifact_lease=lease)
                self.assertEqual((root/'plain.expr').read_bytes(),(root/'shared.expr').read_bytes())
                self.assertEqual(plain['sha256'],shared['sha256'])
                print('Shared budget checkpoint normalization: identicalEmittedBytes=true')
if __name__=='__main__':unittest.main()
