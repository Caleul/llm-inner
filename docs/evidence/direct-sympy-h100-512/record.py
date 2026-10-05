import hashlib,json,re,sys
from pathlib import Path
sys.path.insert(0,str(Path('helpers').resolve()))
from direct_sympy_savepoints import digest_file
root=Path(__file__).resolve().parent
for name,number in [('budget',3),('controller',2),('coherent',22)]:
 log=(root/(name+'-tests.log')).read_text();assert f'Ran {number} tests'in log and '\nOK\n'in log,log
assert 'Shared budget checkpoint normalization: identicalEmittedBytes=true'in (root/'budget-tests.log').read_text()
assert 'completeCoordinates=4 maxWorkersLive=2 bitMismatches=0'in (root/'controller-tests.log').read_text()
assert 'cases=888832 mismatches=0'in (root/'coherent-tests.log').read_text()
def counts(name):
 text=(root/(name+'-tests.log')).read_text()
 return {k:int(re.search(r'ℹ '+k+r' (\d+)',text)[1])for k in ('tests','pass','fail','skipped')}
focused=counts('focused');general=counts('general')
assert focused=={'tests':40,'pass':40,'fail':0,'skipped':0}
assert general=={'tests':620,'pass':571,'fail':0,'skipped':49}
smoke=Path('/private/tmp/llm-inner-512-controller-smoke-20261005-final')
for name in ('run.json','artifact-budget.json','logit-2.json'):
 (root/('smoke-'+name)).write_text(json.dumps(json.loads((smoke/name).read_text()),indent=2)+'\n')
s=json.loads((root/'smoke-run.json').read_text());assert s['state']=='INCOMPLETE' and not s['firstCoordinateValidated'] and not s['vectorEmitted']
assert len(s['runs'])==1 and s['runs'][0]['dimension']==2 and not s['runs'][0]['complete']
assert s['controllerSHA256']==digest_file(Path('helpers/direct_sympy_logits_run.py'))
assert s['artifactBudgetSHA256']==digest_file(Path('helpers/direct_sympy_artifact_budget.py'))
assert json.loads((root/'smoke-artifact-budget.json').read_text())['claims']=={'vector-syntax':15}
archive=json.loads((root/'source-archive.json').read_text());assert archive['sha256']==digest_file(Path(archive['path']))
observation=json.loads((root/'access-broker-observation.json').read_text());assert not observation['H100Allocated'] and observation['activeSessions']==0
value={'defaultExpressionAndConditionBudgetBytes':536870912,'budgetSharedAcrossCoordinates':True,'RAMMonitoredSeparately':True,
 'firstCoordinateMustPassBeforeRemainingLogits':True,'CPUFactorAndSimplify':True,
 'numericalOrderChanged':False,'numericalOrIntegrityChecksRelaxed':False,
 'nativeSchedulingFixture':'Disposable checkpoint with four zero output rows; unmodified Llama not compiled',
 'schedulingFixtureCompleteCoordinates':4,'schedulingFixtureMaxWorkersLive':2,'schedulingFixtureBitMismatches':0,
 'realCheckpointBoundedAttemptDispatchesRemainingLogits':False,
 'checkpointNormalizationIdenticalWithSharedBudget':True,'nativeNormalizationCases':888832,'nativeNormalizationMismatches':0,
 'focusedTests':focused,'portableSuite':general,'accessBrokerObservation':observation,
 'sources':{name:digest_file(Path('helpers')/name)for name in ('direct_sympy_coherent_paths.py','direct_sympy_artifact_budget.py','direct_sympy_logits_run.py','direct_sympy_colab_logits_prepare.py')},
 'sourceArchive':archive,'CUDAUsed':False,'production512MiBAttemptExecuted':False,
 'fullCoordinateArtifactEmitted':False,'fullCoordinateParity':False,'multipleTokenParity':False,'fullVectorParity':False}
(root/'validation.json').write_text(json.dumps(value,indent=2)+'\n')
for file,key,data in [('docs/test-suite-map.json','memoryAdmittedLogitControllerValidation',value),('docs/direct-string-validation.json','memoryAdmittedLogitControllerValidation',value)]:
 path=Path(file);raw=path.read_text();old=json.loads(raw)
 if key in old:assert old[key]==data
 else:
  new=raw.rstrip()[:-1].rstrip()+',\n  '+json.dumps(key)+': '+json.dumps(data,indent=2).replace('\n','\n  ')+'\n}\n'
  assert {k:v for k,v in json.loads(new).items()if k!=key}==old
  path.write_text(new)
print(json.dumps({'focused':focused,'portable':general,'H100Allocated':False,'production512MiBAttemptExecuted':False}))
