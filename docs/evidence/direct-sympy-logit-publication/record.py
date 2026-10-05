import json,re,sys
from pathlib import Path
sys.path.insert(0,str(Path('helpers').resolve()))
from direct_sympy_savepoints import digest_file
root=Path(__file__).resolve().parent
controller=(root/'controller-tests.log').read_text()
assert 'Ran 3 tests'in controller and '\nOK\n'in controller
assert 'completeCoordinates=4 maxWorkersLive=2 bitMismatches=0'in controller
assert 'parityMismatch=refused changedCandidate=refused verifierCrash=refused'in controller
log=(root/'integration-tests.log').read_text()
counts={key:int(re.search(r'ℹ '+key+r' (\d+)',log)[1])for key in ('tests','pass','fail','skipped')}
assert counts=={'tests':2,'pass':2,'fail':0,'skipped':0}
smoke=json.loads((root/'smoke-run.json').read_text());assert smoke['state']=='INCOMPLETE' and not smoke['firstCoordinateValidated'] and not smoke['vectorEmitted']
assert len(smoke['runs'])==1 and not smoke['runs'][0]['artifactPublished'] and smoke['runs'][0]['producersCompleted']==1
assert smoke['controllerSHA256']==digest_file(Path('helpers/direct_sympy_logits_run.py'))
assert json.loads((root/'smoke-artifact-budget.json').read_text())['claims']=={'vector-syntax':15}
observation=json.loads((root/'access-broker-observation.json').read_text())
assert observation['creationTerminalExitCode']==1 and observation['postRequestActiveSessions']==0
archive=json.loads((root/'source-archive.json').read_text());assert archive['sha256']==digest_file(Path(archive['path']))
value={'baselineCommit':'5af9ea55a49807cc102987b3ecd056a2daf110e7','publicationRequiresExactParityAndMatchingHashes':True,
 'pendingCandidatePublishedBeforeParity':False,'failedParityPublishesArtifact':False,'changedCandidatePublishesArtifact':False,
 'verifierCrashPublishesArtifact':False,'existingArtifactsPreserved':True,'failedClaimsReleased':True,
 'failedWorkersRetainProducerAndBranchStatistics':True,'expressionAndConditionBudgetBytes':536870912,
 'numericalOrderChanged':False,'numericalOrIntegrityChecksRelaxed':False,'historicalNumericalStatesReused':False,
 'controllerTests':{'tests':3,'passed':3},'affectedIntegrations':counts,
 'schedulingFixture':'Disposable zero-head checkpoint; not original Llama output parity',
 'realCheckpointBoundedAttemptDispatchesRemainingLogits':False,'accessBrokerObservation':observation,'sourceArchive':archive,
 'sources':{name:digest_file(Path('helpers')/name)for name in ('direct_sympy_logits_run.py','direct_sympy_logits_run_test.py')},
 'CUDAUsed':False,'production512MiBAttemptExecuted':False,'fullCoordinateArtifactEmitted':False,
 'fullCoordinateParity':False,'multipleTokenParity':False,'fullVectorParity':False}
branch=root/'branch-smoke-run.json'
if branch.exists():
 b=json.loads(branch.read_text());assert b['state']=='INCOMPLETE' and not b['firstCoordinateValidated']
 assert len(b['runs'])==1 and not b['runs'][0]['artifactPublished'] and b['runs'][0]['stats']['completedPaths']>0
 assert b['controllerSHA256']==value['sources']['direct_sympy_logits_run.py']
 value['realCheckpointGeneratedCandidatesAtBudgetStop']=b['runs'][0]['stats']['completedPaths']
 value['realCheckpointProducersAtBudgetStop']=b['runs'][0]['producersCompleted']
(root/'validation.json').write_text(json.dumps(value,indent=2)+'\n')
for file,key in [('docs/test-suite-map.json','validatedLogitPublication'),('docs/direct-string-validation.json','validatedLogitPublication')]:
 path=Path(file);raw=path.read_text();old=json.loads(raw)
 if key in old:assert old[key]==value
 else:
  new=raw.rstrip()[:-1].rstrip()+',\n  '+json.dumps(key)+': '+json.dumps(value,indent=2).replace('\n','\n  ')+'\n}\n'
  assert {k:v for k,v in json.loads(new).items()if k!=key}==old
  path.write_text(new)
print(json.dumps({'affectedIntegrations':counts,'H100Allocated':False,'fullCoordinateParity':False}))
