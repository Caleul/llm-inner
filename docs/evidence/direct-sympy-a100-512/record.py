"""Bind terminal runtime observations without changing historical test results."""
import hashlib,json,sys
from pathlib import Path
folder=Path(__file__).resolve().parent
root=folder.parents[2]
local=json.loads((folder/'local-run.json').read_text())
remote=json.loads((folder/'terminal-observation.json').read_text())
run=remote['run'];allocation=json.loads((folder/'allocation.json').read_text())
archive=json.loads((folder/'source-archive.json').read_text())
assert hashlib.sha256(Path(archive['path']).read_bytes()).hexdigest()==archive['sha256']
assert run['state']in ('INCOMPLETE','COMPLETED_POSITION_ZERO','FAILED')
for name,key in [('direct_sympy_logits_run.py','controllerSHA256'),('direct_sympy_artifact_budget.py','artifactBudgetSHA256')]:
 assert local[key]==run[key]==hashlib.sha256((root/'helpers'/name).read_bytes()).hexdigest()
assert local['expressionAndConditionBudgetBytes']==run['expressionAndConditionBudgetBytes']==536870912
logs=remote['logTails']
for name,count in [('coherent_paths',22),('artifact_budget',3),('projection_sign',2)]:
 assert f'Ran {count} tests'in logs[name+'-tests.log']and '\nOK\n'in logs[name+'-tests.log']
cuda=json.loads(logs['cuda-numerics.log'])
assert cuda['halfProducts']['mismatches']==0 and cuda['silu']['mismatches']==0
assert not any(cuda['quadraticThreshold'][k]for k in ('cpuMismatches','cudaMismatches','squareBitMismatches'))
assert run['cudaNumericsExitCode']==0
assert local['runs'][0]['stats']==run['runs'][0]['stats']
assert local['runs'][0]['compilerIdentity']['sources']==run['runs'][0]['compilerIdentity']['sources']
assert not run['firstCoordinateValidated'] and not run['vectorEmitted']
assert not any(item['artifactPublished']for item in run['runs'])
value={'numericalBaselineCommit':'351aa1656ab7eb0bb990d5ddefc248ef5a8b18c2','allocation':allocation,
 'expressionAndConditionBudgetBytes':536870912,'preflightTests':{'tests':27,'passed':27},
 'CUDAUsed':True,'cudaNumerics':cuda,'localState':local['state'],'remoteState':run['state'],
 'localPeakObservedRSSBytes':local['peakObservedRSSBytes'],'remotePeakObservedRSSBytes':run['peakObservedRSSBytes'],
 'localRuns':local['runs'],'remoteRuns':run['runs'],'firstCoordinateValidated':run['firstCoordinateValidated'],
 'fullVectorParity':run['fullVectorParity'],'multipleTokenParity':run['multipleTokenParity'],
 'historicalNumericalStatesReused':False,'sources':{'controllerSHA256':run['controllerSHA256'],'artifactBudgetSHA256':run['artifactBudgetSHA256']}}
(folder/'validation.json').write_text(json.dumps(value,indent=2)+'\n')
for name in ('test-suite-map.json','direct-string-validation.json'):
 p=root/'docs'/name;raw=p.read_text();old=json.loads(raw);key='validatedA100512Runtime'
 if key in old:assert old[key]==value
 else:
  text=raw.rstrip()[:-1].rstrip()+',\n  '+json.dumps(key)+': '+json.dumps(value,indent=2).replace('\n','\n  ')+'\n}\n'
  assert {k:v for k,v in json.loads(text).items()if k!=key}==old;p.write_text(text)
print(json.dumps({'tests':27,'cudaBitMismatches':0,'state':run['state'],'firstCoordinateValidated':run['firstCoordinateValidated']}))
