"""Reconcile actual expression parity, prefix growth and regression scope."""
import json,re,sys
from pathlib import Path
sys.path.insert(0,str(Path('helpers').resolve()))
from direct_sympy_cover_regions import live_identity
from direct_sympy_savepoints import digest_file
root=Path(__file__).resolve().parent
identity=live_identity('docs/evidence/direct-sympy-test-checkpoint',2)
a,b,d32,d96,run=[json.loads((root/name).read_text())for name in ('baseline.json','current.json','diagnostic-32.json','diagnostic-96.json','full-run.json')]
assert all(r['compilerIdentity']==identity for r in (b,d32,d96,run))
assert a['compilerIdentity']['checkpoint']==identity['checkpoint']
assert a['compilerIdentity']['referenceBackend']==identity['referenceBackend']
assert [k for k in identity['sources']if a['compilerIdentity']['sources'][k]!=identity['sources'][k]]==['direct_sympy_arithmetic.py']
for label,r in (('baseline',a),('current',b)):
 assert r['complete'] and r['artifact']['compilerAliases']==0
 assert r['parity']['exhaustiveRegion'] and r['parity']['cases']==66049 and r['parity']['mismatches']==0
 assert r['artifact']['sha256']==r['parity']['sha256']==digest_file(root/(label+'.expr'))
assert a['inputDomains']==b['inputDomains']
assert a['artifact']['characters']==b['artifact']['characters']==326227
assert a['artifact']['paths']==b['artifact']['paths']==2
for d,paths in ((d32,3),(d96,11)):
 assert d['stop']=='Flat artifact budget exceeded; no complete result'
 assert d['stats']['completedPaths']==paths
 assert d['stats']['firstArm']=={'bodyCharacters':2061331,'guardCharacters':1554142,'decisions':6}
assert not run['complete'] and not (root/'full.expr').exists()
def counts(name):
 text=(root/name).read_text()
 return {k:int(re.search(r'ℹ '+k+r' (\d+)',text)[1])for k in ('tests','pass','fail','skipped')}
small,whole=counts('integration.log'),counts('project-test.log')
assert small=={'tests':34,'pass':34,'fail':0,'skipped':0}
failures=re.findall(r'^test at (\S+)',(root/'project-test.log').read_text(),re.M)
previous=json.loads((root.parent/'direct-sympy-intersected-grid/validation.json').read_text())
assert whole==previous['wholeProjectSuite']['current']
assert sorted(failures)==sorted(previous['wholeProjectSuite']['sameFailingTestLocations'])
proof=(root/'coherent-proof.log').read_text()
assert 'Ran 16 tests' in proof and 'OK' in proof
for expected in ('Flat checkpoint normalization: cases=888832 mismatches=0',
 'Prefix recipe dispatch parity: cases=30722 mismatches=0',
 'Intersected grid factor parity: cases=30722 mismatches=0',
 'Selected numeric propagation parity: cases=30722 mismatches=0'):assert expected in proof
factor=(root/'proof.log').read_text()
assert 'Ran 7 tests' in factor and 'OK' in factor
for expected in ('Exact producer factoring: cases=92166 mismatches=0','Linear signed-zero factoring: cases=122888 mismatches=0','Positive-zero multivariable factor: cases=552996 mismatches=0'):assert expected in factor
assert 'characters=2296124090440->2296124090440; finalArtifactEmitted=false' in factor
assert d32['stats']['firstArm']==previous['firstPathAfter']
assert all(d['stats']['lastArm']==prior['stats']['lastArm']for d,prior in zip((d32,d96),previous['fullDomainDiagnostics']))
v={'compilerIdentity':identity,'expressionRepresentation':'SymPy mathematical strings',
 'integrationTests':small,'coherentPathHelperTests':16,
 'wholeProjectSuite':{'current':whole,'sameFailingTestLocations':failures,'newFailingTestLocations':0},
 'nativeNormCases':888832,'nativePropagationCases':30722,'prefixDispatchCases':30722,'exactFactorHelperTests':7,'positiveZeroFactorNativeCases':552996,'intersectedFactorCases':30722,'bitMismatches':0,
 'regionalArtifact':b['artifact'],'inputDomains':b['inputDomains'],
 'regionalOutputCases':66049,'regionalOutputMismatches':0,'regionalOutputExhaustive':True,
 'fullDomainDiagnostics':[{'budgetMiB':d['artifactBudgetMiB'],'stats':d['stats'],'seconds':d['seconds']}for d in (d32,d96)],
 'prior96MiBCompletedPaths':11,'current96MiBCompletedPaths':11,
 'firstPathBefore':previous['fullDomainDiagnostics'][0]['stats']['firstArm'],
 'firstPathAfter':d32['stats']['firstArm'],
 'firstPathBodyReductionFraction':1-d32['stats']['firstArm']['bodyCharacters']/previous['fullDomainDiagnostics'][0]['stats']['firstArm']['bodyCharacters'],
 'fullDomainAttempt':{'seconds':run['seconds'],'workerSeconds':run['workerSeconds'],'workerPeakRSSBytes':run['workerPeakRSSBytes'],'stop':run['stop']},
 'regionalBenchmark':{'baselineCommit':'532c0cc','baselineSeconds':a['seconds'],'currentSeconds':b['seconds'],
 'baselinePeakRSSBytes':a['workerPeakRSSBytes'],'currentPeakRSSBytes':b['workerPeakRSSBytes'],
 'samples':1,'sameCompletedPaths':2,'speedupEstablished':False},
 'fullCoordinateArtifactEmitted':False,'fullCoordinateParity':False,'multipleTokenParity':False,
 'fullVectorParity':False,'historicalNumericalResultsReused':False,'CUDAUsed':False,'parallelSpeedupMeasured':False}
(root/'validation.json').write_text(json.dumps(v,indent=2)+'\n')
p=Path('docs/direct-string-validation.json');raw=p.read_text();oldmap=json.loads(raw);key='positiveZeroFactorValidation'
if key in oldmap:assert oldmap[key]==v
else:
 new=raw.rstrip()[:-1].rstrip()+',\n  '+json.dumps(key)+': '+json.dumps(v,indent=2).replace('\n','\n  ')+'\n}\n'
 assert {k:value for k,value in json.loads(new).items()if k!=key}==oldmap
 p.write_text(new)
print(json.dumps({'regionalCases':66049,'mismatches':0,'integration':small,'generalSuite':whole,'newFailures':0,'fullCoordinateParity':False,'completedPathsAt96MiB':11}))
