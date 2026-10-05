"""Reconcile current actual artifacts, exhaustive parity and regression evidence."""
import json,re,sys
from pathlib import Path
sys.path.insert(0,str(Path('helpers').resolve()))
from direct_sympy_cover_regions import live_identity
from direct_sympy_savepoints import digest_file
root=Path(__file__).resolve().parent;identity=live_identity('docs/evidence/direct-sympy-test-checkpoint',2)
a=json.loads((root/'baseline.json').read_text());b=json.loads((root/'current.json').read_text());d=json.loads((root/'full-domain-diagnostic.json').read_text())
assert b['compilerIdentity']==d['compilerIdentity']==identity
assert a['compilerIdentity']['checkpoint']==identity['checkpoint'] and a['compilerIdentity']['referenceBackend']==identity['referenceBackend']
diff=[k for k in identity['sources'] if a['compilerIdentity']['sources'][k]!=identity['sources'][k]]
assert diff==['direct_sympy_coherent_paths.py'],diff
for label,r in (('baseline',a),('current',b)):
 assert r['complete'] and r['artifact']['compilerAliases']==0 and r['parity']['exhaustiveRegion']
 assert r['parity']['cases']==66049 and r['parity']['mismatches']==0
 assert r['artifact']['sha256']==r['parity']['sha256']==digest_file(root/(label+'.expr'))
assert a['inputDomains']==b['inputDomains'] and a['artifact']['paths']==b['artifact']['paths']==2
assert b['paths']['numericGuardImplications']==0
assert d['stop']=='Flat artifact budget exceeded; no complete result'
small=(root/'integration.log').read_text();counts={k:int(re.search(r'ℹ '+k+r' (\d+)',small)[1])for k in ('tests','pass','fail','skipped')}
assert counts=={'tests':34,'pass':34,'fail':0,'skipped':0},counts
whole=(root/'project-test.log').read_text();wc={k:int(re.search(r'ℹ '+k+r' (\d+)',whole)[1])for k in ('tests','pass','fail','skipped')}
failures=re.findall(r'^test at (\S+)',whole,re.M)
previous=json.loads((root.parent/'direct-sympy-fixed-scale-sqrt/validation.json').read_text())['wholeProjectSuite']
assert wc==previous['current'] and sorted(failures)==sorted(previous['sameFailingTestLocations'])
proof=(root/'proof.log').read_text();assert 'Ran 13 tests' in proof and 'OK' in proof
assert 'cases=888832 mismatches=0' in proof and 'cases=30722 mismatches=0' in proof
assert d['stats']['splitContexts']==6 and d['stats']['numericGuardImplications']==30
v={'compilerIdentity':identity,'expressionRepresentation':'SymPy mathematical strings','integrationTests':counts,
 'wholeProjectSuite':{'current':wc,'sameFailingTestLocations':failures,'newFailingTestLocations':0},
 'coherentPathHelperTests':13,'nativeNormCases':888832,'nativePropagationCases':30722,'nativeBitMismatches':0,
 'regionalArtifact':b['artifact'],'regionalOutputCases':66049,'regionalOutputMismatches':0,'regionalOutputExhaustive':True,
 'inputDomains':b['inputDomains'],'distributedDecisionsBefore':14,'distributedDecisionsAfter':6,'numericGuardImplications':30,
 'benchmark':{'baselineCommit':'d5dd566','baselineArtifact':a['artifact'],'baselineSeconds':a['seconds'],'currentSeconds':b['seconds'],
 'baselineWorkerSeconds':a['workerSeconds'],'currentWorkerSeconds':b['workerSeconds'],
 'baselinePeakRSSBytes':a['workerPeakRSSBytes'],'currentPeakRSSBytes':b['workerPeakRSSBytes'],'samples':1,'sameCompletedPaths':2},
 'fullDomainDiagnostic':d['stats']['lastArm'],'fullCoordinateArtifactEmitted':False,'fullCoordinateParity':False,
 'multipleTokenParity':False,'fullVectorParity':False,'historicalNumericalResultsReused':False,'CUDAUsed':False,'parallelSpeedupMeasured':False}
(root/'validation.json').write_text(json.dumps(v,indent=2)+'\n')
p=Path('docs/direct-string-validation.json');raw=p.read_text();old=json.loads(raw);key='prefixGuardBoundsValidation'
if key in old:assert old[key]==v
else:
 new=raw.rstrip()[:-1].rstrip()+',\n  '+json.dumps(key)+': '+json.dumps(v,indent=2).replace('\n','\n  ')+'\n}\n'
 assert {k:value for k,value in json.loads(new).items()if k!=key}==old
 p.write_text(new)
print(json.dumps({'actualArtifactCharacters':b['artifact']['characters'],'exhaustiveOutputCases':66049,'bitMismatches':0,'integration':counts,'generalSuite':wc,'newFailures':0,'fullCoordinateParity':False}))
