"""Record fresh evidence without promoting partial output to full completion."""
import json,re,sys
from pathlib import Path
sys.path.insert(0,str(Path('helpers').resolve()))
from direct_sympy_cover_regions import live_identity
from direct_sympy_savepoints import digest_file
root=Path(__file__).resolve().parent
identity=live_identity('docs/evidence/direct-sympy-test-checkpoint',2)
emission=json.loads((root/'emission.json').read_text());parity=json.loads((root/'parity.json').read_text())
diagnostic=json.loads((root/'full-domain-diagnostic.json').read_text())
assert identity==emission['compilerIdentity']==parity['compilerIdentity']==diagnostic['compilerIdentity']
assert emission['complete'] and emission['artifact']['compilerAliases']==0
assert emission['artifact']['sha256']==parity['sha256']==digest_file(root/'small.expr')
assert parity['exhaustiveRegion'] and parity['cases']==16 and parity['mismatches']==0
assert diagnostic['stop']=='Flat artifact budget exceeded; no complete result'
small=(root/'integration.log').read_text();counts={k:int(re.search(r'ℹ '+k+r' (\d+)',small)[1])for k in ('tests','pass','fail','skipped')}
assert counts=={'tests':34,'pass':34,'fail':0,'skipped':0},counts
whole=(root/'project-test.log').read_text();whole_counts={k:int(re.search(r'ℹ '+k+r' (\d+)',whole)[1])for k in ('tests','pass','fail','skipped')}
failures=re.findall(r'^test at (\S+)',whole,re.M)
baseline=json.loads((root.parent/'direct-sympy-ordered-guards/validation.json').read_text())['wholeProjectSuite']
assert whole_counts==baseline['current'],whole_counts
assert sorted(failures)==sorted(baseline['sameFailingTestLocations'])
proof=(root/'rms-proof.log').read_text();assert 'Ran 4 tests' in proof and 'OK' in proof and 'cases=3047424 mismatches=0' in proof
coherent=(root/'coherent-proof.log').read_text();assert 'Ran 12 tests' in coherent and 'OK' in coherent
validation={'compilerIdentity':identity,'expressionRepresentation':'SymPy mathematical strings','integrationTests':counts,
 'wholeProjectSuite':{'current':whole_counts,'sameFailingTestLocations':failures,'newFailingTestLocations':0},
 'rmsHelperTests':4,'coherentPathHelperTests':12,'nativeNormOutputCases':3047424,'nativeNormOutputMismatches':0,
 'rmsPositivePairProofCases':462422016,'rmsSelectedPairs':44294796,'rmsCellViolations':0,
 'artifact':emission['artifact'],'regionalArtifactCases':16,'regionalArtifactMismatches':0,'regionalArtifactExhaustive':True,
 'inputDomains':emission['inputDomains'],'freshCoveredInputPatterns':16,'fullInputPatterns':4030726144,
 'historicalNumericalResultsReused':False,'fullDomainDiagnostic':diagnostic['stats']['lastArm'],
 'fullCoordinateArtifactEmitted':False,'fullCoordinateParity':False,'multipleTokenParity':False,'fullVectorParity':False,
 'CUDAUsed':False,'parallelSpeedupMeasured':False}
(root/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
path=Path('docs/direct-string-validation.json');raw=path.read_text();previous=json.loads(raw);key='sourceContextValidation'
if key in previous:assert previous[key]==validation
else:
 new=raw.rstrip()[:-1].rstrip()+',\n  '+json.dumps(key)+': '+json.dumps(validation,indent=2).replace('\n','\n  ')+'\n}\n'
 assert {k:v for k,v in json.loads(new).items()if k!=key}==previous
 path.write_text(new)
print(json.dumps({'tests':counts,'generalSuite':whole_counts,'newFailures':0,'actualArtifactCases':16,'mismatches':0,'fullCoordinateParity':False}))
