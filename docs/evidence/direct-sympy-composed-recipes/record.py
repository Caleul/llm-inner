"""Reconcile live source identity, actual file parity and existing regressions."""
import json,re,sys
from pathlib import Path
sys.path.insert(0,str(Path('helpers').resolve()))
from direct_sympy_cover_regions import live_identity
from direct_sympy_savepoints import digest_file
root=Path(__file__).resolve().parent
load=lambda name:json.loads((root/(name+'.json')).read_text())
identity=live_identity('docs/evidence/direct-sympy-test-checkpoint',2)
before,after=load('recipe-before'),load('recipe-after')
assert after['compilerIdentity']==identity
assert before['compilerIdentity']['checkpoint']==identity['checkpoint']
assert before['compilerIdentity']['referenceBackend']==identity['referenceBackend']
assert [name for name in identity['sources']if identity['sources'][name]!=before['compilerIdentity']['sources'][name]]==['direct_sympy_streaming_literals.py']
old_recipe=next(iter(before['recipes'].values()));new_recipe=next(iter(after['recipes'].values()))
assert len(before['recipes'])==len(after['recipes'])==1 and len(old_recipe)==636
assert new_recipe=='R16(R32(CompileValue24() * CompileValue25()))'
assert before['firstArm']=={'bodyCharacters':212175,'guardCharacters':321876,'decisions':6}
assert after['firstArm']=={'bodyCharacters':6379,'guardCharacters':6944,'decisions':6}
regions=[]
for label,cases,characters in (('near-zero',16,23910),('current',66049,326227)):
 r=load(label)
 assert r['compilerIdentity']==identity and r['complete']
 assert r['parity']['exhaustiveRegion'] and r['parity']['cases']==cases and r['parity']['mismatches']==0
 assert r['artifact']['characters']==characters and r['artifact']['compilerAliases']==0
 assert r['artifact']['sha256']==r['parity']['sha256']==digest_file(root/(label+'.expr'))
 regions.append({'inputDomains':r['inputDomains'],'artifact':r['artifact'],'parity':r['parity'],
  'seconds':r['seconds'],'workerPeakRSSBytes':r['workerPeakRSSBytes']})
diagnostics=[]
for mib,paths in ((32,6),(96,8)):
 d=load('diagnostic-'+str(mib))
 assert d['compilerIdentity']==identity and d['stats']['completedPaths']==paths
 assert d['stats']['firstArm']==after['firstArm'] and 'budget' in d['stop'].lower()
 assert not (root/('full-'+str(mib)+'.expr')).exists()
 diagnostics.append(d)
run=load('full-run');assert run['compilerIdentity']==identity and not run['complete']
assert not (root/'full.expr').exists()
def counts(name):
 text=(root/name).read_text()
 return {key:int(re.search(r'ℹ '+key+r' (\d+)',text)[1])for key in ('tests','pass','fail','skipped')}
small,whole=counts('integration.log'),counts('project-test.log')
previous=json.loads((root.parent/'direct-sympy-selected-scale-precision/validation.json').read_text())
failures=re.findall(r'^test at (\S+)',(root/'project-test.log').read_text(),re.M)
assert small=={'tests':34,'pass':34,'fail':0,'skipped':0}
assert whole==previous['wholeProjectSuite']['current']
assert sorted(failures)==sorted(previous['wholeProjectSuite']['sameFailingTestLocations'])
proof=(root/'streaming-proof.log').read_text();assert 'Ran 8 tests' in proof and '\nOK\n' in proof
for text in ('Emitted streaming literal: cases=30722 mismatches=0',
 'Emitted checkpoint inverse: cases=888832 mismatches=0',
 'Complete compiler composition: cases=60 mismatches=0;'):assert text in proof
v={'compilerIdentity':identity,'baselineCommit':'9d227df','expressionRepresentation':'SymPy mathematical strings',
 'recipeBeforeCharacters':len(old_recipe),'recipeAfterCharacters':len(new_recipe),
 'firstArmBefore':before['firstArm'],'firstArmAfter':after['firstArm'],
 'regionalArtifacts':regions,'fullDomainDiagnostics':diagnostics,'fullDomainAttempt':run,
 'streamingHelperTests':8,'nativeLiteralCases':30722,'nativeInverseCases':888832,'compositionCases':60,
 'integrationTests':small,'wholeProjectSuite':{'current':whole,'sameFailingTestLocations':failures,'newFailingTestLocations':0},
 'fullCoordinateArtifactEmitted':False,'fullCoordinateParity':False,'multipleTokenParity':False,'fullVectorParity':False,
 'historicalNumericalResultsReused':False,'CUDAUsed':False,'parallelSpeedupMeasured':False}
(root/'validation.json').write_text(json.dumps(v,indent=2)+'\n')
p=Path('docs/direct-string-validation.json');raw=p.read_text();old=json.loads(raw);key='composedNumericRecipesValidation'
if key in old:assert old[key]==v
else:
 result=raw.rstrip()[:-1].rstrip()+',\n  '+json.dumps(key)+': '+json.dumps(v,indent=2).replace('\n','\n  ')+'\n}\n'
 assert {k:value for k,value in json.loads(result).items()if k!=key}==old
 p.write_text(result)
print(json.dumps({'recipeCharacters':len(new_recipe),'firstArm':after['firstArm'],
 'integration':small,'generalSuite':whole,'regionalCases':66065,'mismatches':0,'fullCoordinateParity':False}))
