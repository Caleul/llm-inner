"""Reconcile fresh artifacts, source compatibility and project test locations."""
import json,re,sys
from pathlib import Path
sys.path.insert(0,str(Path('helpers').resolve()))
from direct_sympy_cover_regions import live_identity
from direct_sympy_savepoints import digest_file
root=Path(__file__).resolve().parent
load=lambda name:json.loads((root/(name+'.json')).read_text())
identity=live_identity('docs/evidence/direct-sympy-test-checkpoint',2)
previous=json.loads((root.parent/'direct-sympy-prefix-guard-investigation/validation.json').read_text())
old=previous['compilerIdentity']
assert identity['checkpoint']==old['checkpoint'] and identity['referenceBackend']==old['referenceBackend']
assert set(identity['sources'])-set(old['sources'])=={'direct_sympy_recipe_bounds.py'}
assert {name for name in old['sources']if identity['sources'][name]!=old['sources'][name]}=={'direct_sympy_coherent_paths.py','direct_sympy_savepoints.py'}
regions=[]
for label,cases in (('near-zero',16),('current',66049),('mixed',252)):
 r=load(label);assert r['compilerIdentity']==identity and r['complete']
 assert r['parity']['exhaustiveRegion'] and r['parity']['cases']==cases and r['parity']['mismatches']==0
 assert r['artifact']['compilerAliases']==0 and r['artifact']['complete']
 assert r['artifact']['characters']==len((root/(label+'.expr')).read_text())
 assert r['artifact']['sha256']==r['parity']['sha256']==digest_file(root/(label+'.expr'))
 regions.append({'inputDomains':r['inputDomains'],'artifact':r['artifact'],'parity':r['parity'],
  'seconds':r['seconds'],'workerPeakRSSBytes':r['workerPeakRSSBytes']})
diagnostics=[]
for mib,paths in ((32,6),(96,8)):
 d=load('diagnostic-'+str(mib))
 assert d['compilerIdentity']==identity and d['stats']['completedPaths']==paths and 'budget' in d['stop'].lower()
 assert d['stats']['firstArm']=={'bodyCharacters':6379,'guardCharacters':6932,'decisions':6}
 assert d['stats']['recipeConstraintRefinements']>0 and d['stats']['backwardRoundingSteps']>0
 assert not (root/('full-'+str(mib)+'.expr')).exists()
 diagnostics.append(d)
run=load('full-run');assert run['compilerIdentity']==identity and not run['complete']
assert not (root/'full.expr').exists()
def counts(name):
 text=(root/name).read_text()
 return {key:int(re.search(r'ℹ '+key+r' (\d+)',text)[1])for key in ('tests','pass','fail','skipped')}
small,whole=counts('focused-tests.log'),counts('general-tests.log')
failures=re.findall(r'^test at (\S+)',(root/'general-tests.log').read_text(),re.M)
assert small=={'tests':35,'pass':35,'fail':0,'skipped':0},small
expected=dict(previous['wholeProjectSuite']['current']);expected['tests']+=1;expected['skipped']+=1
assert whole==expected,whole
assert sorted(failures)==sorted(previous['wholeProjectSuite']['sameFailingTestLocations'])
proof=(root/'coherent-proof.log').read_text();assert 'Ran 19 tests' in proof and '\nOK\n' in proof
for text in ('Frozen square identity parity: cases=30722 mismatches=0',
 'Flat checkpoint normalization: cases=888832 mismatches=0',
 'Selected numeric propagation parity: cases=30722 mismatches=0'):assert text in proof
bounds=(root/'bounds-proof.log').read_text();assert 'Ran 4 tests' in bounds and '\nOK\n' in bounds
for text in ('Backward residual certificate: cases=10242 selected=4 violations=0',
 'Backward varying inverse certificate: cases=10242 selected=132 violations=0'):assert text in bounds
v={'compilerIdentity':identity,'baselineCommit':'ca4c586','expressionRepresentation':'SymPy mathematical strings',
 'regionalArtifacts':regions,'regionalCases':66317,'regionalBitMismatches':0,
 'fullDomainDiagnostics':diagnostics,'fullDomainAttempt':run,
 'coherentHelperTests':19,'recipeConstraintTests':4,
 'nativeResidualBoundCases':10242,'nativeResidualBoundSelected':4,
 'nativeVaryingInverseBoundCases':10242,'nativeVaryingInverseBoundSelected':132,
 'integrationTests':small,'wholeProjectSuite':{'current':whole,'sameFailingTestLocations':failures,'newFailingTestLocations':0},
 'oldNumericalSavepointsCompatible':False,
 'fullCoordinateArtifactEmitted':False,'fullCoordinateParity':False,'multipleTokenParity':False,'fullVectorParity':False,
 'historicalNumericalResultsReused':False,'CUDAUsed':False,'parallelSpeedupMeasured':False}
(root/'validation.json').write_text(json.dumps(v,indent=2)+'\n')
p=Path('docs/direct-string-validation.json');raw=p.read_text();oldmap=json.loads(raw);key='bidirectionalRecipeBoundsValidation'
if key in oldmap:assert oldmap[key]==v
else:
 result=raw.rstrip()[:-1].rstrip()+',\n  '+json.dumps(key)+': '+json.dumps(v,indent=2).replace('\n','\n  ')+'\n}\n'
 assert {k:value for k,value in json.loads(result).items()if k!=key}==oldmap
 p.write_text(result)
print(json.dumps({'regionalCases':66317,'mismatches':0,'integration':small,'generalSuite':whole,'fullCoordinateParity':False}))
