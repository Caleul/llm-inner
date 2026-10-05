"""Validate current sources, actual artifacts and unchanged failure locations."""
import json,re,sys
from pathlib import Path
sys.path.insert(0,str(Path('helpers').resolve()))
from direct_sympy_cover_regions import live_identity
from direct_sympy_savepoints import digest_file
root=Path(__file__).resolve().parent
load=lambda name:json.loads((root/(name+'.json')).read_text())
identity=live_identity('docs/evidence/direct-sympy-test-checkpoint',2)
previous=json.loads((root.parent/'direct-sympy-composed-recipes/validation.json').read_text())
old=previous['compilerIdentity']
assert identity['checkpoint']==old['checkpoint'] and identity['referenceBackend']==old['referenceBackend']
assert [name for name in identity['sources']if identity['sources'][name]!=old['sources'][name]]==['direct_sympy_coherent_paths.py']
regions=[]
for label,cases,characters in (('near-zero',16,23910),('current',66049,326227),('mixed',252,2913076)):
 r=load(label);assert r['compilerIdentity']==identity and r['complete']
 assert r['parity']['exhaustiveRegion'] and r['parity']['cases']==cases and r['parity']['mismatches']==0
 assert r['artifact']['characters']==characters and r['artifact']['compilerAliases']==0
 assert r['artifact']['sha256']==r['parity']['sha256']==digest_file(root/(label+'.expr'))
 regions.append({'inputDomains':r['inputDomains'],'artifact':r['artifact'],'parity':r['parity'],
  'seconds':r['seconds'],'workerPeakRSSBytes':r['workerPeakRSSBytes']})
diagnostics=[]
for mib,paths in ((32,5),(96,7)):
 d=load('diagnostic-'+str(mib))
 assert d['compilerIdentity']==identity and d['stats']['completedPaths']==paths and 'budget' in d['stop'].lower()
 assert d['stats']['firstArm']=={'bodyCharacters':6379,'guardCharacters':6932,'decisions':6}
 assert d['stats']['selectedRecipeIdentities']>=6 and d['stats']['frozenGuardImplications']==2
 assert not (root/('full-'+str(mib)+'.expr')).exists()
 diagnostics.append(d)
run=load('full-run');assert run['compilerIdentity']==identity and not run['complete']
assert not (root/'full.expr').exists()
def counts(name):
 text=(root/name).read_text()
 return {key:int(re.search(r'ℹ '+key+r' (\d+)',text)[1])for key in ('tests','pass','fail','skipped')}
small,whole=counts('integration-final.log'),counts('project-final.log')
failures=re.findall(r'^test at (\S+)',(root/'project-final.log').read_text(),re.M)
assert small=={'tests':34,'pass':34,'fail':0,'skipped':0}
assert whole==previous['wholeProjectSuite']['current']
assert sorted(failures)==sorted(previous['wholeProjectSuite']['sameFailingTestLocations'])
proof=(root/'proof.log').read_text();assert 'Ran 19 tests' in proof and '\nOK\n' in proof
for text in ('Frozen square identity parity: cases=30722 mismatches=0',
 'Flat checkpoint normalization: cases=888832 mismatches=0',
 'Selected numeric propagation parity: cases=30722 mismatches=0'):assert text in proof
v={'compilerIdentity':identity,'baselineCommit':'0ed82d3','expressionRepresentation':'SymPy mathematical strings',
 'regionalArtifacts':regions,'regionalCases':66317,'regionalBitMismatches':0,
 'fullDomainDiagnostics':diagnostics,'fullDomainAttempt':run,
 'coherentHelperTests':19,'frozenSquareNativeCases':30722,'nativeNormCases':888832,'nativePropagationCases':30722,
 'integrationTests':small,'wholeProjectSuite':{'current':whole,'sameFailingTestLocations':failures,'newFailingTestLocations':0},
 'fullCoordinateArtifactEmitted':False,'fullCoordinateParity':False,'multipleTokenParity':False,'fullVectorParity':False,
 'historicalNumericalResultsReused':False,'CUDAUsed':False,'parallelSpeedupMeasured':False}
(root/'validation.json').write_text(json.dumps(v,indent=2)+'\n')
p=Path('docs/direct-string-validation.json');raw=p.read_text();oldmap=json.loads(raw);key='selectedRecipeIdentityValidation'
if key in oldmap:assert oldmap[key]==v
else:
 result=raw.rstrip()[:-1].rstrip()+',\n  '+json.dumps(key)+': '+json.dumps(v,indent=2).replace('\n','\n  ')+'\n}\n'
 assert {k:value for k,value in json.loads(result).items()if k!=key}==oldmap
 p.write_text(result)
print(json.dumps({'regionalCases':66317,'mismatches':0,'integration':small,'generalSuite':whole,
 'recipeIdentities':diagnostics[-1]['stats']['selectedRecipeIdentities'],
 'frozenGuardImplications':diagnostics[-1]['stats']['frozenGuardImplications'],'fullCoordinateParity':False}))
