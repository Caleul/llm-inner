"""Verify actual emitted expressions, source identity and numerical evidence."""
import json,re,sys
from pathlib import Path
sys.path.insert(0,str(Path('helpers').resolve()))
from direct_sympy_cover_regions import live_identity
from direct_sympy_savepoints import digest_file
root=Path(__file__).resolve().parent
load=lambda label:json.loads((root/(label+'.json')).read_text())
identity=live_identity('docs/evidence/direct-sympy-test-checkpoint',2)
baseline=json.loads((root.parent/'direct-sympy-narrow-root/validation.json').read_text())
old=baseline['compilerIdentity']
assert identity['checkpoint']==old['checkpoint'] and identity['referenceBackend']==old['referenceBackend']
changed={name for name in old['sources']if identity['sources'][name]!=old['sources'][name]}
assert changed=={'direct_sympy_coherent_paths.py','direct_sympy_projection_constraints.py'},changed
regions=[]
for label,cases in (('near-zero',16),('current',66049),('mixed',252)):
 r=load(label);assert r['compilerIdentity']==identity and r['complete']
 a=r['artifact'];assert a['compilerAliases']==0 and a['complete']
 assert a['characters']==len((root/(label+'.expr')).read_text())
 assert a['sha256']==r['parity']['sha256']==digest_file(root/(label+'.expr'))
 assert r['parity']['exhaustiveRegion'] and r['parity']['cases']==cases and not r['parity']['mismatches']
 previous=json.loads((root.parent/'direct-sympy-narrow-root'/(label+'.json')).read_text())
 assert previous['inputDomains']==r['inputDomains'] and previous['artifact']['sha256']==a['sha256']
 regions.append(r)
selected=load('selected');assert selected['compilerIdentity']==identity
assert selected['nativeCases']==135184 and selected['nativeAcceptedCases']==117379 and selected['nativeBitMismatches']==0
assert selected['admittedBodyCharacters']==107 and selected['nonAdmittedBodyCharacters']==8836
assert selected['artifact']['sha256']==digest_file(root/'selected.expr')
assert selected['artifact']['characters']==len((root/'selected.expr').read_text())==62059
assert not re.search(r'\b(?:CompileValue|CASNumericRegion)\d+\(', (root/'selected.expr').read_text())
assert selected['stats']['selectedSignMaskEliminations']>0
diagnostics=[]
for mib,candidates in ((32,15),(96,23)):
 d=load('diagnostic-'+str(mib));assert d['compilerIdentity']==identity
 assert d['stats']['completedPaths']==candidates and 'budget'in d['stop'].lower()
 assert d['stats']['selectedSignMaskEliminations']>0
 assert not (root/(f'full-{mib}.expr')).exists();diagnostics.append(d)
run=load('full-run');assert run['compilerIdentity']==identity and not run['complete']
assert 'budget'in run['stop'].lower() and not (root/'full.expr').exists()
capture=load('candidate');assert capture['compilerIdentity']==identity
assert capture['candidate']==24 and capture['bodyCharacters']==168126471
proof=(root/'sign-proof.log').read_text();assert 'Ran 2 tests'in proof and '\nOK\n'in proof
assert 'cases=37838852 selected=36904348 mismatches=0'in proof
assert 'cases=135184 accepted=117379 mismatches=0'in proof
coherent=(root/'coherent-tests.log').read_text();assert 'Ran 22 tests'in coherent and '\nOK\n'in coherent
def counts(label):
 text=(root/(label+'.log')).read_text()
 return {k:int(re.search(r'ℹ '+k+r' (\d+)',text)[1])for k in ('tests','pass','fail','skipped')}
focused=counts('focused-tests');assert focused=={'tests':39,'pass':39,'fail':0,'skipped':0},focused
last=counts('projection-integration');assert last['pass']==1 and not last['fail'],last
general=counts('general-tests');assert general=={'tests':619,'pass':571,'fail':0,'skipped':48},general
suite=json.loads(Path('docs/test-suite-map.json').read_text())
assert suite['validation']['default']['tests']==618 and suite['validation']['default']['fail']==0
v={'compilerIdentity':identity,'compilerBaselineCommit':'2eece04a3f03ffce8024506e0e4d71792a5580e4',
 'integratedConcurrentCommit':'5ecf74e0e69dc5460ee42446179970e4c64abeb4',
 'expressionRepresentation':'SymPy mathematical strings','selectedCheckpointFragment':selected,
 'nativeProjectionProofCases':37838852,'nativeProjectionSelectedCases':36904348,'nativeProjectionSignMismatches':0,
 'regionalArtifacts':regions,'regionalCases':66317,'regionalBitMismatches':0,'regionalFilesUnchanged':True,
 'fullDomainDiagnostics':diagnostics,'fullDomainAttempt':run,
 'oversizedCandidate':{'candidate':24,'bodyCharacters':168126471,'unchangedFromCompilerBaseline':True},
 'budget96MiBCandidatesGenerated':{'baseline':22,'current':23,'includesRejectedCandidate':True,'matchingPathComparison':False},
 'integrationTests':focused,'wholeProjectSuite':{'current':general,'baselineMap':'docs/test-suite-map.json','newFailingTestLocations':0},
 'projectionProofTestSHA256':digest_file(Path('helpers/direct_sympy_projection_sign_test.py')),
 'oldNumericalSavepointsCompatible':False,'historicalNumericalResultsReused':False,
 'fullCoordinateArtifactEmitted':False,'fullCoordinateParity':False,'multipleTokenParity':False,'fullVectorParity':False,
 'CUDAUsed':False,'parallelSpeedupMeasured':False}
(root/'validation.json').write_text(json.dumps(v,indent=2)+'\n')
for file,key,value in (('docs/direct-string-validation.json','projectionSignSimplificationValidation',v),
 ('docs/test-suite-map.json','directProjectionSignCompilerValidation',{'source':'test/direct-sympy-strings.test.ts',
 'helper':'helpers/direct_sympy_projection_sign_test.py','evidence':'docs/evidence/direct-sympy-projection-sign/validation.json',
 'focused':focused,'default':general,'numericalGuarantee':'Strict branch-local error bound; original magnitude and ordered rounding retained',
 'fullCoordinateParity':False,'multipleTokenParity':False,'fullVectorParity':False})):
 p=Path(file);raw=p.read_text();before=json.loads(raw)
 if key in before:assert before[key]==value
 else:
  result=raw.rstrip()[:-1].rstrip()+',\n  '+json.dumps(key)+': '+json.dumps(value,indent=2).replace('\n','\n  ')+'\n}\n'
  assert {k:x for k,x in json.loads(result).items()if k!=key}==before
  p.write_text(result)
print(json.dumps({'selectedFragmentCharacters':62059,'nativeCases':37838852+135184,'mismatches':0,'focused':focused,'general':general,'fullCoordinateParity':False}))
