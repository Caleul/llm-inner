"""Reconcile exact certificates, real smoke artifacts and unchanged test history."""
import json,re,sys
from pathlib import Path
sys.path.insert(0,str(Path('helpers').resolve()))
from direct_sympy_cover_regions import live_identity
from direct_sympy_savepoints import digest_file
folder=Path(__file__).resolve().parent
load=lambda name:json.loads((folder/(name+'.json')).read_text())
identity=live_identity('docs/evidence/direct-sympy-test-checkpoint',2)
old,current=load('smoke-baseline'),load('smoke')
assert current['compilerIdentity']==identity
assert old['compilerIdentity']['checkpoint']==identity['checkpoint']
assert old['compilerIdentity']['referenceBackend']==identity['referenceBackend']
assert {n for n in identity['sources']if old['compilerIdentity']['sources'][n]!=identity['sources'][n]}=={'direct_sympy_sqrt.py'}
assert old['inputDomains']==current['inputDomains']
for label,item,size in [('smoke-baseline',old,229153),('smoke',current,217033)]:
 assert item['complete']and item['artifact']['complete']and item['artifact']['compilerAliases']==0
 assert item['artifact']['characters']==size and item['artifact']['paths']==2
 assert item['artifact']['sha256']==item['parity']['sha256']==digest_file(folder/(label+'.expr'))
 assert item['parity']['exhaustiveRegion']and item['parity']['cases']==66049 and item['parity']['mismatches']==0
 assert item['parity']['tokens']==1
proof=(folder/'sqrt-tests.log').read_text();assert 'Ran 4 tests'in proof and '\nOK\n'in proof
for message in ('cases=67895912 mismatches=0','cases=19398662 mismatches=0 midpoints=0','cases=25167601 mismatches=0','cases=58720257 mismatches=0','cases=1835032 mismatches=0','cases=524290 mismatches=0 midpoints=0'):assert message in proof
a,b=load('baseline-arms'),load('current-arms');assert len(a['candidates'])==len(b['candidates'])==27
assert b['rootSourceSHA256']==identity['sources']['direct_sympy_sqrt.py']
assert a['rootSourceSHA256']==old['compilerIdentity']['sources']['direct_sympy_sqrt.py']
comparisons=[]
for x,y in zip(a['candidates'],b['candidates']):
 assert x['candidate']==y['candidate']and x['guardSkeletonSHA256']==y['guardSkeletonSHA256']
 assert y['bodyCharacters']<=x['bodyCharacters']and y['guardCharacters']<=x['guardCharacters']
 comparisons.append({'candidate':x['candidate'],'previousBodyCharacters':x['bodyCharacters'],'currentBodyCharacters':y['bodyCharacters'],'previousGuardCharacters':x['guardCharacters'],'currentGuardCharacters':y['guardCharacters'],'sameOrderedGuardSkeleton':True})
full=load('full-run');coordinate=full['coordinate'];assert coordinate['compilerIdentity']==identity
assert not coordinate['complete']and not coordinate['artifactPublished']and not coordinate['parityVerified']
assert coordinate['producersCompleted']==25 and coordinate['stats']['completedPaths']==28
assert 'artifact budget'in coordinate['stop'].lower()and full['expressionAndConditionBudgetBytes']==536870912
assert full['peakObservedRSSBytes']<=full['measuredRAMBudgetBytes']
def counts(name):
 text=(folder/name).read_text();return {k:int(re.search(r'ℹ '+k+r' (\d+)',text)[1])for k in ('tests','pass','fail','skipped')}
focused,portable=counts('integration-tests.log'),counts('portable-tests.log')
assert focused=={'tests':40,'pass':39,'fail':1,'skipped':0}
affected=counts('affected-integration-tests.log')
assert affected=={'tests':2,'pass':2,'fail':0,'skipped':0}
initial=(folder/'integration-tests.log').read_text()
assert '61355' in initial and '60187' in initial and 'fallbackCharacters=8720' in initial
assert initial.count('✖ Projection sign simplification')==2
ledger=load('artifact-budget')
assert ledger['claims']=={'vector-syntax':15}
assert ledger['limit']==536870912 and ledger['peakClaimedCharacters']<=ledger['limit']
assert portable=={'tests':620,'pass':571,'fail':0,'skipped':49}
value={'baselineCommit':'9e6279f64b234ba79c1e7e64afb14ef1da9725ab','compilerIdentity':identity,
 'regionalRootCertificateCases':19398662,'regionalRootEmittedScaleCases':67895912,'rootBitMismatches':0,'rootMidpoints':0,
 'noNewRuntimeKernelBranch':True,'unknownScaleExponentWordAdjustmentRetained':True,'normalizedRootGridCertified':True,'fixedScaleExponentAdjustmentIsExactMultiplication':True,'numericalOrderChanged':False,
 'regionalCoordinatePreviousCharacters':229153,'regionalCoordinateCurrentCharacters':217033,
 'regionalCoordinateParityCases':66049,'regionalCoordinateBitMismatches':0,'regionalArtifact':current['artifact'],
 'regionalInputDomains':current['inputDomains'],'sameOrderedBranchSkeletons':27,'candidateExpansionComparisons':comparisons,
 'initialFocusedIntegrations':focused,'affectedIntegrationsAfterSizeAssertionUpdate':affected,'initialFailureWasStaleSizeAssertion':True,'portableSuite':portable,'wholeDomainAttempt':full,
 'fullCoordinateArtifactEmitted':False,'fullCoordinateParity':False,'multipleTokenParity':False,'fullVectorParity':False,
 'historicalNumericalStatesReused':False}
(folder/'validation.json').write_text(json.dumps(value,indent=2)+'\n')
for file in ('docs/test-suite-map.json','docs/direct-string-validation.json'):
 p=Path(file);raw=p.read_text();previous=json.loads(raw);key='validatedRootGridScale'
 if key in previous:assert previous[key]==value
 else:
  text=raw.rstrip()[:-1].rstrip()+',\n  '+json.dumps(key)+': '+json.dumps(value,indent=2).replace('\n','\n  ')+'\n}\n'
  assert {k:v for k,v in json.loads(text).items()if k!=key}==previous;p.write_text(text)
print(json.dumps({'initialFocused':focused,'affected':affected,'portable':portable,'regionalArtifactCharacters':217033,'regionalParityCases':66049,'fullCoordinateParity':False}))
