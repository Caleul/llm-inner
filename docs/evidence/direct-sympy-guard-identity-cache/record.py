"""Bind an exact, bounded compiler-only identity cache to real runtime evidence."""
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
assert {n for n in identity['sources'] if old['compilerIdentity']['sources'][n]!=identity['sources'][n]}=={'direct_sympy_coherent_paths.py'}
assert old['inputDomains']==current['inputDomains']
for label,item in [('smoke-baseline',old),('smoke',current)]:
 assert item['complete'] and item['artifact']['complete'] and item['artifact']['compilerAliases']==0
 assert item['artifact']['characters']==217033 and item['artifact']['paths']==2
 assert item['artifact']['sha256']==item['parity']['sha256']==digest_file(folder/(label+'.expr'))
 assert item['parity']['exhaustiveRegion'] and item['parity']['cases']==66049 and item['parity']['mismatches']==0
 assert item['parity']['tokens']==1
assert old['artifact']['sha256']==current['artifact']['sha256']
proof=(folder/'coherent-tests.log').read_text()
assert 'Ran 22 tests' in proof and 'OK (skipped=1)' in proof
for name in ('Frozen square identity parity','Intersected grid factor parity','Prefix recipe dispatch parity','Selected numeric propagation parity'):
 assert name+': cases=30722 mismatches=0' in proof
baseline,measured=load('baseline-arms'),load('current-arms')
assert baseline['candidates']==measured['candidates'] and len(measured['candidates'])==27
assert measured['pathsSourceSHA256']==identity['sources']['direct_sympy_coherent_paths.py']
assert baseline['pathsSourceSHA256']==old['compilerIdentity']['sources']['direct_sympy_coherent_paths.py']
assert measured['stats']['guardIdentityCacheHits']==220
assert measured['cacheResidentCharacterCharge']>0 and measured['cacheCharacterBudget']==8388608
assert measured['cacheResidentCharacterCharge']<=measured['cacheCharacterBudget']
timing=measured['dispatchTimings'];assert 0<timing['cachedSeconds']<timing['uncachedSeconds']
full=load('full-run');coordinate=full['coordinate'];assert coordinate['compilerIdentity']==identity
assert not coordinate['complete'] and not coordinate['artifactPublished'] and not coordinate['parityVerified']
assert coordinate['producersCompleted']==25 and coordinate['stats']['completedPaths']==28
assert coordinate['stats']['guardIdentityCacheHits']==232
assert 'artifact budget' in coordinate['stop'].lower() and full['expressionAndConditionBudgetBytes']==536870912
assert full['peakObservedRSSBytes']<=full['measuredRAMBudgetBytes']
ledger=load('artifact-budget');assert ledger['claims']=={'vector-syntax':15}
assert ledger['limit']==536870912 and ledger['peakClaimedCharacters']<=ledger['limit']
def counts(name):
 text=(folder/name).read_text();return {k:int(re.search(r'ℹ '+k+r' (\d+)',text)[1])for k in ('tests','pass','fail','skipped')}
focused,portable=map(counts,('integration-tests.log','portable-tests.log'))
assert focused=={'tests':3,'pass':3,'fail':0,'skipped':0}
assert portable=={'tests':620,'pass':571,'fail':0,'skipped':49}
value={'baselineCommit':'6a907f703879ed6d7315a7fddc9ec38d26775ac5','compilerIdentity':identity,
 'immutableLiteralSnapshotProof':True,'digestCollisionFallbackRetained':True,'reassignedViewInvalidatesCache':True,'cacheLRUBudgetCharacters':8388608,
 'cacheMeasuredCharacterCharge':measured['cacheResidentCharacterCharge'],'cacheHitsIn27Candidates':220,'samePrunedGuardsForCacheEnabledAndDisabled':27,
 'dispatchTimings':timing,'dispatchOperationSpeedup':timing['uncachedSeconds']/timing['cachedSeconds'],
 'wholeCompilerSpeedupProven':False,'regionalCoordinateByteIdentical':True,'regionalCoordinateCharacters':217033,'regionalCoordinateParityCases':66049,'regionalCoordinateBitMismatches':0,
 'regionalInputDomains':current['inputDomains'],'regionalArtifact':current['artifact'],'checkpointSizeReductionDemonstrated':False,
 'directCoherentTests':{'tests':22,'pass':21,'fail':0,'skipped':1,'skipReason':'checkpoint not configured for direct standalone call'},
 'focusedIntegrationsWithCheckpoint':focused,'portableSuite':portable,'wholeDomainAttempt':full,
 'fullCoordinateArtifactEmitted':False,'fullCoordinateParity':False,'multipleTokenParity':False,'fullVectorParity':False,'historicalNumericalStatesReused':False}
(folder/'validation.json').write_text(json.dumps(value,indent=2)+'\n')
for file in ('docs/test-suite-map.json','docs/direct-string-validation.json'):
 p=Path(file);raw=p.read_text();previous=json.loads(raw);key='validatedGuardIdentityCache'
 if key in previous:assert previous[key]==value
 else:
  text=raw.rstrip()[:-1].rstrip()+',\n  '+json.dumps(key)+': '+json.dumps(value,indent=2).replace('\n','\n  ')+'\n}\n'
  assert {k:v for k,v in json.loads(text).items()if k!=key}==previous;p.write_text(text)
print(json.dumps({'focused':focused,'portable':portable,'dispatchOperationSpeedup':value['dispatchOperationSpeedup'],'fullCoordinateParity':False}))
