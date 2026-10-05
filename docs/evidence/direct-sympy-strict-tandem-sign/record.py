"""Bind strict-sign certificates to actual checkpoint artifacts and test history."""
import json,re,sys
from pathlib import Path
sys.path.insert(0,str(Path('helpers').resolve()))
from direct_sympy_cover_regions import live_identity
from direct_sympy_savepoints import digest_file
folder=Path(__file__).resolve().parent
load=lambda name:json.loads((folder/(name+'.json')).read_text())
identity=live_identity('docs/evidence/direct-sympy-test-checkpoint',2)
artifacts=[]
for label,size,cases in [('smoke',217033,66049),('small-smoke',120415,289)]:
 old,current=load(label+'-baseline'),load(label)
 assert current['compilerIdentity']==identity
 assert old['compilerIdentity']['checkpoint']==identity['checkpoint']
 assert old['compilerIdentity']['referenceBackend']==identity['referenceBackend']
 assert {n for n in identity['sources'] if old['compilerIdentity']['sources'][n]!=identity['sources'][n]}=={'direct_sympy_tandem.py'}
 assert old['inputDomains']==current['inputDomains']
 for name,item in [(label+'-baseline',old),(label,current)]:
  assert item['complete'] and item['artifact']['complete'] and item['artifact']['compilerAliases']==0
  assert item['artifact']['characters']==size and item['artifact']['paths']==2
  assert item['artifact']['sha256']==item['parity']['sha256']==digest_file(folder/(name+'.expr'))
  assert item['parity']['exhaustiveRegion'] and item['parity']['cases']==cases and item['parity']['mismatches']==0
  assert item['parity']['tokens']==1
 assert old['artifact']['sha256']==current['artifact']['sha256']
 artifacts.append({'label':label,'previousCharacters':size,'currentCharacters':size,'parityCases':cases,'bitMismatches':0,'byteIdentical':True,'inputDomains':current['inputDomains'],'artifact':current['artifact']})
proof=(folder/'strict-sign-tests.log').read_text()
assert 'Ran 1 test' in proof and '\nOK\n' in proof
assert 'Strict-sign tandem parity: cases=507888 mismatches=0' in proof
assert 'Positive-factor sign parity: cases=507904 mismatches=0' in proof
old,current=load('baseline-arms'),load('current-arms')
assert old['candidates']==current['candidates'] and len(current['candidates'])==27
assert current['tandemSourceSHA256']==identity['sources']['direct_sympy_tandem.py']
assert old['tandemSourceSHA256']==load('smoke-baseline')['compilerIdentity']['sources']['direct_sympy_tandem.py']
full=load('full-run');coordinate=full['coordinate'];assert coordinate['compilerIdentity']==identity
assert not coordinate['complete'] and not coordinate['artifactPublished'] and not coordinate['parityVerified']
assert coordinate['producersCompleted']==25 and coordinate['stats']['completedPaths']==28
assert 'artifact budget' in coordinate['stop'].lower() and full['expressionAndConditionBudgetBytes']==536870912
assert full['peakObservedRSSBytes']<=full['measuredRAMBudgetBytes']
ledger=load('artifact-budget');assert ledger['claims']=={'vector-syntax':15}
assert ledger['limit']==536870912 and ledger['peakClaimedCharacters']<=ledger['limit']
def counts(name):
 text=(folder/name).read_text();return {k:int(re.search(r'ℹ '+k+r' (\d+)',text)[1])for k in ('tests','pass','fail','skipped')}
focused,affected,portable=map(counts,('integration-tests.log','affected-integration-tests.log','portable-tests.log'))
assert focused=={'tests':40,'pass':40,'fail':0,'skipped':0}
assert affected=={'tests':1,'pass':1,'fail':0,'skipped':0}
assert portable=={'tests':620,'pass':571,'fail':0,'skipped':49}
value={'baselineCommit':'85aea9e0358b7f38e2e3f33e9ec53a3d597ba519','compilerIdentity':identity,
 'strictSignNativeCases':507888,'mixedSignNativeCases':507904,'nativeBitMismatches':0,
 'strictEnclosureRequired':True,'zeroInclusiveIntervalsRetainOriginalSign':True,'noNewRuntimeBranches':True,'numericalOrderChanged':False,
 'regionalArtifacts':artifacts,'unchangedOrderedCandidates':27,'checkpointSizeReductionDemonstrated':False,
 'focusedIntegrations':focused,'affectedIntegrationWithStrictSignAssertion':affected,'portableSuite':portable,'wholeDomainAttempt':full,
 'fullCoordinateArtifactEmitted':False,'fullCoordinateParity':False,'multipleTokenParity':False,'fullVectorParity':False,'historicalNumericalStatesReused':False}
(folder/'validation.json').write_text(json.dumps(value,indent=2)+'\n')
for file in ('docs/test-suite-map.json','docs/direct-string-validation.json'):
 p=Path(file);raw=p.read_text();previous=json.loads(raw);key='validatedStrictTandemSign'
 if key in previous:assert previous[key]==value
 else:
  text=raw.rstrip()[:-1].rstrip()+',\n  '+json.dumps(key)+': '+json.dumps(value,indent=2).replace('\n','\n  ')+'\n}\n'
  assert {k:v for k,v in json.loads(text).items()if k!=key}==previous;p.write_text(text)
print(json.dumps({'focused':focused,'affected':affected,'portable':portable,'strictSignCases':507888,'checkpointSizeReductionDemonstrated':False,'fullCoordinateParity':False}))
