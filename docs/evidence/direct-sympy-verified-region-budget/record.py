"""Record actual admitted artifacts, never mocked test coverage, as progress."""
import json
import re
import shutil
import sys
from pathlib import Path
sys.path.insert(0,str(Path('helpers').resolve()))
from direct_sympy_cover_regions import live_identity
from direct_sympy_partition_run import audit_tree,region_character_charge
from direct_sympy_savepoints import digest_file

folder=Path(__file__).resolve().parent
source=Path(sys.argv[1])
state=json.loads((source/'frontier.json').read_text())
assert state['identity']==live_identity('docs/evidence/direct-sympy-test-checkpoint',2)
assert state['regionVerificationPolicy']=={'enabled':True,'randomCases':8192}
covered,remaining=audit_tree(state['tree'],state['root'])
assert covered+remaining==4030726144
charge=10;cases=0;rows=[]
for key,node in sorted(state['tree'].items()):
    if node['status']!='complete':continue
    artifact=node['artifact'];proof=node['parity'];path=source/artifact['file']
    assert digest_file(path)==artifact['sha256']==proof['sha256']
    assert proof['mismatches']==0 and proof['cases']>=8192 and proof['tokens']==1
    cost=region_character_charge(source,node)
    assert cost==artifact['combinedCharacterCharge']
    charge+=cost;cases+=proof['cases']
    shutil.copyfile(path,folder/path.name)
    rows.append({'region':key,'domains':node['domains'],'artifact':artifact,'parity':proof})
assert charge==state['accumulatedArtifactCharacters'] and charge<=536870912
assert remaining>0 and not state['finalArtifactEmitted'] and not state['finalParity']
assert not (source/'coordinate.expr').exists()
def counts(name):
    text=(folder/name).read_text()
    return {key:int(re.search(r'ℹ '+key+r' (\d+)',text)[1])for key in ('tests','pass','fail','skipped')}
focused=counts('integration-tests.log');portable=counts('portable-tests.log')
assert focused=={'tests':4,'pass':4,'fail':0,'skipped':0}
assert portable['fail']==0 and portable['pass']>=571
assert 'Ran 9 tests' in (folder/'partition-tests.log').read_text()
assert 'cases=576 mismatches=0' in (folder/'partition-tests.log').read_text()
value={'compilerIdentity':state['identity'],'expressionAndConditionBudgetBytes':536870912,
    'accumulatedCombinedCharacters':charge,'regionVerificationBeforePromotion':True,
    'streamedAtomicCombination':True,'historicalNumericalResultsReused':False,
    'coveredInputPatterns':covered,'unfinishedInputPatterns':remaining,
    'coveragePercent':100*covered/(covered+remaining),'completeVerifiedRegions':len(rows),
    'nativeSampleCases':cases,'nativeBitMismatches':0,'exhaustiveNativeParity':False,
    'position':0,'tokens':1,'dimension':2,'focusedIntegrations':focused,'portableSuite':portable,
    'partitionTests':9,'fullCoordinateArtifactEmitted':False,'fullCoordinateParity':False,
    'multipleTokenParity':False,'fullVectorParity':False}
(folder/'frontier.json').write_text(json.dumps(state,indent=2)+'\n')
(folder/'artifact-map.json').write_text(json.dumps(rows,indent=2)+'\n')
(folder/'validation.json').write_text(json.dumps(value,indent=2)+'\n')
for name in ('docs/test-suite-map.json','docs/direct-string-validation.json'):
    path=Path(name);raw=path.read_text();old=json.loads(raw);key='validatedVerifiedRegionBudget'
    if key in old:assert old[key]==value
    else:
        text=raw.rstrip()[:-1].rstrip()+',\n  '+json.dumps(key)+': '+json.dumps(value,indent=2).replace('\n','\n  ')+'\n}\n'
        assert {k:v for k,v in json.loads(text).items()if k!=key}==old
        path.write_text(text)
print(json.dumps({'coveragePercent':value['coveragePercent'],'verifiedRegions':len(rows),
    'nativeCases':cases,'accumulatedCharacters':charge,'fullCoordinateParity':False}))
