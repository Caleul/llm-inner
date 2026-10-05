"""Bind structural rule/regression evidence without claiming a final model file."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
folder=Path(__file__).resolve().parent
def counts(name):
    text=(folder/name).read_text()
    return {key:int(re.search(r'ℹ '+key+r' (\d+)',text)[1])for key in ('tests','pass','fail','skipped')}
structural=counts('structural-tests.log');portable=counts('portable-tests.log');integration=counts('integration-tests.log')
assert structural=={'tests':37,'pass':36,'fail':0,'skipped':1}
assert portable=={'tests':624,'pass':575,'fail':0,'skipped':49}
assert integration=={'tests':2,'pass':2,'fail':0,'skipped':0}
assert 'Ran 8 tests' in (folder/'parallel-tests.log').read_text()
parity=re.search(r'Structural coordinate parity: position=0 dimension=2 cases=(\d+) tokenLengths=([^ ]+) mismatches=0 artifactEmitted=false',(folder/'integration-tests.log').read_text())
assert parity is not None
growths=[[json.loads(line)for line in (folder/name).read_text().splitlines()]for name in ('baseline-growth.jsonl','current-growth.jsonl')]
assert len(growths[0])==len(growths[1])
changed=sum(a!=b for a,b in zip(*growths))
sizes=[]
for name in ('baseline-coordinate-run.log','coordinate-run.log'):
    text=(folder/name).read_text()
    assert 'JSON output budget exceeded before expression emission' in text
    sizes.append(int(re.search(r'expressionBytes=(\d+)',text)[1]))
assert sizes==[56393457973004369801]*2
benchmark=json.loads((folder/'parallel-benchmark.json').read_text())
assert benchmark['identicalDomainsAndArtifactHashes'] and not benchmark['fullCoordinateParity']
assert all(benchmark[key]['nativeMismatches']==0 for key in ('sequential','parallel'))
files=['src/direct-json-global-integer-factor.ts','src/direct-json-simplify.ts','helpers/direct_sympy_partition_parallel.py']
baseline_commit='9e6f3ad43c46a6d9a10d16f6a5b22927c5611bf7'
baseline_source=subprocess.check_output(['git','show',baseline_commit+':src/direct-json-simplify.ts'])
assert Path('/private/tmp/llm-inner-json-factor-baseline-clean-20261005/src/direct-json-simplify.ts').read_bytes()==baseline_source
value={'baselineCommit':'9e6f3ad43c46a6d9a10d16f6a5b22927c5611bf7',
    'baselineSimplifierSHA256':hashlib.sha256(baseline_source).hexdigest(),
    'checkpoint':{path.name:hashlib.sha256(path.read_bytes()).hexdigest()for path in sorted(Path('docs/evidence/direct-sympy-test-checkpoint').iterdir())if path.suffix in ('.json','.safetensors')},
    'numericProfiles':{path.name:hashlib.sha256(path.read_bytes()).hexdigest()for path in sorted(Path('numeric-profiles').glob('*.bin'))},
    'sources':{name:hashlib.sha256(Path(name).read_bytes()).hexdigest()for name in files},
    'structuralJsonSourceOfTruth':True,'unsignedModularGlobalSumFactoring':True,
    'floatingReassociationIntroduced':False,'partialOperationsPreserved':True,
    'searchDeclinesWithoutTruncation':True,'modularParityCases':71680,
    'structuralTests':structural,'portableSuite':portable,'nativeIntegrations':integration,
    'structuralCoordinateCorpusCases':int(parity[1]),'structuralCoordinateTokenLengths':list(map(int,parity[2].split(','))),
    'structuralCoordinatePosition':0,'structuralCoordinateDimension':2,'structuralCoordinateBitMismatches':0,
    'expressionAndConditionBudgetBytes':536870912,'wholeCoordinatePredictedBytesBefore':str(sizes[0]),
    'wholeCoordinatePredictedBytesAfter':str(sizes[1]),'differentGrowthRecords':changed,
    'checkpointSizeReductionDemonstrated':False,'parallelAccumulatedAdmissionBeforeParity':True,
    'parallelVerificationPolicyDowngradeRejected':True,
    'parallelRegionalBenchmark':benchmark,'fullCoordinateArtifactEmitted':False,'fullCoordinateParity':False,
    'multipleTokenLastPositionParity':False,'fullVectorParity':False}
(folder/'validation.json').write_text(json.dumps(value,indent=2)+'\n')
for name in ('docs/test-suite-map.json','docs/direct-string-validation.json'):
    path=Path(name);raw=path.read_text();old=json.loads(raw);key='validatedGlobalIntegerFactor'
    if key in old:
        if old[key]!=value:
            assert list(old)[-1]==key
            prefix=raw[:raw.rindex('\n  '+json.dumps(key)+': ')]
            text=prefix+'\n  '+json.dumps(key)+': '+json.dumps(value,indent=2).replace('\n','\n  ')+'\n}\n'
            assert {k:v for k,v in json.loads(text).items()if k!=key}=={k:v for k,v in old.items()if k!=key}
            path.write_text(text)
    else:
        text=raw.rstrip()[:-1].rstrip()+',\n  '+json.dumps(key)+': '+json.dumps(value,indent=2).replace('\n','\n  ')+'\n}\n'
        assert {k:v for k,v in json.loads(text).items()if k!=key}==old
        path.write_text(text)
print(json.dumps({'growthRecordsChanged':changed,'coordinateCorpusCases':int(parity[1]),'checkpointSizeReduction':False,'portable':portable}))
