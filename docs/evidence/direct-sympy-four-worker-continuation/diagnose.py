"""Measure a failed region before flat path distribution, without expansion."""
from pathlib import Path
import json,sys,time
sys.path.insert(0,str(Path('helpers').resolve()))
from direct_sympy_checkpoint import CheckpointStrings
from direct_sympy_partition_run import compile_region,decode
from direct_sympy_streaming_literals import streaming_literals
from direct_sympy_strings import StringCompiler
root=Path(__file__).resolve().parent
domains=json.loads(Path('artifacts/direct-sympy-input-partitions/square-probe/four-worker-probe-domains.json').read_text())
started=time.monotonic()
with CheckpointStrings('docs/evidence/direct-sympy-test-checkpoint',StringCompiler(max_characters=8388608),input_domains=decode(domains)) as model:
    with streaming_literals(model) as registry:
        expression=model.coordinate(2)
        report={'inputDomains':domains,'logicalCharacters':registry.size(expression),
            'storedCharacters':sum(map(len,registry.definitions)),'dominantSquareProofs':model.rms_dominant_squares,
            'constantRMSComponents':model.rms_constant_components,'secondsBeforeFlatDistribution':time.monotonic()-started,
            'producerGrowth':[{'producer':key,'storedCharacters':size,'expandedCharacters':expanded} for key,size,expanded,_ in registry.events]}
(root/'diagnosis.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
result=compile_region('docs/evidence/direct-sympy-test-checkpoint',2,decode(domains),root/'probe.expr',
    max_characters=1048576,cas_characters=8388608,max_paths=128,max_seconds=30)
(root/'extended-budget.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
