"""Measure actual producer growth without constructing the expanded string."""
from pathlib import Path
import json,sys
from collections import Counter
sys.path.insert(0,str(Path('helpers').resolve()))
from direct_sympy_checkpoint import CheckpointStrings
from direct_sympy_partition_run import decode
from direct_sympy_streaming_literals import streaming_literals,ALIASES
from direct_sympy_strings import StringCompiler
reports=[]
for limit in (8192,12288,19455):
    domains={'X1':[-limit,limit],'X2':[-31743,-20480]}
    with CheckpointStrings('docs/evidence/direct-sympy-test-checkpoint',StringCompiler(max_characters=8388608),input_domains=decode(domains)) as model:
        with streaming_literals(model) as registry:
            expression=model.coordinate(2)
            definitions=[{'definition':i,'storedCharacters':len(body),'expandedCharacters':registry.size('CompileValue'+str(i)+'()'),'directDependencyOccurrences':dict(Counter(int(m[1]) for m in ALIASES.finditer(body)))} for i,body in enumerate(registry.definitions)]
            reports.append({'domains':domains,'logicalCharacters':registry.size(expression),'storedCharacters':sum(map(len,registry.definitions)),
                'producerGrowth':[{'producer':key,'storedCharacters':stored,'expandedCharacters':expanded,'definitions':count} for key,stored,expanded,count in registry.events],
                'definitions':definitions,'finalArtifactEmitted':False})
Path(__file__).with_name('growth.json').write_text(json.dumps(reports,indent=2)+'\n')
for report in reports:
    print(report['domains'],report['logicalCharacters'])
    for event in report['producerGrowth'][-6:]:print(event)
