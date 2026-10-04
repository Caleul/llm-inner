"""Compare the unchanged generic conversion with certified fixed-grid lowering."""
import hashlib,json,subprocess,sys,time
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path('helpers').resolve()))
import direct_sympy_conversions as conversions
from direct_sympy_checkpoint import CheckpointStrings
from direct_sympy_coherent_paths import CoherentPaths
from direct_sympy_input_partitions import interval
from direct_sympy_streaming_literals import streaming_literals
from direct_sympy_strings import StringCompiler
root=Path(__file__).resolve().parent
original=conversions.fixed_grid_conversion
rows=[];domains={'X1':interval(.03125,.0625),'X2':interval(-120,-96)}
for name,implementation in [('previous',lambda *_:None),('fixed-grid',original)]:
    started=time.monotonic()
    with patch.object(conversions,'fixed_grid_conversion',implementation):
        with CheckpointStrings('docs/evidence/direct-sympy-test-checkpoint',StringCompiler(max_characters=8388608),input_domains=domains) as model:
            with streaming_literals(model) as registry:
                expression=model.coordinate(2)
                report={'mode':name,'producers':len(model.memo),'logicalCharacters':registry.size(expression),
                    'storedCharacters':sum(map(len,registry.definitions)),
                    'producerGrowth':[{'producer':key,'storedCharacters':size,'expandedCharacters':expanded} for key,size,expanded,_ in registry.events]}
                plan=CoherentPaths(registry,max_paths=128)
                try:report['artifact']=plan.write(root/(name+'.expr'),expression,max_characters=1048576)
                except ValueError as e:
                    if 'budget' not in str(e).lower():raise
                    report['stop']=str(e)
                report['paths']=plan.stats
    report['seconds']=time.monotonic()-started;rows.append(report);print(json.dumps(report),flush=True)
old_source=subprocess.check_output(['git','show','4308c8e:helpers/direct_sympy_conversions.py'])
(root/'diagnosis.json').write_text(json.dumps({'baseline':'Same source with fixed_grid_conversion disabled; existing generic kernel unchanged',
    'previousSourceSHA256':hashlib.sha256(old_source).hexdigest(),
    'currentSourceSHA256':hashlib.sha256(Path('helpers/direct_sympy_conversions.py').read_bytes()).hexdigest(),
    'domains':{'X1':[.03125,.0625],'X2':[-120,-96]},'results':rows},indent=2)+'\n')
