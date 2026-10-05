"""Read real job state and process liveness; never dispatch work."""
import json,os
from pathlib import Path
root=Path('/content/llm-inner-h100-512-20261005')
report={}
for label,path in [('launch',root.with_suffix('.launch.json')),('preparation',root/'preparation.json'),('run',root/'logits-results/run.json')]:
 if path.exists():report[label]=json.loads(path.read_text())
for label,field in [('launch','preparationPID'),('preparation','controllerPID')]:
 if label not in report or field not in report[label]:continue
 pid=report[label][field]
 try:os.kill(pid,0);alive=True
 except ProcessLookupError:alive=False
 report[label]['observedPIDExists']=alive
folder=root/'logits-results'
report['coordinateProgress']={p.name:json.loads(p.read_text())for p in folder.glob('*.progress.json')}
print(json.dumps(report,indent=2))
