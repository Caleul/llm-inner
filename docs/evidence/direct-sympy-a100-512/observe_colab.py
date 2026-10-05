"""Read real job state and process liveness; never dispatch work."""
import json,os
from pathlib import Path
root=Path('/content/llm-inner-a100-512-20261005')
report={}
for label,path in [('launch',root.with_suffix('.launch.json')),('preparation',root/'preparation.json'),('run',root/'logits-results/run.json')]:
 if path.exists():report[label]=json.loads(path.read_text())
for label,field in [('launch','preparationPID'),('preparation','controllerPID')]:
 if label not in report or field not in report[label]:continue
 pid=report[label][field]
 try:os.kill(pid,0);alive=True
 except ProcessLookupError:alive=False
 report[label]['observedPIDExists']=alive
 stat=Path(f'/proc/{pid}/stat')
 if stat.exists():
  report[label]['processState']=stat.read_text().split(') ',1)[1].split()[0]
  report[label]['observedRunning']=report[label]['processState']not in ('Z','X')
 else:report[label]['observedRunning']=False
folder=root/'logits-results'
report['coordinateProgress']={p.name:json.loads(p.read_text())for p in folder.glob('*.progress.json')}
report['logTails']={}
for p in [root.with_suffix('.preparation-launch.log'),*root.glob('*tests.log'),root/'dependency-setup.log',root/'controller.log',folder/'cuda-numerics.log']:
 if p.exists():
  with p.open('rb')as stream:
   stream.seek(max(0,p.stat().st_size-2500));report['logTails'][p.name]=stream.read().decode(errors='replace')
report['coordinateResults']={p.name:json.loads(p.read_text())for p in folder.glob('logit-?.json')}
print(json.dumps(report,indent=2))
