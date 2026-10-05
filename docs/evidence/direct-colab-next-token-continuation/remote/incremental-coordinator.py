import os,json,psutil,subprocess,time,signal,sys
from pathlib import Path
root=Path('/content/llm-inner-final-logits-eligible-candidates');t=time.monotonic();peak=0;stop=None
command=[sys.executable,'helpers/direct_sympy_checkpoint_run.py',str(root/'docs/evidence/direct-sympy-test-checkpoint'),'incremental-state','incremental-result.json','--dimension','0','--max-characters','536870912','--max-seconds','120']
with (root/'incremental.log').open('w') as log:
 p=subprocess.Popen(command,cwd=root,env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'},stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 while p.poll() is None:
  try:q=psutil.Process(p.pid);peak=max(peak,sum(x.memory_info().rss for x in [q,*q.children(recursive=True)]))
  except psutil.NoSuchProcess:pass
  if peak>4*1024**3 or time.monotonic()-t>150:
   stop='Aggregate RAM or wall-clock limit';os.killpg(p.pid,signal.SIGTERM);break
  time.sleep(.5)
 code=p.wait()
(root/'incremental-coordinator-result.json').write_text(json.dumps({'exitCode':code,'seconds':time.monotonic()-t,'peakTreeRSSBytes':peak,'RAMBudgetBytes':4*1024**3,'expressionConditionBudgetBytes':512*1024**2,'stop':stop},indent=2))
