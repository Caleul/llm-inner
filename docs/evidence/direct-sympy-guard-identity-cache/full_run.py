"""First-coordinate attempt with measured RAM, no admission from CAS worst-case."""
import json,os,signal,subprocess,sys,time
from pathlib import Path
import psutil
sys.path.insert(0,str(Path('helpers').resolve()))
from direct_sympy_artifact_budget import ArtifactBudget
from direct_sympy_logits_run import snapshot
folder=Path(__file__).resolve().parent;output=Path('/private/tmp/llm-inner-guard-identity-cache-snapshot-512-20261005');output.mkdir(exist_ok=False)
limit=512*1024**2;memory=psutil.virtual_memory().available*4//5
budget=ArtifactBudget(output/'artifact-budget.json',limit,create=True);budget.change('vector-syntax',15,exact=True)
start=time.monotonic();peak=0;stop=None
with (output/'worker.log').open('w')as log:
 process=subprocess.Popen([sys.executable,str(Path('helpers/direct_sympy_logits_run.py').resolve()),str(Path('docs/evidence/direct-sympy-test-checkpoint').resolve()),str(output),'--worker','--dimension','2','--expression-bytes',str(limit),'--max-paths','2048','--parity-cases','8192'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
 while process.poll()is None:
  rss=psutil.Process().memory_info().rss+snapshot(process);peak=max(peak,rss)
  if rss>memory or time.monotonic()-start>600:
   stop='Measured RAM budget exceeded'if rss>memory else'Wall-clock budget exceeded';os.killpg(process.pid,signal.SIGKILL);break
  time.sleep(.2)
 process.wait()
if stop:
 budget.change('logit-2',0,exact=True)
 for name in ('logit-2.expr','.logit-2.pending.expr'):(output/name).unlink(missing_ok=True)
report={'expressionAndConditionBudgetBytes':limit,'measuredRAMBudgetBytes':memory,'peakObservedRSSBytes':peak,'seconds':time.monotonic()-start,'workerExitCode':process.returncode,'stop':stop,'previousControllerAdmissionStop':'Conservative 16xCAS reservation exceeded 80% available RAM; no worker dispatched by controller'}
result=output/'logit-2.json'
if result.exists():report['coordinate']=json.loads(result.read_text())
(folder/'full-run.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items()if k!='coordinate'}))
