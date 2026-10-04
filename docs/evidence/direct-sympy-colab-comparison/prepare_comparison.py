from pathlib import Path
import json,os,subprocess,sys,time,traceback
root=Path(__file__).parent
status={'state':'RUNNING','pid':os.getpid(),'startedAt':time.time()}
def save():
 p=root/'comparison-preparation.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(status,indent=2));os.replace(t,p)
try:
 save()
 with (root/'preparation.log').open('w') as log:
  p=subprocess.run([sys.executable,'-m','pip','install','--quiet','sympy==1.14.0','transformers==5.5.0','safetensors==0.8.0','psutil'],stdout=log,stderr=subprocess.STDOUT)
 if p.returncode:raise RuntimeError('Dependency preparation failed')
 import torch,sympy,psutil
 status['runtime']={'torch':str(torch.__version__),'sympy':sympy.__version__,'cpuCount':os.cpu_count(),'ramBytes':psutil.virtual_memory().total,'cuda':torch.version.cuda,'cudaAvailable':torch.cuda.is_available(),'device':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None};save()
 with (root/'comparison-launch.log').open('w') as log:p=subprocess.run([sys.executable,str(root/'compare.py')],stdout=log,stderr=subprocess.STDOUT)
 if p.returncode:raise RuntimeError('Comparison failed')
 status['state']='COMPLETED';save()
except BaseException as e:
 status.update(state='FAILED',error=str(e));save();traceback.print_exc();raise
