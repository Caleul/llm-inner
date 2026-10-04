from pathlib import Path
import hashlib,json,subprocess,sys,tarfile
archive=Path('/content/llm-inner-input-partitions.tar.gz')
if hashlib.sha256(archive.read_bytes()).hexdigest()!='7b7f8ab5a4708699f47917b76aa03aad7d03b6f6a16f5a001f26b43e6d1e2e70':raise RuntimeError('Archive identity mismatch')
root=Path('/content/llm-inner-input-partitions');root.mkdir(exist_ok=True)
if (root/'comparison-preparation.json').exists() or (root/'comparison-job.json').exists():raise RuntimeError('Already dispatched; inspect existing job')
with tarfile.open(archive) as stream:
 for member in stream.getmembers():
  if member.name.startswith('/') or '..' in Path(member.name).parts or not (member.isfile() or member.isdir()):raise RuntimeError('Invalid archive member')
 stream.extractall(root)
(root/'results').mkdir(exist_ok=True)
(root/'compare.py').write_bytes(Path('/content/compare.py').read_bytes())
job=root/'prepare_comparison.py'
job.write_text('''from pathlib import Path
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
''')
with (root/'preparation-launch.log').open('w') as log:p=subprocess.Popen([sys.executable,str(job)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({'pid':p.pid,'started':True,'root':str(root)}))
