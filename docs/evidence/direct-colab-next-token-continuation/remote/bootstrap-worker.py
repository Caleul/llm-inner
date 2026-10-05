import json,os,sys,time,subprocess,urllib.request,tarfile,psutil,importlib.util
from pathlib import Path
root=Path('/content/llm-inner-final-logits-eligible-candidates');start=time.monotonic()
def record(name,data):(root/name).write_text(json.dumps(data,indent=2))
try:
 import torch
 resources={'cpuCount':os.cpu_count(),'RAMBytes':psutil.virtual_memory().total,'cudaAvailable':torch.cuda.is_available(),'torch':torch.__version__}
 if torch.cuda.is_available():resources.update({'GPU':torch.cuda.get_device_name(),'GPUBytes':torch.cuda.get_device_properties(0).total_memory})
 record('resources.json',resources)
 node=root/'node-v22.22.2-linux-x64/bin/node'
 if not node.exists():
  urllib.request.urlretrieve('https://nodejs.org/dist/v22.22.2/node-v22.22.2-linux-x64.tar.xz',root/'node.tar.xz')
  with tarfile.open(root/'node.tar.xz') as archive:archive.extractall(root,filter='data')
 missing=[m for m in ['safetensors','sympy','psutil'] if importlib.util.find_spec(m) is None]
 if missing:subprocess.run([sys.executable,'-m','pip','install',*missing],check=True,timeout=180)
 files=['dist/test/direct-json-cofactor-rounds.test.js','dist/test/direct-json-cofactor-parallel.test.js','dist/test/direct-json.test.js','dist/test/direct-json-next-token.test.js']
 with (root/'focused.log').open('w') as log:
  t=time.monotonic();p=subprocess.Popen([str(node),'--test',*files],cwd=root,stdout=log,stderr=subprocess.STDOUT);peak=0
  while p.poll() is None:
   try:q=psutil.Process(p.pid);peak=max(peak,sum(x.memory_info().rss for x in [q,*q.children(recursive=True)]))
   except psutil.NoSuchProcess:pass
   time.sleep(.2)
 record('focused-result.json',{'exitCode':p.returncode,'seconds':time.monotonic()-t,'peakTreeRSSBytes':peak})
except Exception as error:record('setup-error.json',{'error':str(error),'seconds':time.monotonic()-start});raise
