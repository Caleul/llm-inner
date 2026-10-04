from pathlib import Path
import tarfile,hashlib,json,subprocess,sys,os,time
root=Path('/content/llm-inner-current');root.mkdir(exist_ok=True)
with tarfile.open('/content/llm-inner-current-sources.tar.gz') as tar:tar.extractall(root,filter='data')
manifest=json.loads((root/'source-manifest.json').read_text())
for name,digest in manifest['files'].items():
 if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:raise RuntimeError('Source archive integrity mismatch: '+name)
with (root/'preparation.log').open('w') as log:
 p=subprocess.run([sys.executable,'-m','pip','install','--quiet','sympy==1.14.0','transformers==5.5.0','safetensors==0.8.0','psutil'],stdout=log,stderr=subprocess.STDOUT)
if p.returncode:raise RuntimeError('Colab dependency preparation failed')
(root/'results').mkdir(exist_ok=True)
import psutil,torch,sympy
runtime={'sourceCommit':manifest['commit'],'torch':str(torch.__version__),'sympy':sympy.__version__,'cpuCount':os.cpu_count(),'ramBytes':psutil.virtual_memory().total,'cudaAvailable':torch.cuda.is_available(),'device':torch.cuda.get_device_name() if torch.cuda.is_available() else None}
(root/'runtime.json').write_text(json.dumps(runtime,indent=2)+'\n')
with (root/'comparison.log').open('w') as log:
 p=subprocess.run([sys.executable,str(root/'compare.py')],stdout=log,stderr=subprocess.STDOUT)
print(json.dumps(runtime));print((root/'comparison-job.json').read_text())
if p.returncode:raise RuntimeError('Colab comparison failed: '+(root/'comparison.log').read_text()[-3000:])
