from pathlib import Path
import hashlib,json,subprocess,sys,shutil,os,statistics,tarfile
root=Path('/content/llm-inner-current');os.chdir(root);sys.path.insert(0,str(root/'helpers'))
report=json.loads((root/'comparison-job.json').read_text())
if len(report['runs'])!=6 or any(row['exitCode'] for row in report['runs']):raise RuntimeError('Six completed compiler runs required')
manifest=json.loads((root/'source-manifest.json').read_text())
for name,digest in manifest['files'].items():
 if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:raise RuntimeError('Benchmark sources changed: '+name)
if shutil.which('clang++') is None:
 with (root/'compiler-preparation.log').open('w') as log:
  for command in (['apt-get','update','-qq'],['apt-get','install','-y','-qq','clang']):
   p=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
   if p.returncode:raise RuntimeError('Native verifier compiler preparation failed')
from direct_sympy_input_partitions import interval
from direct_sympy_region_parity import verify_region
report['medians']={str(w):{'compileSeconds':statistics.median(r['compileSeconds'] for r in report['runs'] if r['workers']==w),'wallSeconds':statistics.median(r['wallSeconds'] for r in report['runs'] if r['workers']==w),'sampledAggregateRSSBytes':statistics.median(r['sampledAggregateRSSBytes'] for r in report['runs'] if r['workers']==w)} for w in (1,2)}
report['identicalArtifacts']=len({r['artifact']['sha256'] for r in report['runs']})==1
for row in report['runs']:
 p=Path(row['artifact']['path'])
 if hashlib.sha256(p.read_bytes()).hexdigest()!=row['artifact']['sha256']:raise RuntimeError('Compiled artifact changed')
report['regionParity']=verify_region(root/'docs/evidence/direct-sympy-test-checkpoint',2,report['runs'][0]['artifact']['path'],{'X1':interval(32,65504),'X2':interval(32,65504)},random_cases=1024)
report['state']='COMPLETED' if report['identicalArtifacts'] and report['regionParity']['mismatches']==0 else 'FAILED'
report['verificationRepair']='Compiler runs were not repeated; clang was installed after the original native readback failed because clang++ was absent.'
(root/'comparison-job.json').write_text(json.dumps(report,indent=2)+'\n')
archive=root/'evidence.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
 for name in ('comparison-job.json','source-manifest.json','runtime.json','compare.py','comparison.log','preparation.log','compiler-preparation.log'):
  if (root/name).exists():tar.add(root/name,arcname=name)
 tar.add(root/'results',arcname='results')
print(json.dumps(report))
print(json.dumps({'archive':str(archive),'sha256':hashlib.sha256(archive.read_bytes()).hexdigest()}))
if report['state']!='COMPLETED':raise RuntimeError('Compiled regional parity failed')
