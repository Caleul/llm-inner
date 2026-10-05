import os,subprocess,json,time,sys
from pathlib import Path
root=Path('/content/llm-inner-sympy-rust-2g');os.environ['PATH']=str(Path.home()/'.cargo/bin')+':'+os.environ['PATH']
def run(name,cmd,cwd=root,timeout=900):
 start=time.monotonic()
 with (root/(name+'.log')).open('w') as log:
  try:
   p=subprocess.run(cmd,cwd=cwd,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
   result={'exitCode':p.returncode,'seconds':time.monotonic()-start,'command':cmd}
  except Exception as e:result={'exitCode':-1,'seconds':time.monotonic()-start,'error':str(e)}
 (root/(name+'-result.json')).write_text(json.dumps(result,indent=2));return result['exitCode']
if run('rust-install',['bash','-c','curl --proto =https --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal'],timeout=180)==0:
 run('rust-tests',['python','-m','unittest','direct_sympy_rust_test'],cwd=root/'helpers',timeout=120)
run('incremental-2g',['python','helpers/direct_sympy_checkpoint_run.py',str(root/'docs/evidence/direct-sympy-test-checkpoint'),str(root/'incremental-2g-state'),str(root/'incremental-2g-report.json'),'--dimension','0','--max-characters','2147483648','--max-seconds','300','--max-address-space-mib','8192'],timeout=360)
run('architecture-2g',['python','helpers/direct_sympy_architecture_run.py',str(root/'docs/evidence/direct-sympy-test-checkpoint'),str(root/'architecture-2g'),'--reference',str(root/'docs/evidence/direct-sympy-envelope-sync-reference.json'),'--workers','2','--memory-mib','8192','--max-characters','2147483648','--seconds-per-length','120','--lower','--lower-seconds','60'],timeout=3600)
