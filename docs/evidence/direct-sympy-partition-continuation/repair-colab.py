from pathlib import Path
import json,subprocess,sys
root=Path('/content/llm-inner-input-partitions')
status=json.loads((root/'job.json').read_text())
if status['state']!='FAILED' or status.get('error')!='Native parity compiler unavailable':raise RuntimeError('Unexpected job state; refusing restart')
if (root/'repair.json').exists():
    print((root/'repair.json').read_text());raise SystemExit('Repair already dispatched; inspect before any retry')
repair=root/'repair_native.py'
repair.write_text('''from pathlib import Path
import json,subprocess,sys,os,time
root=Path(__file__).parent
status={'state':'RUNNING','action':'install-native-compiler'}
def save():
 p=root/'repair.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(status));os.replace(t,p)
save()
with (root/'native-install.log').open('w') as log:
 for command in (['apt-get','update'],['apt-get','install','-y','--no-install-recommends','clang']):
  p=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
  if p.returncode:status.update(state='FAILED',exitCode=p.returncode);save();raise SystemExit(p.returncode)
status['state']='COMPLETED';save()
(root/'initial-failed-job.json').write_bytes((root/'job.json').read_bytes())
with (root/'job.log').open('a') as log:subprocess.run([sys.executable,str(root/'run_job.py')],cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
''')
with (root/'repair.log').open('w') as log:
 p=subprocess.Popen([sys.executable,str(repair)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({'repairPid':p.pid,'repairDispatched':True}))
