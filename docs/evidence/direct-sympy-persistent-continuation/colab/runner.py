import pathlib,subprocess,sys,time,json,os
root=pathlib.Path.cwd();report={'steps':[]}
def run(name,args):
 start=time.monotonic()
 with (root/(name+'.log')).open('w') as out:p=subprocess.run(args,stdout=out,stderr=subprocess.STDOUT)
 report['steps'].append({'name':name,'exitCode':p.returncode,'seconds':time.monotonic()-start});(root/'report.json').write_text(json.dumps(report,indent=2))
 return p.returncode
code=run('compile',[sys.executable,'helpers/direct_sympy_checkpoint_run.py','docs/evidence/direct-sympy-test-checkpoint','state','diagnostic.json','--resume','--max-characters',str(16*1024**3),'--max-address-space-mib',str(72*1024),'--max-seconds','600'])
report['compileStoppedWithinBudget']=code==1
run('parity',[sys.executable,'helpers/direct_sympy_frontier_validation.py','state','docs/evidence/direct-sympy-test-checkpoint','docs/evidence/direct-sympy-offset-inputs.json'])
report['terminal']=True;(root/'report.json').write_text(json.dumps(report,indent=2))
