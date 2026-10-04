import pathlib,sys,subprocess,json,time,resource
out=pathlib.Path(__file__).parent;checkpoint='docs/evidence/direct-sympy-test-checkpoint';report={'runs':[],'coordinateComplete':False}
def run(name,args):
 start=time.monotonic()
 with (out/(name+'.log')).open('w') as f:r=subprocess.run(args,cwd=out,stdout=f,stderr=subprocess.STDOUT)
 report['runs'].append({'name':name,'seconds':time.monotonic()-start,'exitCode':r.returncode,'childrenPeakRSSKiB':resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss});(out/'report.json').write_text(json.dumps(report,indent=2));return r.returncode
for test in ('direct_sympy_signs_test.py','direct_sympy_savepoints_test.py'):
 if run(test,[sys.executable,'helpers/'+test]):raise RuntimeError('Numerical gate failed')
code=run('compile',[sys.executable,'helpers/direct_sympy_scan_cli.py',checkpoint,str(out/'coordinate.expr'),'--savepoint-directory',str(out/'state'),'--compressed-savepoints','--max-characters',str(16*1024**3),'--max-seconds','600','--workers','1','--memory-mib','98304'])
if code and 'budget' not in (out/'compile.log').read_text().lower():raise RuntimeError('Compilation failed outside resource budget')
if run('parity',[sys.executable,'helpers/direct_sympy_frontier_validation.py',str(out/'state'),checkpoint,'docs/evidence/direct-sympy-offset-inputs.json']):raise RuntimeError('Saved producer parity failed')
report['terminal']='complete';(out/'report.json').write_text(json.dumps(report,indent=2));print('SIGN_PROJECTION_COMPLETE',flush=True)
