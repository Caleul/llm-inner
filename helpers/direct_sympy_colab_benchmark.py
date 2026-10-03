"""Controlled one-coordinate Colab comparison, run remotely via Access Broker.

Invoke in the uploaded repository root in a fresh CPU process. CUDA validation
uses its own child process. Never initialize CUDA before fork-based CAS pools.
Actual producer growth and sampled RSS accompany timings; no final coordinate
is admitted by a reference-boundary test or by a budget-limited frontier.
"""
import hashlib,json,os,pathlib,resource,subprocess,sys,time
root=pathlib.Path.cwd();out=root/'results';out.mkdir(exist_ok=True)
os.environ['LLM_INNER_DIRECT_JSON_CHECKPOINT']=str(root/'docs/evidence/direct-sympy-test-checkpoint')
os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
report={'runs':[],'finalCoordinateAdmitted':False}
def run(name,command):
    start=time.monotonic()
    with (out/(name+'.log')).open('w') as stream:
        process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT)
        peak=0
        while process.poll() is None:
            try:
                ids=[process.pid]
                children=pathlib.Path(f'/proc/{process.pid}/task/{process.pid}/children').read_text().split()
                ids.extend(map(int,children))
                rss=0
                for pid in ids:
                    for line in pathlib.Path(f'/proc/{pid}/status').read_text().splitlines():
                        if line.startswith('VmRSS:'):rss+=int(line.split()[1])*1024
                peak=max(peak,rss)
            except FileNotFoundError:pass
            time.sleep(.2)
    item={'name':name,'exitCode':process.returncode,'seconds':time.monotonic()-start,'peakSampledRSSBytes':peak}
    report['runs'].append(item);(out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(item),flush=True)
    return process.returncode
run('cuda-numerics',[sys.executable,'helpers/direct_sympy_cuda_validation.py'])
for module in ('parallel','savepoints','strings','scan_backend','checkpoint'):
    code=run(module+'-tests',[sys.executable,'helpers/direct_sympy_'+module+'_test.py'])
    if code:raise SystemExit('Refusing compilation after failed '+module+' validation')
for name,workers in (('sequential',1),('parallel',2)):
    command=[sys.executable,'helpers/direct_sympy_scan_cli.py',os.environ['LLM_INNER_DIRECT_JSON_CHECKPOINT'],str(out/name),
        '--max-characters',str(9*1024**3),'--max-seconds','600','--workers',str(workers),'--memory-mib','98304','--block-size','1',
        '--savepoint-directory',str(out/(name+'-state'))]
    run(name,command)
    growth=out/(name+'.growth.tsv')
    if growth.exists():
        rows=growth.read_text().splitlines()[1:];report[name]={'completedDependencies':len(rows),'growth':rows}
    frontier=out/(name+'-state/frontier.json')
    if frontier.exists():
        payload=json.loads(frontier.read_text())['payload']
        report.setdefault(name,{})['records']=payload['records'];report[name]['identity']=payload['identity']
    (out/'report.json').write_text(json.dumps(report,indent=2))
print('COMPARISON_COMPLETE',flush=True)
