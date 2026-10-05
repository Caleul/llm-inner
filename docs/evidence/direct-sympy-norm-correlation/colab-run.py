"""Run current-source CPU compilation and isolated CUDA checks via Broker."""
import json,os,platform,subprocess,sys,time,traceback
from pathlib import Path

root=Path('/content/llm-inner-norm-correlation');os.chdir(root)
sys.path.insert(0,str(root/'helpers'))
os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
os.environ['LLM_INNER_DIRECT_JSON_CHECKPOINT']=str(root/'docs/evidence/direct-sympy-test-checkpoint')
out=root/'results';out.mkdir(exist_ok=True)
report={'terminal':False,'fullCoordinateParity':False,'runs':[]}
def save():
    p=out/'report.tmp';p.write_text(json.dumps(report,indent=2)+'\n');p.replace(out/'report.json')
def child(name,command):
    start=time.monotonic()
    with (out/(name+'.log')).open('w') as log:
        result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
    report['runs'].append({'name':name,'exitCode':result.returncode,'seconds':time.monotonic()-start});save()
    if result.returncode:raise RuntimeError(name+' failed; inspect its retained log')
try:
    import torch
    from direct_sympy_cover_regions import live_identity
    from direct_sympy_input_partitions import interval
    from direct_sympy_partition_parallel import compile_wave
    from direct_sympy_partition_run import cardinality,encode,compile_region
    from direct_sympy_region_parity import verify_region
    torch.set_num_threads(1)
    checkpoint=str(root/'docs/evidence/direct-sympy-test-checkpoint')
    report['resources']={'cpu':os.cpu_count(),'platform':platform.platform(),
        'memory':Path('/proc/meminfo').read_text().splitlines()[:3],
        'gpu':subprocess.run(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader'],capture_output=True,text=True).stdout.strip()}
    report['compilerIdentity']=live_identity(checkpoint,2);save()
    # CUDA lives in a child process; CPU CAS pools never inherit CUDA state.
    child('cuda',[sys.executable,'helpers/direct_sympy_cuda_validation.py'])
    child('norm-tests',[sys.executable,'helpers/direct_sympy_norm_correlation_test.py'])
    domains=[{'X1':interval(*x),'X2':interval(*y)} for x in ((-2**-16,0),(2**-24,2**-16)) for y in ((-65504,-1),(1,65504))]
    report['comparison']=[]
    for workers in (1,2):
        destination=out/('workers-'+str(workers));destination.mkdir(exist_ok=True)
        jobs=[(str(i),encode(domain),destination/(str(i)+'.expr')) for i,domain in enumerate(domains)]
        started=time.monotonic()
        compiled,stats=compile_wave(checkpoint,2,jobs,workers=workers,memory_bytes=3*2**30,
            cas_characters=8388608,max_characters=1048576,max_paths=128,max_seconds=60)
        if not all(r['complete'] for r in compiled):raise RuntimeError('Incomplete comparison region')
        parity=[verify_region(checkpoint,2,j[2],d,random_cases=256) for j,d in zip(jobs,domains)]
        if any(r['mismatches'] for r in parity):raise RuntimeError('Comparison parity failed')
        report['comparison'].append({'workers':workers,'seconds':time.monotonic()-started,
            'memory':stats,'compiled':compiled,'parity':parity,'artifactHashes':[r['sha256'] for r in parity]});save()
    first,second=report['comparison']
    if first['artifactHashes']!=second['artifactHashes']:raise RuntimeError('Parallel output differs')
    report['comparisonScope']='The same four regions, including native candidate parity; not full-coordinate performance.'
    report['comparisonPatterns']=sum(cardinality(encode(d)) for d in domains)
    report['speedRatio']=first['seconds']/second['seconds'];save()
    domain={'X1':interval(-2,-.0625),'X2':interval(.0625,1)}
    artifact=out/'mixed.expr'
    result=compile_region(checkpoint,2,domain,artifact,max_characters=16777216,
        cas_characters=16777216,max_paths=128,max_seconds=300)
    report['mixedCompilation']=result;save()
    if not result['complete']:raise RuntimeError('Mixed region did not complete within its explicit budget')
    report['mixedParity']=verify_region(checkpoint,2,artifact,domain,random_cases=8192);save()
    if report['mixedParity']['mismatches']:raise RuntimeError('Mixed actual-file parity failed')
    report['terminal']=True;report['success']=True;save()
except BaseException:
    report['terminal']=True;report['success']=False;report['error']=traceback.format_exc();save();raise
