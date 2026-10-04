from pathlib import Path
import hashlib,json,os,subprocess,sys,tarfile
archive=Path('/content/llm-inner-input-partitions.tar.gz')
expected='7b7f8ab5a4708699f47917b76aa03aad7d03b6f6a16f5a001f26b43e6d1e2e70'
if hashlib.sha256(archive.read_bytes()).hexdigest()!=expected:raise RuntimeError('Source archive integrity mismatch')
root=Path('/content/llm-inner-input-partitions');root.mkdir(exist_ok=True)
status=root/'job.json'
if status.exists():
    print(status.read_text())
    raise SystemExit('Existing job must be inspected; refusing duplicate launch')
with tarfile.open(archive) as stream:
    for member in stream.getmembers():
        if member.name.startswith('/') or '..' in Path(member.name).parts or not (member.isfile() or member.isdir()):raise RuntimeError('Invalid source archive member')
    stream.extractall(root)
job=root/'run_job.py'
job.write_text('''from pathlib import Path
import hashlib,json,os,subprocess,sys,time,traceback,shutil
root=Path(__file__).parent;os.chdir(root)
os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
os.environ['LLM_INNER_DIRECT_JSON_CHECKPOINT']=str(root/'docs/evidence/direct-sympy-test-checkpoint')
out=root/'results';out.mkdir(exist_ok=True)
report={'state':'RUNNING','runs':[],'fullCoordinateParity':False}
def save():
    path=root/'job.json';temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(report,indent=2));os.replace(temporary,path)
def run(name,args):
    start=time.monotonic()
    with (out/(name+'.log')).open('w') as stream:p=subprocess.run(args,stdout=stream,stderr=subprocess.STDOUT)
    row={'name':name,'exitCode':p.returncode,'seconds':time.monotonic()-start};report['runs'].append(row);save();return p.returncode
try:
    save()
    if run('dependencies',[sys.executable,'-m','pip','install','--quiet','sympy==1.14.0','transformers==5.5.0','safetensors==0.8.0']):raise RuntimeError('Dependency installation failed')
    import torch,sympy,psutil
    report['runtime']={'torch':str(torch.__version__),'sympy':sympy.__version__,'cpuCount':os.cpu_count(),'ramBytes':psutil.virtual_memory().total,'cuda':torch.version.cuda,'cudaAvailable':torch.cuda.is_available(),'device':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,'clang':shutil.which('clang++')};save()
    if not shutil.which('clang++'):raise RuntimeError('Native parity compiler unavailable')
    if torch.cuda.is_available():run('cuda-numerics',[sys.executable,'helpers/direct_sympy_cuda_validation.py'])
    # Symbolic factor/simplify and exact CPU-reference parity remain on CPU.
    for name in ('input_partitions','parallel'):
        if run(name+'-tests',[sys.executable,'helpers/direct_sympy_'+name+'_test.py']):raise RuntimeError('CPU validation failed: '+name)
    state=root/'state'
    command=[sys.executable,'helpers/direct_sympy_partition_run.py',os.environ['LLM_INNER_DIRECT_JSON_CHECKPOINT'],str(state),'--max-attempts','12','--region-seconds','12']
    code=run('initial-partitions',command)
    if code not in (0,1):raise RuntimeError('Partition process failed')
    if not (state/'frontier.json').exists():raise RuntimeError('No persistent partition frontier')
    manifest=json.loads((state/'frontier.json').read_text());report['coverage']={k:manifest.get(k) for k in ('coveredInputPatterns','unfinishedInputPatterns','totalInputPatterns','finalArtifactEmitted','finalParity')};save()
    if run('native-parity',[sys.executable,'helpers/direct_sympy_region_parity.py',os.environ['LLM_INNER_DIRECT_JSON_CHECKPOINT'],str(state),str(out/'native-parity.json')]):raise RuntimeError('Emitted region parity failed')
    report['state']='COMPLETED';save()
except BaseException as e:
    report['state']='FAILED';report['error']=str(e);save();traceback.print_exc();raise
''')
with (root/'job.log').open('w') as log:
    process=subprocess.Popen([sys.executable,str(job)],cwd=root,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({'pid':process.pid,'root':str(root),'archiveSHA256':expected,'jobStarted':True}))
