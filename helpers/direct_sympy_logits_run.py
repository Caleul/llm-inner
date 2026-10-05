"""Memory-admitted independent logits, gated by one complete coordinate.

Colab entry point via Access Broker colab_cli. Every worker owns its source,
CAS and branch context. Numerical order inside a coordinate is unchanged.
512 MiB bounds accumulated emitted expressions AND conditions across logits;
RAM has a separate measured limit. CUDA is used only by parity-audited numerical
batches, never for factor/simplify or an unaudited reference replacement.
"""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from direct_sympy_savepoints import atomic,canonical,digest_file

DEFAULT_EXPRESSION_BYTES=512*1024**2

def snapshot(process):
    import psutil
    try:root=psutil.Process(process.pid)
    except psutil.NoSuchProcess:return 0
    total=0
    try:children=root.children(recursive=True)
    except psutil.NoSuchProcess:return 0
    for item in [root,*children]:
        try:total+=item.memory_info().rss
        except psutil.NoSuchProcess:pass
    return total

def worker(args):
    import torch
    from direct_sympy_artifact_budget import ArtifactBudget
    from direct_sympy_checkpoint import CheckpointStrings
    from direct_sympy_coherent_paths import CoherentPaths
    from direct_sympy_cover_regions import live_identity
    from direct_sympy_region_parity import verify_region
    from direct_sympy_streaming_literals import streaming_literals
    from direct_sympy_strings import StringCompiler
    torch.set_num_threads(1);directory=Path(args.output);result={'dimension':args.dimension,'complete':False,'parityVerified':False}
    identity=live_identity(args.checkpoint,args.dimension);result['compilerIdentity']=identity
    started=time.monotonic();target=directory/f'logit-{args.dimension}.expr'
    budget=ArtifactBudget(directory/'artifact-budget.json',args.expression_bytes)
    try:
        with budget.lease(f'logit-{args.dimension}')as lease:
            with CheckpointStrings(args.checkpoint,StringCompiler(max_characters=args.expression_bytes))as model:
                domains=dict(model.domains)
                with streaming_literals(model)as registry:
                    expression=model.coordinate(args.dimension)
                    plan=CoherentPaths(registry,max_paths=args.max_paths)
                    class ProgressLease:
                        last=0.0
                        def claim(self,amount):
                            lease.claim(amount)
                            now=time.monotonic()
                            if now-self.last<.25:return
                            self.last=now
                            progress={'dimension':args.dimension,'claimedCharacters':lease.claimed,'stats':plan.stats,'elapsedSeconds':time.monotonic()-started}
                            atomic(directory/f'logit-{args.dimension}.progress.json',canonical(progress))
                        def commit(self,amount):lease.commit(amount)
                    result['artifact']=plan.write(target,expression,max_characters=args.expression_bytes,artifact_lease=ProgressLease())
                    result['stats']=plan.stats;result['weightReads']=model.read_weights
            result['complete']=True;result['compileSeconds']=time.monotonic()-started
            result['parity']=verify_region(args.checkpoint,args.dimension,target,domains,random_cases=args.parity_cases)
            if result['parity']['mismatches']:raise ValueError('Bit parity failed; coordinate not admitted')
            if live_identity(args.checkpoint,args.dimension)!=identity:raise ValueError('Compiler identity changed during parity')
            result['parityVerified']=True
    except ValueError as error:
        result['stop']=str(error)
        if not result['parityVerified']:
            target.unlink(missing_ok=True);budget.change(f'logit-{args.dimension}',0,exact=True)
        if 'budget'not in str(error).lower():result['semanticFailure']=True
    finally:
        if not result['parityVerified']:
            target.unlink(missing_ok=True);budget.change(f'logit-{args.dimension}',0,exact=True)
        result['seconds']=time.monotonic()-started
        atomic(directory/f'logit-{args.dimension}.json',canonical(result))
    return 1 if result.get('semanticFailure')else 0

def run(args):
    import psutil,torch
    directory=Path(args.output);directory.mkdir(parents=True,exist_ok=False)
    hardware={'cpuCount':os.cpu_count(),'ramTotalBytes':psutil.virtual_memory().total,'ramAvailableBytes':psutil.virtual_memory().available,
        'cudaAvailable':torch.cuda.is_available(),'torch':str(torch.__version__),'cuda':torch.version.cuda}
    if hardware['cudaAvailable']:
        prop=torch.cuda.get_device_properties(0);hardware.update(gpu=prop.name,gpuVRAMBytes=prop.total_memory)
    memory_bytes=min(args.memory_mib*1024**2 if args.memory_mib else hardware['ramAvailableBytes']*4//5,hardware['ramAvailableBytes']*4//5)
    report={'hardware':hardware,'expressionAndConditionBudgetBytes':args.expression_bytes,'memoryBudgetBytes':memory_bytes,
        'controllerSHA256':digest_file(Path(__file__)),
        'artifactBudgetSHA256':digest_file(Path(__file__).with_name('direct_sympy_artifact_budget.py')),
        'workersRequested':args.workers,'maxWorkersLive':0,'peakObservedRSSBytes':0,'runs':[],
        'state':'PREPARING','firstCoordinateValidated':False,'vectorEmitted':False,'fullVectorParity':False,'multipleTokenParity':False}
    def save():atomic(directory/'run.json',canonical(report))
    save()
    if args.require_gpu!='none' and (not hardware['cudaAvailable']or args.require_gpu not in hardware.get('gpu','')):
        report.update(state='BLOCKED',stop='Requested GPU is not present');save();return 1
    if args.require_gpu!='none':
        with (directory/'cuda-numerics.log').open('w')as log:
            cuda=subprocess.Popen([sys.executable,str(Path(__file__).with_name('direct_sympy_cuda_validation.py'))],stdout=log,stderr=subprocess.STDOUT)
            started=time.monotonic()
            while cuda.poll()is None:
                rss=psutil.Process().memory_info().rss+snapshot(cuda)
                report['peakObservedRSSBytes']=max(report['peakObservedRSSBytes'],rss)
                if rss>memory_bytes or time.monotonic()-started>args.worker_seconds:
                    cuda.kill();cuda.wait();report.update(state='FAILED',stop='CUDA numerical batch resource budget exceeded');save();return 1
                time.sleep(.2)
        report['cudaNumericsExitCode']=cuda.returncode
        if cuda.returncode:report.update(state='FAILED',stop='CUDA numerical parity failed');save();return 1
    from direct_sympy_artifact_budget import ArtifactBudget
    budget=ArtifactBudget(directory/'artifact-budget.json',args.expression_bytes,create=True)
    config=json.loads((Path(args.checkpoint)/'config.json').read_text());vocab=config['vocab_size']
    if not 0<=args.dimension<vocab:raise ValueError('First output coordinate outside checkpoint vocabulary')
    # Reserve Tuple wrapper bytes independently of worker claims. No vector
    # can exceed the common cap simply because each coordinate fits alone.
    budget.change('vector-syntax',7+2*vocab,exact=True)
    reservation=768*1024**2+16*args.expression_bytes
    report['workerReservationBytes']=reservation;report.update(state='RUNNING');save()
    def wave(dimensions,workers):
        todo=list(dimensions);active=[];completed=[]
        try:
            while todo or active:
                parent=psutil.Process().memory_info().rss
                rss=parent+sum(snapshot(t['process'])for t in active)
                report['peakObservedRSSBytes']=max(report['peakObservedRSSBytes'],rss)
                if rss>memory_bytes:raise ValueError('Measured RAM budget exceeded')
                committed=parent+sum(max(reservation,snapshot(t['process']))for t in active)
                while todo and len(active)<workers and committed+reservation<=memory_bytes:
                    dim=todo.pop(0);log=(directory/f'logit-{dim}.log').open('w')
                    command=[sys.executable,str(Path(__file__).resolve()),args.checkpoint,str(directory),'--worker',
                        '--dimension',str(dim),'--expression-bytes',str(args.expression_bytes),'--max-paths',str(args.max_paths),'--parity-cases',str(args.parity_cases)]
                    environment={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
                    process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=environment,start_new_session=True)
                    active.append({'dimension':dim,'process':process,'log':log,'started':time.monotonic()});committed+=reservation
                    report['maxWorkersLive']=max(report['maxWorkersLive'],len(active))
                if todo and not active:raise ValueError('RAM budget cannot admit one coordinate')
                for task in list(active):
                    process=task['process']
                    if process.poll()is None and time.monotonic()-task['started']>args.worker_seconds:raise ValueError('Coordinate wall-clock budget exceeded')
                    if process.poll()is None:continue
                    task['log'].close();active.remove(task)
                    if process.returncode:raise ValueError(f'Coordinate {task["dimension"]} failed; see its log')
                    item=json.loads((directory/f'logit-{task["dimension"]}.json').read_text());completed.append(item);report['runs'].append(item);save()
                    if not item['complete']or not item['parityVerified']:return completed
                save()
                if active:time.sleep(.2)
            return completed
        finally:
            for task in active:
                process=task['process']
                if process.poll()is None:
                    import signal
                    os.killpg(process.pid,signal.SIGKILL)
                process.wait();task['log'].close()
                budget.change(f'logit-{task["dimension"]}',0,exact=True)
                (directory/f'logit-{task["dimension"]}.expr').unlink(missing_ok=True)
    try:
        first=wave([args.dimension],1)
        if not first or not first[0]['complete']or not first[0]['parityVerified']:
            report.update(state='INCOMPLETE',stop='First complete coordinate was not admitted; remaining logits not dispatched');save();return 0
        report['firstCoordinateValidated']=True;save()
        others=wave([i for i in range(vocab)if i!=args.dimension],args.workers)
        if len(others)!=vocab-1 or any(not item['complete']or not item['parityVerified']for item in others):
            report.update(state='INCOMPLETE',stop='Output vector contains unfinished coordinates');save();return 0
        if report['controllerSHA256']!=digest_file(Path(__file__))or report['artifactBudgetSHA256']!=digest_file(Path(__file__).with_name('direct_sympy_artifact_budget.py')):
            raise ValueError('Compilation controller changed before vector emission')
        temporary=directory/'vector.expr.tmp';destination=directory/'vector.expr'
        with temporary.open('w')as stream:
            stream.write('Tuple(')
            for dimension in range(vocab):
                if dimension:stream.write(', ')
                with (directory/f'logit-{dimension}.expr').open()as source:
                    while block:=source.read(1024**2):stream.write(block)
            stream.write(')')
        if temporary.stat().st_size>args.expression_bytes:raise ValueError('Accumulated vector artifact budget exceeded')
        os.replace(temporary,destination)
        report.update(state='COMPLETED_POSITION_ZERO',vectorEmitted=True,vectorSHA256=digest_file(destination),vectorCharacters=destination.stat().st_size,
            parityScope='Per-coordinate position-zero native corpus; not multiple-token parity')
    except ValueError as error:report.update(state='INCOMPLETE',stop=str(error))
    finally:save()
    return 0

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('checkpoint');parser.add_argument('output')
    parser.add_argument('--worker',action='store_true');parser.add_argument('--dimension',type=int,default=2)
    parser.add_argument('--expression-bytes',type=int,default=DEFAULT_EXPRESSION_BYTES);parser.add_argument('--memory-mib',type=int,default=0)
    parser.add_argument('--workers',type=int,default=4);parser.add_argument('--worker-seconds',type=int,default=600)
    parser.add_argument('--max-paths',type=int,default=2048);parser.add_argument('--parity-cases',type=int,default=8192)
    parser.add_argument('--require-gpu',default='H100',choices=['H100','A100','none'])
    args=parser.parse_args()
    if min(args.expression_bytes,args.workers,args.worker_seconds,args.max_paths)<1 or args.parity_cases<1 or args.memory_mib<0:parser.error('Positive budgets and parity corpus required')
    return worker(args)if args.worker else run(args)
if __name__=='__main__':raise SystemExit(main())
