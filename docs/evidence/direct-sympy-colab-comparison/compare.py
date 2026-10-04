from pathlib import Path
import json,os,subprocess,sys,time
root=Path('/content/llm-inner-input-partitions');os.chdir(root)
sys.path.insert(0,str(root/'helpers'))
os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
if len(sys.argv)>1:
 import torch
 from direct_sympy_checkpoint import CheckpointStrings
 from direct_sympy_strings import StringCompiler
 from direct_sympy_input_partitions import interval
 from direct_sympy_streaming_literals import streaming_literals
 from direct_sympy_coherent_paths import CoherentPaths
 from direct_sympy_parallel import ParallelBudget
 torch.set_num_threads(1);workers=int(sys.argv[1]);start=time.monotonic()
 checkpoint=root/'docs/evidence/direct-sympy-test-checkpoint'
 config=json.loads((checkpoint/'config.json').read_text());domains={f'X{i+1}':interval(32,65504) for i in range(config['hidden_size'])}
 budget=None if workers==1 else ParallelBudget(workers=workers,memory_bytes=8*1024**3,block_size=1)
 with CheckpointStrings(checkpoint,StringCompiler(max_characters=8388608),input_domains=domains,parallel_budget=budget) as model:
  with streaming_literals(model) as registry:
   expression=model.coordinate(2);plan=CoherentPaths(registry,max_paths=64)
   result=plan.write(root/'results'/f'comparison-{workers}.expr',expression,max_characters=1048576)
   row={'workers':workers,'compileSeconds':time.monotonic()-start,'artifact':result,'producers':len(model.memo),'parallelEvents':model.parallel_events,'scope':'position-zero coordinate, input range [32,65504] for every dimension'}
 (root/'results'/f'comparison-{workers}.json').write_text(json.dumps(row,indent=2))
else:
 import psutil
 report={'state':'RUNNING','runs':[],'fullCoordinateParity':False}
 def save():
  p=root/'comparison-job.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(report,indent=2));os.replace(t,p)
 save()
 for workers in (1,2):
  started=time.monotonic();peak=0
  with (root/'results'/f'comparison-{workers}.log').open('w') as log:
   process=subprocess.Popen([sys.executable,__file__,str(workers)],stdout=log,stderr=subprocess.STDOUT)
   while process.poll() is None:
    try:
     parent=psutil.Process(process.pid);processes=[parent]+parent.children(recursive=True)
     rss=sum(p.memory_info().rss for p in processes if p.is_running());peak=max(peak,rss)
    except (psutil.NoSuchProcess,psutil.ZombieProcess):pass
    time.sleep(.05)
  row={'workers':workers,'wallSeconds':time.monotonic()-started,'sampledAggregateRSSBytes':peak,'exitCode':process.returncode}
  if process.returncode:report.update(state='FAILED',error='Coordinate comparison failed');save();raise SystemExit(process.returncode)
  row.update(json.loads((root/'results'/f'comparison-{workers}.json').read_text()));report['runs'].append(row);save()
 report['identicalArtifacts']=report['runs'][0]['artifact']['sha256']==report['runs'][1]['artifact']['sha256']
 report['state']='COMPLETED' if report['identicalArtifacts'] else 'FAILED';save()
