"""Colab coordinator: last-position-only compilation, bounded remote workers.

This is orchestration/evidence code, never the generated model runtime.
The original CPU arm64 reference and target snapshot arrive with the bundle.
"""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import psutil

ROOT=Path(__file__).resolve().parent.parent
OUTPUT=ROOT/'colab-results'


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def run_unit(command,path,*,seconds=180,memory_bytes=6*1024**3):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    start=time.monotonic();peak=0;stop=None
    with path.with_suffix('.log').open('w') as log:
        process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        observed=psutil.Process(process.pid)
        while process.poll() is None:
            try:
                rss=observed.memory_info().rss
                for child in observed.children(recursive=True):
                    try:rss+=child.memory_info().rss
                    except psutil.Error:pass
                peak=max(peak,rss)
            except psutil.Error:pass
            if peak>memory_bytes:stop='RAM budget exceeded'
            elif time.monotonic()-start>seconds:stop='Remote unit deadline exceeded'
            if stop:
                os.killpg(process.pid,signal.SIGKILL);break
            time.sleep(.1)
        code=process.wait()
    result=json.loads(path.read_text()) if path.exists() else {}
    result.update({'exitCode':code,'wallSeconds':time.monotonic()-start,'peakObservedAggregateRSSBytes':peak,'coordinatorStop':stop})
    save(path,result);return result


def main():
    os.chdir(ROOT);OUTPUT.mkdir(exist_ok=True)
    import torch
    torch.set_num_threads(1)
    resources={'device':torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        'cuda':torch.version.cuda,'cpuCount':os.cpu_count(),'RAMBytes':psutil.virtual_memory().total,
        'GPUBytes':torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0,
        'expressionConditionBudgetBytes':512*1024**2,'scope':'Four logits of the final position only',
        'sourceTarget':'pytorch-2.12.1-cpu-arm64','hostArchitecture':os.uname().machine}
    save(OUTPUT/'resources.json',resources)
    checkpoint=ROOT/'docs/evidence/direct-sympy-test-checkpoint';reference=ROOT/'source-reference.json'
    # Numerical batch preparation/verification on CUDA, factor/simplify on CPU.
    from direct_sympy_architecture import architecture_plan,prune_plan,last_position_indices
    from direct_sympy_architecture_cuda import validate
    config=json.loads((checkpoint/'config.json').read_text());lengths=range(1,config['max_position_embeddings']+1);vocab=config['vocab_size']
    plans=[]
    for length in lengths:
        plan=architecture_plan(checkpoint,length)
        blocks=prune_plan(plan,last_position_indices(plan))
        plans.append({'length':length,'vocab':plan.vocab,'outputIndices':last_position_indices(plan),
            'blocks':[{'inputs':b.inputs,'outputs':b.outputs} for b in blocks]})
    cuda=validate(plans,json.loads(reference.read_text())['cases']);save(OUTPUT/'cuda.json',cuda)
    # Four independent logits run concurrently, and a parent RSS watchdog
    # bounds each tree. Every job preserves original reductions/rounding.
    jobs=[(length,coordinate) for length in lengths for coordinate in range(vocab)]
    def closed(job):
        length,coordinate=job;path=OUTPUT/'closed-json'/f'length-{length}-logit-{coordinate}.json'
        result=run_unit(['node','--expose-gc','--max-old-space-size=4096','helpers/direct_json_final_position_probe.mjs',
            str(checkpoint),'target-discovery.json',str(reference),str(length),str(path),str(coordinate)],path,
            seconds=180,memory_bytes=6*1024**3)
        save(OUTPUT/'latest-closed-progress.json',{'length':length,'coordinate':coordinate,'result':result})
        return result
    with ThreadPoolExecutor(max_workers=4) as executor:
        closed_results=list(executor.map(closed,jobs))
    save(OUTPUT/'closed-json-summary.json',{'results':closed_results,'finalArtifactParity':False})
    # Controlled sequential/parallel comparison for the smallest multi-token
    # case, followed by the corrected batch; identical target and budgets.
    string_results=[]
    for length,workers in [(2,1),*[(n,4) for n in lengths]]:
        directory=OUTPUT/'sympy'/f'length-{length}-workers-{workers}'
        path=directory/'result.json'
        result=run_unit([sys.executable,'helpers/direct_sympy_architecture_run.py',str(checkpoint),str(directory),
            '--unit-length',str(length),'--reference',str(reference),'--workers',str(workers),
            '--memory-mib','24576','--seconds-per-length','120','--lower','--lower-seconds','30'],path,
            seconds=255,memory_bytes=24*1024**3)
        string_results.append(result);save(OUTPUT/'sympy-summary.json',{'results':string_results,'finalArtifactParity':False})
    save(OUTPUT/'done.json',{'closedUnits':len(closed_results),'sympyUnits':len(string_results),'finalArtifactParity':False})


if __name__=='__main__':main()
