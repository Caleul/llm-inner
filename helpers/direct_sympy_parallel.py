"""CPU block composition of the original F32 folds, never reassociation.

Blocks are mathematical strings with one compiler-only Xn continuation.
Composing right(left(Xn)) preserves the original ordered fold. Strings are
passed through files, not pickled into a process queue; only descriptors cross
process boundaries. Every substitution still uses StringCompiler's CAS gate.
"""
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from dataclasses import dataclass
import multiprocessing
import os
from pathlib import Path
import resource
import tempfile
import time

_BASE=None


def resident_bytes(pid=None):
    pid=os.getpid() if pid is None else pid
    try:
        for line in Path(f'/proc/{pid}/status').read_text().splitlines():
            if line.startswith('VmRSS:'):return int(line.split()[1])*1024
        return 0
    except FileNotFoundError:
        if pid!=os.getpid():return 0
        value=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return value if os.uname().sysname=='Darwin' else value*1024


def write_string(path,text):
    with Path(path).open('w',encoding='utf-8') as stream:
        for offset in range(0,len(text),1024*1024):stream.write(text[offset:offset+1024*1024])


@dataclass(frozen=True)
class ParallelBudget:
    workers:int=2
    memory_bytes:int=2*1024**3
    block_size:int=64

    def __post_init__(self):
        if any(type(v) is not int or v<=0 for v in (self.workers,self.memory_bytes,self.block_size)):
            raise ValueError('Positive integer parallel budgets required')


def _job(job):
    compiler,domains=_BASE
    before=len(compiler.events);started=time.monotonic()
    kind,paths,hole,output=job
    if kind=='block':
        expression=hole
        for path in paths:
            term=Path(path).read_text()
            expression=compiler.substitute('R32(X999999998 + X999999999)','X999999998',expression,domains)
            expression=compiler.substitute(expression,'X999999999',term,domains)
    elif kind=='merge':
        left,right=(Path(path).read_text() for path in paths)
        expression=compiler.substitute(right,hole,left,domains)
    elif kind=='seed':expression=compiler.substitute(Path(paths[0]).read_text(),hole,'0.0',domains)
    else:raise ValueError('Unknown composition job')
    write_string(output,expression)
    return {'path':output,'characters':len(expression),'events':compiler.events[before:],
        'kind':kind,'seconds':time.monotonic()-started,'pid':os.getpid(),'rssBytes':resident_bytes()}


def _wave(jobs,budget,compiler,domains,stats):
    global _BASE
    if not jobs:return []
    parent=resident_bytes()
    # Reserve for decoded inputs, replacements, restored CAS strings and
    # interpreter/CAS overhead. RSS monitoring also catches underestimated
    # work. Fork shares immutable prior proofs; conservative accounting may
    # count those pages again, but never assumes they are free.
    reservations=[384*1024**2+8*sum(Path(p).stat().st_size for p in job[1]) for job in jobs]
    worst=max(reservations)
    capacity=(budget.memory_bytes-parent)//(parent+worst)
    if capacity<1:raise ValueError(f'Parallel memory budget cannot admit one block: parent={parent} reservation={worst} limit={budget.memory_bytes}')
    workers=min(budget.workers,len(jobs),capacity)
    stats['maxWorkersAdmitted']=max(stats['maxWorkersAdmitted'],workers)
    _BASE=(compiler,domains);results=[None]*len(jobs)
    context=multiprocessing.get_context('fork')
    executor=ProcessPoolExecutor(max_workers=workers,mp_context=context)
    try:
        pending={};next_job=0
        while next_job<len(jobs) or pending:
            while next_job<len(jobs) and len(pending)<workers:
                pending[executor.submit(_job,jobs[next_job])]=next_job;next_job+=1
            processes=list(executor._processes.values())
            rss=resident_bytes()+sum(resident_bytes(p.pid) for p in processes)
            stats['peakObservedRSSBytes']=max(stats['peakObservedRSSBytes'],rss)
            if rss>budget.memory_bytes:raise ValueError(f'Parallel RSS exceeded memory budget: rss={rss} limit={budget.memory_bytes}')
            completed,_=wait(pending,timeout=.1,return_when=FIRST_COMPLETED)
            for future in completed:
                index=pending.pop(future);result=future.result();results[index]=result
                stats['completedJobs']+=1
                stats['completedBlocks']+=result['kind']=='block'
                stats['completedPairMerges']+=result['kind']=='merge'
                stats['seededLanes']+=result['kind']=='seed'
        # Merge evidence in original descriptor order, independently of the
        # workers' completion order. No child compiler cache is published.
        for result in results:compiler.events.extend(result['events'])
        return results
    finally:
        for process in list(executor._processes.values()):
            if process.is_alive():process.terminate()
        executor.shutdown(wait=True,cancel_futures=True);_BASE=None


def parallel_lanes(terms,compiler,domains,budget):
    """Return the same four lane folds used by the sequential adapter.

    Pairwise merging composes continuation contexts, not float additions.
    Final lane pairing remains the caller's original (0+1)+(2+3) tree.
    The continuation is eliminated before a result leaves this function.
    """
    if 'fork' not in multiprocessing.get_all_start_methods():
        raise ValueError('Parallel CAS requires a fork-capable CPU process')
    hole='X'+str(max((int(name[1:]) for name in domains),default=0)+1)
    stats={'completedJobs':0,'completedBlocks':0,'completedPairMerges':0,
        'seededLanes':0,'maxWorkersAdmitted':0,'peakObservedRSSBytes':resident_bytes()}
    started=time.monotonic()
    with tempfile.TemporaryDirectory(prefix='llm-inner-cas-blocks-') as directory:
        root=Path(directory);term_paths=[]
        for index,term in enumerate(terms):
            path=root/f'term-{index}.expr';write_string(path,term);term_paths.append(str(path))
        jobs=[];lanes=[[] for _ in range(4)]
        for lane in range(4):
            paths=term_paths[lane::4]
            for begin in range(0,len(paths),budget.block_size):
                output=str(root/f'block-{len(jobs)}.expr')
                lanes[lane].append(output)
                jobs.append(('block',paths[begin:begin+budget.block_size],hole,output))
        _wave(jobs,budget,compiler,domains,stats)
        level=0
        while any(len(lane)>1 for lane in lanes):
            jobs=[];combined=[]
            for lane_index,lane in enumerate(lanes):
                outputs=[]
                for index in range(0,len(lane),2):
                    if index+1==len(lane):outputs.append(lane[index]);continue
                    output=str(root/f'merge-{level}-{lane_index}-{index}.expr')
                    jobs.append(('merge',lane[index:index+2],hole,output));outputs.append(output)
                combined.append(outputs)
            _wave(jobs,budget,compiler,domains,stats);lanes=combined;level+=1
        jobs=[];final=[]
        for lane_index,lane in enumerate(lanes):
            if not lane:final.append(None);continue
            output=str(root/f'lane-{lane_index}.expr');final.append(output)
            jobs.append(('seed',lane,hole,output))
        _wave(jobs,budget,compiler,domains,stats)
        values=[Path(path).read_text() if path else '0.0' for path in final]
    stats['seconds']=time.monotonic()-started
    return values,stats
