"""Compile-only vector operator composition, preserving original IEEE order.

The file descriptors use JSON as transport metadata, not as the mathematical
representation. Each output is a SymPy-compatible mathematical string. No
interface identifier or descriptor is part of the final expression.
"""
from dataclasses import dataclass
import json
from pathlib import Path
import re
import tempfile
import time

from direct_sympy_parallel import _wave,resident_bytes,write_string


@dataclass(frozen=True)
class OperatorBlock:
    inputs:tuple[str,...]
    outputs:tuple[str,...]


def compose_operators(blocks,compiler,domains,budget,*,max_accumulated_characters=512*1024**2,on_wave=None):
    """Prepare independently; compose adjacent pairs until one block remains.

    Each stage owns disjoint Xn interface names. Only original input domains
    may be supplied: an internal interface must not inherit an input bound.
    This intentionally sacrifices unproved local algebra rather than moving
    a certificate across a numerical boundary. Conditions stay in their arms.
    """
    if not blocks:raise ValueError('At least one operator required')
    if type(max_accumulated_characters) is not int or max_accumulated_characters<=0:
        raise ValueError('Positive accumulated expression budget required')
    seen=set()
    for index,block in enumerate(blocks):
        names=set(block.inputs)
        if len(names)!=len(block.inputs) or any(not re.fullmatch(r'X[1-9][0-9]*',n) for n in names):
            raise ValueError('Invalid operator interface')
        if seen&names:raise ValueError('Operator interfaces must be disjoint')
        if index and names&domains.keys():raise ValueError('Internal interface cannot inherit input domains')
        seen.update(names)
        if not block.outputs:raise ValueError('Empty operator output')
        if index and len(block.inputs)!=len(blocks[index-1].outputs):raise ValueError('Operator interface dimension mismatch')
        for expression in block.outputs:
            if set(re.findall(r'\bX[1-9][0-9]*\b',expression))-names:
                raise ValueError('Operator output contains an undeclared dependency')
    stats={'completedJobs':0,'completedBlocks':0,'completedPairMerges':0,'seededLanes':0,
        'maxWorkersAdmitted':0,'peakObservedRSSBytes':resident_bytes(),'compositionLevels':0}
    start=time.monotonic()
    with tempfile.TemporaryDirectory(prefix='llm-inner-operators-') as directory:
        root=Path(directory)
        def charge():
            size=sum(p.stat().st_size for p in root.iterdir())
            stats['peakAccumulatedBytes']=max(stats.get('peakAccumulatedBytes',0),size)
            if size>max_accumulated_characters:raise ValueError('Accumulated operator expression/condition budget exceeded')
        def wave(jobs):
            charge()
            remaining=max_accumulated_characters-sum(p.stat().st_size for p in root.iterdir())
            allowance=remaining//len(jobs)
            reserved=[(kind,sources,allowance,output) for kind,sources,_,output in jobs]
            _wave(reserved,budget,compiler,domains,stats,on_progress=on_wave);charge()
            if on_wave is not None:on_wave(dict(stats))
        paths=[];jobs=[]
        for index,block in enumerate(blocks):
            source=str(root/f'source-{index}');output=str(root/f'prepared-{index}')
            write_string(source,json.dumps({'inputs':block.inputs,'outputs':block.outputs},separators=(',',':')))
            paths.append(output);jobs.append(('operator-prepare',[source],None,output))
        wave(jobs)
        for job in jobs:Path(job[1][0]).unlink()
        while len(paths)>1:
            level=stats['compositionLevels'];jobs=[];next_paths=[]
            for index in range(0,len(paths),2):
                if index+1==len(paths):next_paths.append(paths[index]);continue
                output=str(root/f'merged-{level}-{index}')
                jobs.append(('operator-merge',paths[index:index+2],None,output));next_paths.append(output)
            wave(jobs)
            for job in jobs:
                for path in job[1]:Path(path).unlink()
            paths=next_paths;stats['compositionLevels']+=1
        result=json.loads(Path(paths[0]).read_text())
    allowed=set(blocks[0].inputs)
    if any(set(re.findall(r'\bX[1-9][0-9]*\b',e))-allowed for e in result['outputs']):
        raise ValueError('Unresolved operator interface')
    stats['seconds']=time.monotonic()-start
    return tuple(result['outputs']),stats
