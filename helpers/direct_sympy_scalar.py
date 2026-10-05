"""Pairwise composition with fully defined scalar locals instead of formula copies.

All effective expressions remain mathematical strings. JSON files only carry
compiler descriptors. Local definitions are emitted as straight-line Rust,
never interpreted as a graph at runtime. A branch stays inside its producer.
"""
from dataclasses import dataclass
import json
from pathlib import Path
import re
import tempfile
import time

from direct_sympy_parallel import _wave,resident_bytes,write_string


def dependencies(text):return set(re.findall(r'\bX[1-9][0-9]*\b',text))


def prepare_block(block,compiler,domains):
    definitions=[];outputs=[]
    for name,text in zip(block['locals'],block['outputs']):
        text=compiler.stabilize('('+text+')',domains)
        if text in block['inputs']:outputs.append(text)
        else:definitions.append([name,text]);outputs.append(name)
    return {'inputs':block['inputs'],'outputs':outputs,'definitions':definitions}


def merge_blocks(left,right,compiler,domains):
    if len(left['outputs'])!=len(right['inputs']):raise ValueError('Scalar interface mismatch')
    bindings=dict(zip(right['inputs'],left['outputs']))
    def replace(text):
        used=dependencies(text)
        for name,value in bindings.items():
            if name in used:text=compiler.substitute(text,name,value,domains)
        return compiler.stabilize('('+text+')',domains)
    return {'inputs':left['inputs'],'outputs':[replace(text) for text in right['outputs']],
        'definitions':left['definitions']+[[name,replace(text)] for name,text in right['definitions']]}


@dataclass(frozen=True)
class ScalarProgram:
    inputs:tuple[str,...]
    definitions:tuple[tuple[str,str],...]
    outputs:tuple[str,...]

    def validate(self):
        available=set(self.inputs)
        if len(available)!=len(self.inputs):raise ValueError('Duplicate original input')
        for name,text in self.definitions:
            if name in available or not re.fullmatch(r'X[1-9][0-9]*',name):raise ValueError('Duplicate/invalid scalar producer')
            if dependencies(text)-available:raise ValueError('Unresolved or future scalar producer')
            available.add(name)
        if any(dependencies(text)-available for text in self.outputs):raise ValueError('Unresolved output')
        return self

    def evaluate(self,values,functions):
        from direct_sympy_architecture_test import program
        if len(values)!=len(self.inputs):raise ValueError('Input shape mismatch')
        bindings=dict(zip(self.inputs,values));scope={'__builtins__':{},**functions}
        for name,text in self.definitions:bindings[name]=eval(program(text),scope,bindings)
        return tuple(eval(program(text),scope,bindings) for text in self.outputs)


def compose_scalar_operators(blocks,compiler,domains,budget,*,max_bytes=2*1024**3,on_wave=None):
    if not blocks or max_bytes<=0:raise ValueError('Nonempty architecture and positive budget required')
    seen=set();serial=max((int(name[1:]) for block in blocks for name in block.inputs),default=0)+1
    for i,block in enumerate(blocks):
        names=set(block.inputs)
        if len(names)!=len(block.inputs) or seen&names:raise ValueError('Scalar interfaces must be distinct')
        if any(not re.fullmatch(r'X[1-9][0-9]*',n) for n in names):raise ValueError('Invalid scalar interface')
        if i and (len(block.inputs)!=len(blocks[i-1].outputs) or names&domains.keys()):raise ValueError('Invalid scalar connection or inherited domain')
        if any(dependencies(text)-names for text in block.outputs):raise ValueError('Undeclared producer')
        seen.update(names)
    stats={'completedJobs':0,'completedBlocks':0,'completedPairMerges':0,'seededLanes':0,
        'maxWorkersAdmitted':0,'peakObservedRSSBytes':resident_bytes(),'compositionLevels':0}
    started=time.monotonic()
    with tempfile.TemporaryDirectory(prefix='llm-inner-scalar-pairs-') as directory:
        root=Path(directory)
        def charge():
            size=sum(path.stat().st_size for path in root.iterdir())
            stats['peakAccumulatedBytes']=max(stats.get('peakAccumulatedBytes',0),size)
            if size>max_bytes:raise ValueError('Accumulated scalar expression/condition budget exceeded')
            return size
        def wave(jobs):
            allowance=(max_bytes-charge())//len(jobs)
            _wave([(kind,paths,allowance,out) for kind,paths,out in jobs],budget,compiler,domains,stats,on_progress=on_wave)
            charge()
        paths=[];jobs=[]
        for i,block in enumerate(blocks):
            source=str(root/f'source-{i}');out=str(root/f'prepared-{i}')
            names=[f'X{serial+j}' for j in range(len(block.outputs))];serial+=len(names)
            write_string(source,json.dumps({'inputs':block.inputs,'outputs':block.outputs,'locals':names}))
            paths.append(out);jobs.append(('scalar-prepare',[source],out))
        wave(jobs)
        for _,sources,_ in jobs:Path(sources[0]).unlink()
        while len(paths)>1:
            jobs=[];following=[]
            for i in range(0,len(paths),2):
                if i+1==len(paths):following.append(paths[i]);continue
                out=str(root/f'merged-{stats["compositionLevels"]}-{i}')
                jobs.append(('scalar-merge',paths[i:i+2],out));following.append(out)
            wave(jobs)
            for _,sources,_ in jobs:
                for path in sources:Path(path).unlink()
            paths=following;stats['compositionLevels']+=1
        result=json.loads(Path(paths[0]).read_text())
    value=ScalarProgram(tuple(result['inputs']),tuple(tuple(row) for row in result['definitions']),tuple(result['outputs'])).validate()
    # Remove definitions made unreachable by a constant/condition simplification.
    needed=set().union(*(dependencies(text) for text in value.outputs));kept=[]
    for name,text in reversed(value.definitions):
        if name in needed:kept.append((name,text));needed.update(dependencies(text))
    value=ScalarProgram(value.inputs,tuple(reversed(kept)),value.outputs).validate()
    stats.update(seconds=time.monotonic()-started,scalarProducers=len(value.definitions),
        scalarCharacters=sum(len(text) for _,text in value.definitions)+sum(map(len,value.outputs)),
        factorSimplifyEvents=len(compiler.events),substitutions=len(compiler.substitution_events))
    return value,stats
