"""Adaptive, complete-domain compilation; incomplete partitions are never final.

The saved tree and strings are compiler state only. Final dispatch expands
all weights/dependencies inside each flat arm, without runtime aliases.
"""
import argparse
import ast
import fcntl
import hashlib
import json
import math
import platform
from pathlib import Path
import subprocess
import sys
import time

import sympy
import torch
from direct_sympy_checkpoint import CheckpointStrings
from direct_sympy_coherent_paths import CoherentPaths
from direct_sympy_input_partitions import interval,rank,value,split
from direct_sympy_savepoints import SOURCES,atomic,canonical,digest_file
from direct_sympy_streaming_literals import streaming_literals
from direct_sympy_strings import StringCompiler,syntax,symbolic

EXTRA_SOURCES=('direct_sympy_partition_run.py','direct_sympy_input_partitions.py',
    'direct_sympy_streaming_literals.py','direct_sympy_coherent_paths.py')


def encode(domains):return {name:[rank(d.minimum),rank(d.maximum)] for name,d in domains.items()}
def decode(record):return {name:interval(value(a),value(b)) for name,(a,b) in record.items()}
def cardinality(record):
    return math.prod(b-a+1+int(a<=0<=b) for a,b in record.values())


def _compile_region(checkpoint,dimension,domains,path,*,max_characters,cas_characters,max_paths):
    started=time.monotonic();report={'complete':False,'inputDomains':encode(domains)}
    try:
        with CheckpointStrings(checkpoint,StringCompiler(max_characters=cas_characters),input_domains=domains) as model:
            with streaming_literals(model) as registry:
                expression=model.coordinate(dimension)
                report.update(producers=len(model.memo),elidedUpdates=len(model.elided_updates),
                    logicalCharacters=registry.size(expression),storedCharacters=sum(map(len,registry.definitions)))
                plan=CoherentPaths(registry,max_paths=max_paths)
                report['artifact']=plan.write(path,expression,max_characters=max_characters)
                report.update(complete=True,paths=plan.stats)
    except ValueError as error:
        # Resource exhaustion requests subdivision. Semantic failures must
        # surface instead of being disguised as another partition.
        if 'budget' not in str(error).lower():raise
        report['stop']=str(error)
    finally:
        report['seconds']=time.monotonic()-started
    return report


def compile_region(checkpoint,dimension,domains,path,*,max_characters,cas_characters,max_paths,max_seconds):
    # A Python signal raised inside a weakref/GC callback can be swallowed.
    # Bound the worker externally instead; timed-out proof tables are never
    # reused, and subprocess.run kills/reaps the worker before returning.
    started=time.monotonic()
    arguments=[sys.executable,str(Path(__file__).resolve()),'--worker',str(checkpoint),str(path),
        str(dimension),str(max_characters),str(cas_characters),str(max_paths),json.dumps(encode(domains))]
    try:process=subprocess.run(arguments,capture_output=True,text=True,timeout=max_seconds)
    except subprocess.TimeoutExpired:
        return {'complete':False,'inputDomains':encode(domains),'stop':'Region worker wall-clock budget exceeded',
            'seconds':time.monotonic()-started}
    if process.returncode:raise ValueError('Region worker semantic failure: '+process.stderr[-2000:])
    report=json.loads(process.stdout);report['workerSeconds']=time.monotonic()-started
    return report


def audit_tree(tree,root):
    """Prove saved leaf coverage by replaying every exact binary split."""
    pending=[('',root)];seen=set();covered=0;unfinished=0
    while pending:
        key,expected=pending.pop()
        node=tree.get(key)
        if node is None or node['domains']!=expected:raise ValueError('Partition tree coverage mismatch')
        seen.add(key)
        if node['status']=='split':
            _,left,right=split(decode(expected),node['axis'])
            pending.extend(((key+'0',encode(left)),(key+'1',encode(right))))
        elif node['status']=='complete':covered+=cardinality(expected)
        elif node['status']=='pending':unfinished+=cardinality(expected)
        else:raise ValueError('Invalid partition status')
    if seen!=set(tree) or covered+unfinished!=cardinality(root):raise ValueError('Incomplete partition coverage')
    return covered,unfinished


def combine(directory,tree,path,max_characters):
    root=tree['']['domains'];covered,unfinished=audit_tree(tree,root)
    if unfinished:raise ValueError('Unfinished regions cannot be emitted as a final coordinate')
    pieces=[];characters=10
    for key,node in sorted(tree.items()):
        if node['status']!='complete':continue
        artifact=directory/node['artifact']['file']
        if digest_file(artifact)!=node['artifact']['sha256']:raise ValueError('Partition artifact integrity mismatch')
        text=artifact.read_text();expression=syntax(text)
        # Every outer condition precedes leaf-local conditions, preserving
        # lazy evaluation and the exact domain of each numeric proof.
        guards=[]
        for name,d in decode(node['domains']).items():
            guards.extend((f'{name} >= {repr(float(d.minimum))}',f'{name} <= {repr(float(d.maximum))}'))
        arms=expression.args if isinstance(expression,ast.Call) and expression.func.id=='Piecewise' else [ast.Tuple(elts=[expression,ast.Constant(value=True)],ctx=ast.Load())]
        for pair in arms:
            body,condition=map(ast.unparse,pair.elts)
            guard='And('+', '.join(guards+[condition])+')'
            # The combined envelope also passes factor then simplify. Bodies
            # remain opaque: their own substitutions already stabilized.
            sp=sympy.Symbol('PartitionBody',real=True)
            sympy.simplify(sympy.factor(sympy.Piecewise((sp,symbolic(syntax(guard))),evaluate=False)))
            piece='(('+body+'), '+guard+')';characters+=len(piece)+2
            if characters>max_characters:raise ValueError('Combined artifact character budget exceeded')
            pieces.append(piece)
    text='Piecewise('+', '.join(pieces)+')'
    if 'CompileValue' in text:raise ValueError('Unsubstituted compiler alias')
    atomic(path,text.encode())
    return {'file':Path(path).name,'characters':len(text),'sha256':digest_file(path),
        'coveredInputPatterns':covered,'complete':True,'parityVerified':False}


def run(args):
    directory=Path(args.state);directory.mkdir(parents=True,exist_ok=True)
    lock=(directory/'writer.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        identity={'version':1,'dimension':args.dimension,'position':0,'python':sys.version,'sympy':sympy.__version__,
            'platform':platform.platform(),'byteOrder':sys.byteorder,
            'referenceBackend':{'name':'torch CPU','version':str(torch.__version__),'gitVersion':torch.version.git_version,'capability':torch.backends.cpu.get_cpu_capability()},
            'checkpoint':{p.name:digest_file(p) for p in [Path(args.checkpoint)/'config.json',*sorted(Path(args.checkpoint).glob('*.safetensors'))]},
            'sources':{name:digest_file(Path(__file__).parent/name) for name in sorted(set(SOURCES+EXTRA_SOURCES))}}
        manifest=directory/'frontier.json'
        model=CheckpointStrings(args.checkpoint,StringCompiler())
        root=encode(model.domains)
        if manifest.exists():
            if not args.resume:raise ValueError('Existing partition state requires --resume')
            state=json.loads(manifest.read_text())
            if state['identity']!=identity:raise ValueError('Incompatible partition state; no saved expression reused')
            if state['root']!=root:raise ValueError('Partition root differs from the full admitted input domain')
            audit_tree(state['tree'],state['root'])
            for node in state['tree'].values():
                if node['status']=='complete' and digest_file(directory/node['artifact']['file'])!=node['artifact']['sha256']:
                    raise ValueError('Saved partition artifact integrity mismatch')
        else:
            if args.resume:raise ValueError('No partition state to resume')
            state={'identity':identity,'root':root,'tree':{'':{'domains':root,'status':'pending'}},'attempts':[],
                'finalArtifactEmitted':False,'finalParity':False}
        started=time.monotonic()
        for _ in range(args.max_attempts):
            pending=sorted(key for key,node in state['tree'].items() if node['status']=='pending')
            if not pending:break
            key=pending[0];node=state['tree'][key];domains=decode(node['domains'])
            file='region-'+hashlib.sha256(canonical(node['domains'])).hexdigest()+'.expr'
            result=compile_region(args.checkpoint,args.dimension,domains,directory/file,max_characters=args.max_characters,
                cas_characters=args.cas_characters,max_paths=args.max_paths,max_seconds=args.region_seconds)
            state['attempts'].append({'region':key,**result})
            if result['complete']:
                node['status']='complete';node['artifact']={'file':file,**{k:result['artifact'][k] for k in ('characters','sha256')}}
            else:
                choices=[name for name,d in domains.items() if rank(d.maximum)-rank(d.minimum)+1>2*args.min_values]
                if not choices:
                    state['stop']='Cannot subdivide further without enumerating individual inputs';break
                # The smallest center controls whether a rounded residual
                # update is visible. Avoid repeatedly splitting an already
                # large center while its small sibling still needs expansion.
                def priority(name):
                    d=domains[name]
                    minimum=d.minimum if d.minimum>0 else -d.maximum if d.maximum<0 else 0
                    return minimum,-(rank(d.maximum)-rank(d.minimum)),name
                axis=min(choices,key=priority)
                _,left,right=split(domains,axis);node.update(status='split',axis=axis)
                state['tree'][key+'0']={'domains':encode(left),'status':'pending'}
                state['tree'][key+'1']={'domains':encode(right),'status':'pending'}
            covered,unfinished=audit_tree(state['tree'],state['root'])
            state.update(coveredInputPatterns=covered,unfinishedInputPatterns=unfinished,totalInputPatterns=cardinality(state['root']))
            atomic(manifest,canonical(state))
            print(json.dumps({'region':key,'complete':result['complete'],'coveredInputPatterns':covered,'unfinishedInputPatterns':unfinished,'attemptSeconds':result['seconds']}),flush=True)
        covered,unfinished=audit_tree(state['tree'],state['root'])
        state.update(coveredInputPatterns=covered,unfinishedInputPatterns=unfinished,totalInputPatterns=cardinality(state['root']),runSeconds=time.monotonic()-started)
        if not unfinished:
            state['finalArtifact']=combine(directory,state['tree'],directory/'coordinate.expr',args.total_characters)
            state['finalArtifactEmitted']=True
        atomic(manifest,canonical(state))
        return 0 if state['finalArtifactEmitted'] else 1
    finally:lock.close()


def main():
    if len(sys.argv)>1 and sys.argv[1]=='--worker':
        if len(sys.argv)!=9:raise ValueError('Invalid isolated region worker arguments')
        _,_,checkpoint,path,dimension,characters,cas,paths,domains=sys.argv
        report=_compile_region(checkpoint,int(dimension),decode(json.loads(domains)),path,
            max_characters=int(characters),cas_characters=int(cas),max_paths=int(paths))
        print(json.dumps(report),flush=True)
        return 0
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint');parser.add_argument('state');parser.add_argument('--resume',action='store_true')
    parser.add_argument('--dimension',type=int,default=2);parser.add_argument('--max-attempts',type=int,default=8)
    parser.add_argument('--region-seconds',type=int,default=30);parser.add_argument('--max-paths',type=int,default=64)
    parser.add_argument('--max-characters',type=int,default=1048576);parser.add_argument('--cas-characters',type=int,default=8388608)
    parser.add_argument('--total-characters',type=int,default=67108864);parser.add_argument('--min-values',type=int,default=32)
    args=parser.parse_args()
    if min(args.max_attempts,args.region_seconds,args.max_paths,args.max_characters,args.cas_characters,args.total_characters,args.min_values)<1:parser.error('Positive budgets required')
    return run(args)


if __name__=='__main__':raise SystemExit(main())
