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
import os
import tempfile
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
    'direct_sympy_streaming_literals.py','direct_sympy_coherent_paths.py','direct_sympy_region_parity.py')


def encode(domains):return {name:[rank(d.minimum),rank(d.maximum)] for name,d in domains.items()}
def decode(record):return {name:interval(value(a),value(b)) for name,(a,b) in record.items()}
def cardinality(record):
    return math.prod(b-a+1+int(a<=0<=b) for a,b in record.values())


def next_region(tree,order='coverage',thresholds=None):
    """Visit the largest unfinished domain first, with reproducible ties."""
    if order not in ('coverage','lexical','update-cells'):raise ValueError('Unknown frontier order')
    pending=[key for key,node in tree.items() if node['status']=='pending']
    if not pending:return None
    if order=='update-cells':
        if not thresholds:raise ValueError('Update-cell ordering requires threshold geometry')
        def outside(key):
            domains=tree[key]['domains']
            return set(domains)==set(thresholds) and all(
                threshold is not None and (domains[name][1]<=-threshold or domains[name][0]>=threshold)
                for name,threshold in thresholds.items())
        # Visit all provably separated seed boxes before returning to the
        # normal lexical continuation. Scheduling creates no numeric proof.
        return min(pending,key=lambda key:(not outside(key),key))
    return min(pending,key=lambda key:(-cardinality(tree[key]['domains']),key)) if order=='coverage' else min(pending)


def update_threshold(bound):
    """First finite Half magnitude whose strict symmetric cell exceeds bound."""
    from direct_sympy_conversions import half_cell_radius_bound
    if bound is None or not math.isfinite(bound) or bound<0:return None
    if bound<2**-25:return rank(2**-24)
    for exponent in range(-14,16):
        base=2**exponent
        for candidate in (base,base+2**(exponent-10)):
            if bound<half_cell_radius_bound(candidate):return rank(candidate)
    return None


def seed_update_cells(model,root,max_regions):
    """Propose geometry from streaming checkpoint bounds, never outputs.

    Each region recompiles independently. A cut is only an opportunity to
    prove an update irrelevant; it is never an adopted numerical result.
    Preserve every central band and signed zero, bounding geometric fanout.
    """
    if type(max_regions) is not int or max_regions<1:raise ValueError('Positive seed region budget required')
    from direct_sympy_layer_bounds import layer
    bounds=[layer(model,f'model.layers.{i}.') for i in range(model.layers)]
    thresholds={name:None for name in root}
    if bounds and all(bound is not None for bound in bounds):
        for coordinate,name in enumerate(sorted(root,key=lambda key:int(key[1:]))):
            peak=max(bound[kind][coordinate] for bound in bounds for kind in ('attention','mlp'))
            thresholds[name]=update_threshold(peak)
    tree={'':{'domains':root,'status':'pending'}};leaves=[''];axes=[]
    for name,threshold in thresholds.items():
        if threshold is None or len(leaves)*3>max_regions:continue
        low,high=root[name]
        if not low<=-threshold<threshold<=high:continue
        remaining=[]
        for key in leaves:
            node=tree[key];domains=decode(node['domains'])
            _,negative,rest=split(domains,name,-threshold)
            node.update(status='split',axis=name,cut=-threshold)
            tree[key+'0']={'domains':encode(negative),'status':'pending'}
            middle=key+'1';tree[middle]={'domains':encode(rest),'status':'split','axis':name,'cut':threshold-1}
            _,central,positive=split(rest,name,threshold-1)
            tree[middle+'0']={'domains':encode(central),'status':'pending'}
            tree[middle+'1']={'domains':encode(positive),'status':'pending'}
            remaining.extend((key+'0',middle+'0',middle+'1'))
        leaves=remaining;axes.append(name)
    audit_tree(tree,root)
    return tree,{'thresholdRanks':thresholds,'seededAxes':axes,'leafRegions':len(leaves),
        'maxRegions':max_regions,'numericalResultsReused':False}


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
            _,left,right=split(decode(expected),node['axis'],node.get('cut'))
            pending.extend(((key+'0',encode(left)),(key+'1',encode(right))))
        elif node['status']=='complete':covered+=cardinality(expected)
        elif node['status']=='pending':unfinished+=cardinality(expected)
        else:raise ValueError('Invalid partition status')
    if seen!=set(tree) or covered+unfinished!=cardinality(root):raise ValueError('Incomplete partition coverage')
    return covered,unfinished


def replan(previous,identity,root):
    """Reuse only audited domain geometry, never expressions or proofs."""
    for field in ('version','dimension','position','checkpoint'):
        if previous['identity'].get(field)!=identity.get(field):
            raise ValueError('Partition geometry belongs to another checkpoint or coordinate')
    if previous['root']!=root:raise ValueError('Partition geometry root differs from admitted domain')
    audit_tree(previous['tree'],root)
    tree={}
    for key,node in previous['tree'].items():
        clean={'domains':{name:list(bounds) for name,bounds in node['domains'].items()},'status':'pending'}
        if node['status']=='split':
            clean.update(status='split',axis=node['axis'])
            if 'cut' in node:clean['cut']=node['cut']
        tree[key]=clean
    covered,unfinished=audit_tree(tree,root)
    if covered or unfinished!=cardinality(root):raise ValueError('Replanned geometry retained numerical results')
    return {'identity':identity,'root':root,'tree':tree,'attempts':[],
        'coveredInputPatterns':0,'unfinishedInputPatterns':unfinished,'totalInputPatterns':unfinished,
        'finalArtifactEmitted':False,'finalParity':False}


def region_pieces(directory,node):
    """Yield input-guarded flat pieces of one complete, integrity-bound region."""
    artifact=Path(directory)/node['artifact']['file']
    if digest_file(artifact)!=node['artifact']['sha256']:raise ValueError('Partition artifact integrity mismatch')
    text=artifact.read_text();expression=syntax(text)
    guards=[]
    for name,d in decode(node['domains']).items():
        guards.extend((f'{name} >= {repr(float(d.minimum))}',f'{name} <= {repr(float(d.maximum))}'))
    arms=expression.args if isinstance(expression,ast.Call) and expression.func.id=='Piecewise' else [ast.Tuple(elts=[expression,ast.Constant(value=True)],ctx=ast.Load())]
    for pair in arms:
        body,condition=map(ast.unparse,pair.elts)
        guard='And('+', '.join(guards+[condition])+')'
        sp=sympy.Symbol('PartitionBody',real=True)
        sympy.simplify(sympy.factor(sympy.Piecewise((sp,symbolic(syntax(guard))),evaluate=False)))
        piece='(('+body+'), '+guard+')'
        if 'CompileValue' in piece:raise ValueError('Unsubstituted compiler alias')
        yield piece


def region_character_charge(directory,node):
    # Two separator characters per arm conservatively exceed the exact final
    # syntax by one character. Outer input conditions belong to the budget too.
    return sum(len(piece)+2 for piece in region_pieces(directory,node))


def combine(directory,tree,path,max_characters):
    root=tree['']['domains'];covered,unfinished=audit_tree(tree,root)
    if unfinished:raise ValueError('Unfinished regions cannot be emitted as a final coordinate')
    path=Path(path);digest=hashlib.sha256();written=0;count=0
    with tempfile.NamedTemporaryFile(dir=path.parent,delete=False)as stream:
        temporary=Path(stream.name)
        def emit(text):
            nonlocal written
            written+=len(text)
            if written>max_characters:raise ValueError('Combined artifact character budget exceeded')
            block=text.encode();digest.update(block);stream.write(block)
        try:
            emit('Piecewise(')
            for key,node in sorted(tree.items()):
                if node['status']!='complete':continue
                for piece in region_pieces(directory,node):
                    if count:emit(', ')
                    emit(piece);count+=1
            emit(')');stream.flush();os.fsync(stream.fileno());os.replace(temporary,path)
        finally:temporary.unlink(missing_ok=True)
    return {'file':path.name,'characters':written,'sha256':digest.hexdigest(),
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
            if getattr(args,'repartition_from',None):raise ValueError('Geometry import requires a fresh destination state')
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
            source=getattr(args,'repartition_from',None)
            if source:
                source=Path(source)/'frontier.json';raw=source.read_bytes()
                state=replan(json.loads(raw),identity,root)
                state['geometryImport']={'sourceManifest':str(source.resolve()),'sha256':hashlib.sha256(raw).hexdigest(),
                    'numericalResultsReused':False,'leafRegions':sum(node['status']=='pending' for node in state['tree'].values())}
            else:
                state={'identity':identity,'root':root,'tree':{'':{'domains':root,'status':'pending'}},'attempts':[],
                    'finalArtifactEmitted':False,'finalParity':False}
                if getattr(args,'seed_update_cells',False):
                    with model:
                        state['tree'],state['updateCellGeometry']=seed_update_cells(model,root,getattr(args,'seed_max_regions',81))
        verify=bool(getattr(args,'verify_regions',False));parity_cases=getattr(args,'parity_cases',256)
        policy={'enabled':verify,'randomCases':parity_cases if verify else None}
        if manifest.exists() and state.get('regionVerificationPolicy',{'enabled':False,'randomCases':None})!=policy:
            raise ValueError('Incompatible region verification policy; no saved numerical result reused')
        state['regionVerificationPolicy']=policy
        charge=10
        for node in state['tree'].values():
            if node['status']!='complete':continue
            cost=region_character_charge(directory,node)
            if verify and (not node.get('parity') or node['parity'].get('mismatches')!=0
                or node['parity'].get('sha256')!=node['artifact']['sha256']):
                raise ValueError('Unverified saved region cannot be counted as verified coverage')
            node['artifact']['combinedCharacterCharge']=cost;charge+=cost
        if charge>args.total_characters:raise ValueError('Saved regions exceed accumulated artifact budget')
        state['accumulatedArtifactCharacters']=charge
        state.pop('stop',None)
        started=time.monotonic()
        order=getattr(args,'frontier_order',None) or ('update-cells' if 'updateCellGeometry' in state else 'lexical')
        state['frontierOrder']=order
        for _ in range(args.max_attempts):
            key=next_region(state['tree'],order,state.get('updateCellGeometry',{}).get('thresholdRanks'))
            if key is None:break
            node=state['tree'][key];domains=decode(node['domains'])
            file='region-'+hashlib.sha256(canonical(node['domains'])).hexdigest()+'.expr'
            # Candidate and worker leftovers stay isolated until integrity,
            # accumulated size and optional numerical parity all pass.
            with tempfile.TemporaryDirectory(dir=directory,prefix='.region-')as staging:
                candidate=Path(staging)/file
                result=compile_region(args.checkpoint,args.dimension,domains,candidate,max_characters=args.max_characters,
                    cas_characters=args.cas_characters,max_paths=args.max_paths,max_seconds=args.region_seconds)
                if result['complete']:
                    prospective={'domains':node['domains'],'artifact':{'file':file,'sha256':result['artifact']['sha256']}}
                    cost=region_character_charge(Path(staging),prospective)
                    if charge+cost>args.total_characters:
                        result.update(complete=False,stop='Accumulated expression/condition budget exceeded',compiledCandidate=True)
                        result['candidateArtifact']=result.pop('artifact');result['candidateArtifactRemoved']=True
                        state['stop']=result['stop']
                    else:
                        parity=None
                        if verify:
                            from direct_sympy_region_parity import verify_region
                            parity=verify_region(args.checkpoint,args.dimension,candidate,domains,random_cases=parity_cases)
                            if parity['mismatches'] or parity['sha256']!=prospective['artifact']['sha256']:
                                raise ValueError('Region parity failed; candidate not promoted')
                        os.replace(candidate,directory/file)
                        result['artifact']['path']=str(directory/file)
                        node['status']='complete';node['artifact']={'file':file,'combinedCharacterCharge':cost,
                            **{k:result['artifact'][k]for k in ('characters','sha256')}}
                        if parity is not None:
                            parity['artifact']=str(directory/file);node['parity']=parity;result['parity']=parity
                        charge+=cost;state['accumulatedArtifactCharacters']=charge
            state['attempts'].append({'region':key,'selectionOrder':order,**result})
            if not result['complete'] and not state.get('stop'):
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
            if state.get('stop'):break
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
    parser.add_argument('checkpoint');parser.add_argument('state')
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--resume',action='store_true')
    mode.add_argument('--repartition-from',help='Import audited input-domain splits; recompile every leaf with current sources')
    mode.add_argument('--seed-update-cells',action='store_true',help='Seed complete-domain cuts from checkpoint update bounds; compile every leaf afresh')
    parser.add_argument('--dimension',type=int,default=2);parser.add_argument('--max-attempts',type=int,default=8)
    parser.add_argument('--frontier-order',choices=('coverage','lexical','update-cells'))
    parser.add_argument('--seed-max-regions',type=int,default=81)
    parser.add_argument('--verify-regions',action='store_true',help='Require native checkpoint parity before admitting any region')
    parser.add_argument('--parity-cases',type=int,default=256)
    parser.add_argument('--region-seconds',type=int,default=30);parser.add_argument('--max-paths',type=int,default=64)
    parser.add_argument('--max-characters',type=int,default=1048576);parser.add_argument('--cas-characters',type=int,default=8388608)
    parser.add_argument('--total-characters',type=int,default=67108864);parser.add_argument('--min-values',type=int,default=32)
    args=parser.parse_args()
    if min(args.max_attempts,args.region_seconds,args.max_paths,args.max_characters,args.cas_characters,args.total_characters,args.min_values,args.seed_max_regions,args.parity_cases)<1:parser.error('Positive budgets required')
    return run(args)


if __name__=='__main__':raise SystemExit(main())
