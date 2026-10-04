"""Compile a broad region afresh and cover only its pending intersections.

This is compiler-state geometry tooling. The numerical compiler is unchanged;
its complete source/backend/checkpoint identity is checked before any reuse.
Shared files are compilation artifacts, never runtime caches or responses.
"""
import argparse
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import platform
import sys
import tempfile

import sympy
import torch
from direct_sympy_checkpoint import CheckpointStrings
from direct_sympy_partition_run import EXTRA_SOURCES,audit_tree,cardinality,compile_region,decode,encode
from direct_sympy_input_partitions import split
from direct_sympy_region_parity import verify_region
from direct_sympy_savepoints import SOURCES,atomic,canonical,digest_file
from direct_sympy_strings import StringCompiler


def live_identity(checkpoint,dimension):
    # Match the controller's full numerical identity. This tool does not
    # change numerical compilation or relax source/backend compatibility.
    return {'version':1,'dimension':dimension,'position':0,'python':sys.version,'sympy':sympy.__version__,
        'platform':platform.platform(),'byteOrder':sys.byteorder,
        'referenceBackend':{'name':'torch CPU','version':str(torch.__version__),'gitVersion':torch.version.git_version,'capability':torch.backends.cpu.get_cpu_capability()},
        'checkpoint':{p.name:digest_file(p) for p in [Path(checkpoint)/'config.json',*sorted(Path(checkpoint).glob('*.safetensors'))]},
        'sources':{name:digest_file(Path(__file__).parent/name) for name in sorted(set(SOURCES+EXTRA_SOURCES))}}


def promote_tree(tree,root,rectangle,artifact):
    """Refine pending boxes at exact intersection boundaries, preserving old leaves.

    Caller owns compilation/parity admission. A contained subregion inherits
    the same input-only expression, valid throughout the compiled rectangle.
    No old completed leaf is replaced; no overlapping coverage is summed.
    """
    audit_tree(tree,root)
    if set(rectangle)!=set(root):raise ValueError('All input coordinates required')
    decode(rectangle)
    if any(not root[name][0]<=a<=b<=root[name][1] for name,(a,b) in rectangle.items()):
        raise ValueError('Cover region outside admitted domain')
    result=copy.deepcopy(tree);pending=[''];admitted=[]
    while pending:
        key=pending.pop();node=result[key];domains=node['domains']
        if node['status']=='split':pending.extend((key+'0',key+'1'));continue
        if node['status']=='complete':continue
        if any(b<rectangle[name][0] or a>rectangle[name][1] for name,(a,b) in domains.items()):continue
        cut=None
        for name,(low,high) in domains.items():
            a,b=rectangle[name]
            if low<a:cut=(name,a-1);break
            if high>b:cut=(name,b);break
        if cut is not None:
            axis,boundary=cut;_,left,right=split(decode(domains),axis,boundary)
            node.update(status='split',axis=axis,cut=boundary)
            result[key+'0']={'domains':encode(left),'status':'pending'}
            result[key+'1']={'domains':encode(right),'status':'pending'}
            pending.extend((key+'0',key+'1'))
        else:
            node.update(status='complete',artifact=dict(artifact),compiledDomain=copy.deepcopy(rectangle))
            admitted.append(key)
    before=audit_tree(tree,root)[0];after=audit_tree(result,root)[0]
    added=after-before
    if not 0<=added<=cardinality(rectangle):raise ValueError('Invalid overlapping coverage delta')
    return result,admitted,added


def cover(checkpoint,directory,rectangle,*,dimension=2,max_seconds=30,random_cases=8192):
    directory=Path(directory);manifest=directory/'frontier.json'
    # Share the controller's writer lock; uncertain or concurrent writes
    # cannot silently replace progress from another compilation process.
    with (directory/'writer.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        original=manifest.read_bytes();state=json.loads(original)
        expected=live_identity(checkpoint,dimension)
        if state['identity']!=expected:raise ValueError('Incompatible cover state; no saved expression reused')
        root=encode(CheckpointStrings(checkpoint,StringCompiler()).domains)
        if state['root']!=root:raise ValueError('Cover root differs from admitted input domain')
        before,pending=audit_tree(state['tree'],root)
        for node in state['tree'].values():
            if node['status']=='complete' and digest_file(directory/node['artifact']['file'])!=node['artifact']['sha256']:
                raise ValueError('Saved cover artifact integrity mismatch')
        _,_,possible=promote_tree(state['tree'],root,rectangle,{})
        report={'compilerIdentity':expected,'inputDomains':rectangle,'addedInputPatterns':0,
            'coveredBefore':before,'coveredAfter':before,'fullCoordinateParity':False,
            'toolSHA256':digest_file(Path(__file__))}
        if not possible:return {**report,'stop':'Region already covered; no manifest mutation'}
        with tempfile.TemporaryDirectory(prefix='direct-cover-') as temporary:
            artifact=Path(temporary)/'coordinate.expr'
            compiled=compile_region(checkpoint,dimension,decode(rectangle),artifact,
                max_characters=1048576,cas_characters=8388608,max_paths=128,max_seconds=max_seconds)
            report['compilation']=compiled
            if not compiled['complete']:return {**report,'stop':compiled['stop']}
            sha=digest_file(artifact)
            if sha!=compiled['artifact']['sha256']:raise ValueError('Fresh cover artifact integrity mismatch')
            torch.set_num_threads(1)
            parity=verify_region(checkpoint,dimension,artifact,decode(rectangle),random_cases=random_cases)
            report['parity']=parity
            if parity['mismatches']:raise ValueError('Fresh cover parity failed; manifest unchanged')
            # Recheck source/checkpoint identity after the isolated worker.
            if live_identity(checkpoint,dimension)!=expected:raise ValueError('Cover sources changed during compilation')
            filename='cover-'+sha+'.expr';record={'file':filename,'sha256':sha,'characters':compiled['artifact']['characters']}
            tree,admitted,added=promote_tree(state['tree'],root,rectangle,record)
            after,unfinished=audit_tree(tree,root)
            if manifest.read_bytes()!=original:raise ValueError('Cover manifest changed before publication')
            atomic(directory/filename,artifact.read_bytes())
            parity['artifact']=str(directory/filename)
            compiled['artifact']['path']=str(directory/filename)
            report.update(addedInputPatterns=added,coveredAfter=after,admittedRegions=admitted,
                sharedArtifact=record,manifestBeforeSHA256=hashlib.sha256(original).hexdigest())
            state.update(tree=tree,coveredInputPatterns=after,unfinishedInputPatterns=unfinished)
            state.setdefault('coveragePromotions',[]).append(report)
            # The normal controller still owns final dispatch emission. Even
            # a fully covered tree is not final parity or a completed vector.
            atomic(manifest,canonical(state))
            return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint');parser.add_argument('state');parser.add_argument('report')
    parser.add_argument('--domains',required=True,help='Input rank intervals; expressions remain mathematical strings')
    parser.add_argument('--dimension',type=int,default=2);parser.add_argument('--max-seconds',type=int,default=30)
    parser.add_argument('--random-cases',type=int,default=8192)
    args=parser.parse_args()
    if args.max_seconds<1 or args.random_cases<0:parser.error('Positive time budget and nonnegative sample count required')
    result=cover(args.checkpoint,args.state,json.loads(args.domains),dimension=args.dimension,
        max_seconds=args.max_seconds,random_cases=args.random_cases)
    atomic(args.report,json.dumps(result,indent=2).encode())
    print(json.dumps(result),flush=True)
    return 0 if result['addedInputPatterns'] else 1


if __name__=='__main__':raise SystemExit(main())
