"""Memory-admitted independent regions, with ordered atomic compiler commits.

Numerical workers use the unchanged controller worker protocol. Regions own
all CAS/branch contexts. No arithmetic is reassociated or runtime alias added.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import uuid

import torch
from direct_sympy_cover_regions import live_identity
from direct_sympy_checkpoint import CheckpointStrings
from direct_sympy_partition_run import audit_tree,cardinality,decode,encode,next_region,region_character_charge
from direct_sympy_input_partitions import rank,split
from direct_sympy_region_parity import verify_region
from direct_sympy_savepoints import atomic,canonical,digest_file
from direct_sympy_strings import StringCompiler


def rss_bytes(pid):
    """Current RSS on macOS/Linux; unavailable live accounting is an error."""
    result=subprocess.run(['ps','-o','rss=','-p',str(pid)],capture_output=True,text=True)
    text=result.stdout.strip()
    if not text:return None
    return int(text)*1024


def publish_coordinate(args,directory,state,identity,manifest,original):
    """Coalesce exact bodies and validate the actual full-domain artifact.

    Failed emission/parity leaves both the manifest and any previous final
    expression untouched. Compilation proofs remain leaf-local; grouping
    only byte-identical bodies introduces no numerical transformation.
    """
    from direct_sympy_partition_coalesce import emit
    candidate=directory/('.coordinate-'+uuid.uuid4().hex+'.expr')
    destination=directory/'coordinate.expr'
    try:
        emission=emit(directory,state['tree'],candidate,max_characters=args.total_characters)
        parity=verify_region(args.checkpoint,args.dimension,candidate,decode(state['root']),random_cases=args.random_cases)
        if parity['mismatches']:raise ValueError('Combined coordinate parity failed; final artifact not published')
        if digest_file(candidate)!=emission['artifact']['sha256'] or parity['sha256']!=emission['artifact']['sha256']:
            raise ValueError('Combined coordinate integrity changed during parity; final artifact not published')
        if live_identity(args.checkpoint,args.dimension)!=identity:raise ValueError('Sources changed during combined coordinate parity')
        if manifest.read_bytes()!=original:raise ValueError('Parallel manifest changed before final publication')
        artifact=emission['artifact']
        parity['artifact']=str(destination)
        final={'file':destination.name,'characters':artifact['characters'],'sha256':artifact['sha256'],
            'coveredInputPatterns':state['coveredInputPatterns'],'complete':True,'parityVerified':True}
        # The temporary expression is the exact file verified above.
        os.replace(candidate,destination)
        # The current numerical adapter/verifier covers position zero only.
        # Native corpus parity must not stand in for variable-length parity.
        state.update(finalArtifact=final,finalArtifactEmitted=True,finalParity=False,
            positionZeroArtifactParityVerified=True,
            finalParityEvidence=parity,finalParityScope='position-zero-native-corpus',
            finalCoalescing={name:emission[name] for name in ('completeRegions','coalescedRectangles',
                'distinctExpressions','copiedBodyCharacters','sharedBodyCharacters')})
        atomic(manifest,canonical(state))
        return {'artifact':final,'parity':parity,'coalescing':state['finalCoalescing']}
    finally:candidate.unlink(missing_ok=True)


def compile_wave(checkpoint,dimension,jobs,*,workers,memory_bytes,cas_characters,
                 max_characters,max_paths,max_seconds,job_seconds=None):
    if min(workers,memory_bytes,cas_characters,max_characters,max_paths,max_seconds)<1:
        raise ValueError('Positive region budgets required')
    limits=[max_seconds]*len(jobs) if job_seconds is None else job_seconds
    if len(limits)!=len(jobs) or any(limit<1 for limit in limits):raise ValueError('Positive deadline per region required')
    reservation=768*1024**2+16*cas_characters
    active=[];results=[None]*len(jobs);index=0
    stats={'maxWorkersLive':0,'peakObservedRSSBytes':0,'workerReservationBytes':reservation}
    started=time.monotonic()
    with tempfile.TemporaryDirectory(prefix='parallel-region-') as temporary:
        try:
            while index<len(jobs) or active:
                parent=rss_bytes(os.getpid())
                if parent is None:raise ValueError('Cannot measure compiler parent RSS')
                resident=parent
                for task in active:
                    process=task['process'];rss=rss_bytes(process.pid)
                    if rss is None and process.poll() is None:raise ValueError('Cannot measure live region worker RSS')
                    resident+=rss or 0
                stats['peakObservedRSSBytes']=max(stats['peakObservedRSSBytes'],resident)
                if resident>memory_bytes:raise ValueError('Parallel region RSS exceeded memory budget')
                # Reserve a full worst-case slot for every live worker, even
                # when its early import phase has not allocated its CAS data.
                committed=parent+sum(max(reservation,rss_bytes(t['process'].pid) or 0) for t in active)
                while index<len(jobs) and len(active)<workers and committed+reservation<=memory_bytes:
                    key,domains,path=jobs[index]
                    out=open(Path(temporary)/f'{index}.out','w+b');err=open(Path(temporary)/f'{index}.err','w+b')
                    arguments=[sys.executable,str(Path(__file__).with_name('direct_sympy_partition_run.py')),
                        '--worker',str(checkpoint),str(path),str(dimension),str(max_characters),
                        str(cas_characters),str(max_paths),json.dumps(domains)]
                    process=subprocess.Popen(arguments,stdin=subprocess.DEVNULL,stdout=out,stderr=err)
                    active.append({'index':index,'process':process,'out':out,'err':err,'started':time.monotonic(),'limit':limits[index]})
                    index+=1;committed+=reservation
                    stats['maxWorkersLive']=max(stats['maxWorkersLive'],len(active))
                if not active and index<len(jobs):raise ValueError('Parallel memory budget cannot admit one region')
                for task in list(active):
                    process=task['process'];elapsed=time.monotonic()-task['started']
                    timed_out=process.poll() is None and elapsed>=task['limit']
                    if timed_out:process.kill();process.wait()
                    if process.poll() is None:continue
                    try:
                        if timed_out:result={'complete':False,'stop':'Region worker wall-clock budget exceeded','seconds':elapsed}
                        else:
                            task['out'].seek(0);task['err'].seek(0)
                            if process.returncode:raise ValueError('Region worker semantic failure: '+task['err'].read().decode()[-2000:])
                            result=json.loads(task['out'].read());result['workerSeconds']=elapsed
                        result['inputDomains']=jobs[task['index']][1]
                        result['timeoutSeconds']=task['limit']
                        results[task['index']]=result
                    finally:task['out'].close();task['err'].close();active.remove(task)
                if active:time.sleep(.05)
        finally:
            for task in active:
                if task['process'].poll() is None:task['process'].kill()
                task['process'].wait();task['out'].close();task['err'].close()
    stats['seconds']=time.monotonic()-started
    return results,stats


def run(args):
    directory=Path(args.state);manifest=directory/'frontier.json';torch.set_num_threads(1)
    with (directory/'writer.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        original=manifest.read_bytes();state=json.loads(original)
        identity=live_identity(args.checkpoint,args.dimension)
        if state['identity']!=identity:raise ValueError('Incompatible parallel state; no expression reused')
        root=encode(CheckpointStrings(args.checkpoint,StringCompiler()).domains)
        if state['root']!=root:raise ValueError('Parallel state must cover the complete admitted root')
        audit_tree(state['tree'],root)
        policy=state.get('regionVerificationPolicy',{})
        if policy.get('enabled') and policy.get('randomCases')!=args.random_cases:
            raise ValueError('Incompatible region verification policy; saved proofs not downgraded')
        for node in state['tree'].values():
            if node['status']=='complete' and digest_file(directory/node['artifact']['file'])!=node['artifact']['sha256']:
                raise ValueError('Saved parallel artifact integrity mismatch')
        charge=10
        for node in state['tree'].values():
            if node['status']!='complete':continue
            cost=region_character_charge(directory,node)
            if policy.get('enabled') and (
                not node.get('parity') or node['parity'].get('mismatches')!=0
                or node['parity'].get('sha256')!=node['artifact']['sha256']):
                raise ValueError('Unverified saved region cannot enter parallel compilation')
            node['artifact']['combinedCharacterCharge']=cost;charge+=cost
        if charge>args.total_characters:raise ValueError('Saved regions exceed accumulated artifact budget')
        initial=state['coveredInputPatterns'];report={'controllerSHA256':digest_file(Path(__file__)),
            'compilerIdentity':identity,'workersRequested':args.workers,'memoryBudgetBytes':args.memory_bytes,
            'waves':[],'nativeCases':0,'nativeMismatches':0,'fullCoordinateParity':False,
            'coveredInputPatterns':initial,'addedInputPatterns':0,'unfinishedInputPatterns':state['unfinishedInputPatterns']}
        started=time.monotonic();performed=0;run_id=uuid.uuid4().hex
        order=getattr(args,'frontier_order',None) or ('update-cells' if 'updateCellGeometry' in state else 'lexical')
        while performed<args.max_attempts:
            queue={key:dict(node) for key,node in state['tree'].items()};jobs=[]
            for _ in range(min(args.workers,args.max_attempts-performed)):
                key=next_region(queue,order,state.get('updateCellGeometry',{}).get('thresholdRanks'))
                if key is None:break
                queue[key]['status']='reserved'
                jobs.append((key,state['tree'][key]['domains'],directory/f'.parallel-{run_id}-{key}.expr'))
            if not jobs:break
            try:
                results,stats=compile_wave(args.checkpoint,args.dimension,jobs,workers=args.workers,
                    memory_bytes=args.memory_bytes,cas_characters=args.cas_characters,max_characters=args.max_characters,
                    max_paths=args.max_paths,max_seconds=args.region_seconds,
                    job_seconds=[args.region_seconds*getattr(args,'retry_factor',3)**state['tree'][key].get('timeoutRetries',0) for key,_,_ in jobs])
                # Reap the entire wave before native validation/publication.
                # Ordered admission never depends on worker completion order.
                if live_identity(args.checkpoint,args.dimension)!=identity:raise ValueError('Sources changed during parallel compilation')
                if manifest.read_bytes()!=original:raise ValueError('Parallel manifest changed during worker wave')
                admissions=[];costs=[];prospective_charge=charge
                for (key,domains,path),result in zip(jobs,results):
                    cost=0
                    if result['complete']:
                        cost=region_character_charge(directory,{'domains':domains,
                            'artifact':{'file':path.name,'sha256':result['artifact']['sha256']}})
                    costs.append(cost);prospective_charge+=cost
                if prospective_charge>args.total_characters:
                    raise ValueError('Accumulated expression/condition budget exceeded; wave not published')
                for (key,domains,path),result in zip(jobs,results):
                    if result['complete']:
                        if digest_file(path)!=result['artifact']['sha256']:raise ValueError('Fresh worker artifact integrity mismatch')
                        parity=verify_region(args.checkpoint,args.dimension,path,decode(domains),random_cases=args.random_cases)
                        if parity['mismatches'] or parity['sha256']!=result['artifact']['sha256']:
                            raise ValueError('Fresh parallel region parity failed; wave not published')
                        report['nativeCases']+=parity['cases'];report['nativeMismatches']+=parity['mismatches']
                        admissions.append(parity)
                    else:admissions.append(None)
                if live_identity(args.checkpoint,args.dimension)!=identity:raise ValueError('Sources changed during parallel parity')
                if manifest.read_bytes()!=original:raise ValueError('Parallel manifest changed before publication')
                for (key,domains,path),result,parity,cost in zip(jobs,results,admissions,costs):
                    node=state['tree'][key]
                    if result['complete']:
                        filename='parallel-'+result['artifact']['sha256']+'.expr';destination=directory/filename
                        if destination.exists():
                            if digest_file(destination)!=result['artifact']['sha256']:raise ValueError('Shared artifact integrity mismatch')
                        else:atomic(destination,path.read_bytes())
                        result['artifact']['path']=str(destination);parity['artifact']=str(destination)
                        node.update(status='complete',artifact={'file':filename,'combinedCharacterCharge':cost,
                            **{name:result['artifact'][name] for name in ('characters','sha256')}},parity=parity)
                    elif ('wall-clock budget exceeded' in result.get('stop','').lower()
                          and node.get('timeoutRetries',0)<getattr(args,'timeout_retries',1)):
                        # A deadline is not proof that a region is too large.
                        # Retry the same numerical domain before creating cuts.
                        node['timeoutRetries']=node.get('timeoutRetries',0)+1
                    else:
                        choices=[name for name,d in decode(domains).items() if rank(d.maximum)-rank(d.minimum)+1>2*args.min_values]
                        if choices:
                            def priority(name):
                                d=decode(domains)[name];minimum=d.minimum if d.minimum>0 else -d.maximum if d.maximum<0 else 0
                                return minimum,-(rank(d.maximum)-rank(d.minimum)),name
                            axis=min(choices,key=priority);_,left,right=split(decode(domains),axis)
                            node.update(status='split',axis=axis)
                            state['tree'][key+'0']={'domains':encode(left),'status':'pending'}
                            state['tree'][key+'1']={'domains':encode(right),'status':'pending'}
                        else:state['stop']='Cannot subdivide further without enumerating individual inputs'
                    state['attempts'].append({'region':key,'selectionOrder':order,'parallelRun':run_id,**result})
                    performed+=1
                covered,pending=audit_tree(state['tree'],root)
                charge=prospective_charge;state['accumulatedArtifactCharacters']=charge
                stats.update(regions=[key for key,_,_ in jobs],completed=sum(r['complete'] for r in results),coveredInputPatterns=covered,
                    deferredTimeoutRegions=[key for key,_,_ in jobs if state['tree'][key]['status']=='pending'])
                report['waves'].append(stats)
                report.update(seconds=time.monotonic()-started,attempts=performed,coveredInputPatterns=covered,
                    addedInputPatterns=covered-initial,unfinishedInputPatterns=pending)
                state.update(coveredInputPatterns=covered,unfinishedInputPatterns=pending,totalInputPatterns=cardinality(root),frontierOrder=order)
                state.setdefault('parallelRuns',{})[run_id]=report
                atomic(manifest,canonical(state));original=manifest.read_bytes()
                print(json.dumps(stats),flush=True)
            finally:
                for _,_,path in jobs:path.unlink(missing_ok=True)
            if 'stop' in state:break
        if state['unfinishedInputPatterns']==0:
            final=publish_coordinate(args,directory,state,identity,manifest,original)
            report.update(finalCoordinate=final,positionZeroArtifactParityVerified=True,
                parityScope='position-zero-native-corpus')
            report['nativeCases']+=final['parity']['cases']
        report['seconds']=time.monotonic()-started
        return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint');parser.add_argument('state');parser.add_argument('report')
    parser.add_argument('--dimension',type=int,default=2);parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--memory-bytes',type=int,default=3*1024**3);parser.add_argument('--max-attempts',type=int,default=8)
    parser.add_argument('--timeout-retries',type=int,default=1);parser.add_argument('--retry-factor',type=int,default=3)
    parser.add_argument('--region-seconds',type=int,default=10);parser.add_argument('--random-cases',type=int,default=256)
    parser.add_argument('--max-paths',type=int,default=128);parser.add_argument('--max-characters',type=int,default=1048576)
    parser.add_argument('--cas-characters',type=int,default=8388608);parser.add_argument('--min-values',type=int,default=32)
    parser.add_argument('--total-characters',type=int,default=67108864)
    parser.add_argument('--frontier-order',choices=('lexical','coverage','update-cells'))
    args=parser.parse_args()
    if min(args.workers,args.memory_bytes,args.max_attempts,args.region_seconds,args.max_paths,args.max_characters,args.cas_characters,args.min_values,args.total_characters,args.retry_factor)<1 or args.random_cases<0 or args.timeout_retries<0:parser.error('Positive budgets and nonnegative corpus required')
    report=run(args);atomic(args.report,json.dumps(report,indent=2).encode())
    return 0 if report.get('addedInputPatterns',0) or report.get('positionZeroArtifactParityVerified') else 1


if __name__=='__main__':raise SystemExit(main())
