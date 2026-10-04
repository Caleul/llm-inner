"""Bounded, persistent continuation of one direct checkpoint coordinate.

This controller does not alter numerical compilation. It validates normal
savepoint identity before reuse and saves each newly completed producer.
Reports and partial savepoints never constitute complete-coordinate parity.
"""
import argparse
import json
import os
from pathlib import Path
import resource
import signal
import sys
import time
import traceback


def main():
    from direct_sympy_checkpoint import CheckpointStrings,write_expression
    from direct_sympy_envelope_diagnostic import capture_envelopes
    from direct_sympy_savepoints import ProducerSavepoints
    from direct_sympy_scan_backend import install
    from direct_sympy_strings import StringCompiler
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint');parser.add_argument('state');parser.add_argument('report')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--dimension',type=int,default=2)
    parser.add_argument('--max-characters',type=int,default=8*1024**3)
    parser.add_argument('--max-seconds',type=int,default=600)
    parser.add_argument('--max-address-space-mib',type=int,default=0,
                        help='Optional process virtual-memory cap, including imported libraries')
    args=parser.parse_args()
    if args.max_characters<1 or args.max_seconds<1 or args.max_address_space_mib<0:
        parser.error('Positive character/time budgets and nonnegative memory cap required')
    exists=(Path(args.state)/'frontier.json').is_file()
    if args.resume!=exists:parser.error('Existing state requires --resume; resume requires an existing state')
    # Imports happen first: the requested cap includes their mappings. Lower
    # only this process's soft limit, preserving its original hard limit.
    original_limit=resource.getrlimit(resource.RLIMIT_AS)
    if args.max_address_space_mib:
        _,hard=resource.getrlimit(resource.RLIMIT_AS)
        cap=args.max_address_space_mib*1024**2
        if hard!=resource.RLIM_INFINITY:cap=min(cap,hard)
        statm=Path('/proc/self/statm')
        if statm.exists() and cap<=int(statm.read_text().split()[0])*os.sysconf('SC_PAGE_SIZE'):
            parser.error('Address-space cap must exceed already mapped compiler libraries')
        resource.setrlimit(resource.RLIMIT_AS,(cap,hard))
    def timeout(*_):raise TimeoutError('Compilation wall-clock budget exceeded')
    previous=signal.signal(signal.SIGALRM,timeout);signal.alarm(args.max_seconds)
    started=time.monotonic();report={'coordinateComplete':False,'parityVerified':False,
        'resumeCompatible':False,'restoredDependencies':0,'completedDependencies':0,'envelopes':[],
        'budgets':{'characters':args.max_characters,'seconds':args.max_seconds,
                   'addressSpaceBytes':args.max_address_space_mib*1024**2}}
    result=1
    def record_stop(error,phase):
        report['stop']=str(error) or 'Process address-space budget exhausted'
        report['failurePhase']=phase
        report['failureFrames']=[{'file':Path(frame.f_code.co_filename).name,
            'function':frame.f_code.co_name,'line':line}
            for frame,line in traceback.walk_tb(error.__traceback__)][-12:]
    try:
        with install(),CheckpointStrings(args.checkpoint,StringCompiler(max_characters=args.max_characters)) as model:
            with ProducerSavepoints(args.state,model,args.dimension,compressed=True) as store:
                if args.resume:
                    report['restoredDependencies']=store.restore(model)
                    report['resumeCompatible']=True
                    print(json.dumps({'event':'compatible-state-restored',
                        'dependencies':report['restoredDependencies']}),flush=True)
                def persist(completed):
                    store.save(completed)
                    name=next(reversed(completed.memo))
                    print(json.dumps({'event':'producer-persisted','name':name,
                        'characters':len(completed.memo[name]),'dependencies':len(completed.memo),
                        'seconds':time.monotonic()-started}),flush=True)
                model.on_completed=persist
                with capture_envelopes(model,report['envelopes']):
                    try:
                        expression=model.coordinate(args.dimension)
                        artifact=Path(str(args.report)+'.coordinate.work.expr')
                        artifact.parent.mkdir(parents=True,exist_ok=True)
                        write_expression(artifact,expression)
                        report['coordinateArtifact']=str(artifact)
                        report['coordinateCompilationFinished']=True
                        result=0
                    except (ValueError,TimeoutError,MemoryError) as error:
                        record_stop(error,'compilation')
                report['completedDependencies']=len(model.memo)
                report['newDependencies']=len(model.memo)-report['restoredDependencies']
                report['lastDependency']=next(reversed(model.memo),None)
    except (ValueError,TimeoutError,MemoryError) as error:
        record_stop(error,'restore/setup')
    finally:
        signal.alarm(0);signal.signal(signal.SIGALRM,previous)
        report['seconds']=time.monotonic()-started
        if args.max_address_space_mib:resource.setrlimit(resource.RLIMIT_AS,original_limit)
        native=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report['peakRSSBytes']=native if sys.platform=='darwin' else native*1024
        manifest=Path(args.state)/'frontier.json'
        if manifest.is_file() and (not args.resume or report['resumeCompatible']):
            records=json.loads(manifest.read_text())['payload']['records']
            report['persistedDependencies']=len(records)
            report['lastPersistedDependency']=records[-1]['name'] if records else None
        path=Path(args.report);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='envelopes'}),flush=True)
    return result


if __name__=='__main__':raise SystemExit(main())
