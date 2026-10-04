"""Observe rejected compact envelopes without changing numeric compilation.

The report contains compiler aliases and costs only, never a runtime artifact.
Saved expressions are restored with the normal strict identity/hash checks.
"""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import re
import resource
import signal
import sys
import time


@contextmanager
def capture_envelopes(model, reports):
    from direct_sympy_conversions import ConversionSession
    target=ConversionSession._close_completed.__code__
    previous=sys.gettrace()
    def observe(frame,event,arg):
        if event=='exception':
            _,error,_=arg
            if isinstance(error,ValueError) and str(error).startswith('Closed numeric envelope exceeds string budget before allocation:'):
                values=frame.f_locals
                result=values['result']
                if len(result)>1048576:
                    reports.append({'error':str(error),'compactCharacters':len(result),'compactOmitted':True})
                    return observe
                protected=values['protected']
                owners={id(text):name for name,text in model.memo.items()}
                counts={name:len(re.findall(r'\b'+re.escape(name)+r'\(\)',result)) for name in protected}
                aliases=[{'alias':name,'producer':owners.get(id(text)),
                          'characters':len(text),'occurrences':counts[name],
                          'restoredContributionCharacters':counts[name]*len(text)} for name,text in protected.items()]
                # The substitution keeps every occurrence; only alias spelling
                # is replaced. This equality also detects accounting mistakes.
                expanded=len(result)+sum(counts[name]*(len(text)-len(name)-2) for name,text in protected.items())
                reports.append({'error':str(error),'compact':result,'compactCharacters':len(result),
                                'expandedCharacters':expanded,'aliases':aliases,
                                'inputCharacters':values['input_characters'],
                                'scope':'Rejected compiler envelope; aliases are not final runtime variables'})
        return observe
    def dispatch(frame,event,arg):
        return observe if frame.f_code is target else None
    sys.settrace(dispatch)
    try:yield
    finally:sys.settrace(previous)


def main():
    from direct_sympy_checkpoint import CheckpointStrings,write_expression
    from direct_sympy_savepoints import ProducerSavepoints
    from direct_sympy_scan_backend import install
    from direct_sympy_strings import StringCompiler
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint');parser.add_argument('state');parser.add_argument('report')
    parser.add_argument('--fresh',action='store_true',help='Create a new state; refuse to overwrite existing state')
    parser.add_argument('--dimension',type=int,default=2)
    parser.add_argument('--max-characters',type=int,default=16*1024**3)
    parser.add_argument('--max-seconds',type=int,default=600)
    args=parser.parse_args()
    if args.max_characters<1 or args.max_seconds<1:parser.error('Positive budgets required')
    state=Path(args.state)
    if args.fresh and (state/'frontier.json').exists():parser.error('Fresh diagnostic cannot overwrite existing savepoint')
    if not args.fresh and not (state/'frontier.json').is_file():parser.error('Existing savepoint required')
    def timeout(*_):raise TimeoutError('Diagnostic compilation wall-clock budget exceeded')
    previous=signal.signal(signal.SIGALRM,timeout);signal.alarm(args.max_seconds)
    started=time.monotonic();report={'coordinateComplete':False,'envelopes':[]}
    try:
        with install(),CheckpointStrings(args.checkpoint,StringCompiler(max_characters=args.max_characters)) as model:
            with ProducerSavepoints(state,model,args.dimension,compressed=True) as store:
                report['restoredDependencies']=0 if args.fresh else store.restore(model)
                # Resumed diagnostics never overwrite the source state.
                if args.fresh:model.on_completed=store.save
                with capture_envelopes(model,report['envelopes']):
                    try:
                        expression=model.coordinate(args.dimension)
                        artifact=Path(str(args.report)+'.coordinate.work.expr')
                        artifact.parent.mkdir(parents=True,exist_ok=True)
                        write_expression(artifact,expression)
                        report['coordinateArtifact']=str(artifact)
                        report['coordinateCompilationFinished']=True
                        report['parityVerified']=False
                    except (ValueError,TimeoutError) as error:report['stop']=str(error)
                report['completedDependencies']=len(model.memo)
    finally:
        signal.alarm(0);signal.signal(signal.SIGALRM,previous)
        report['seconds']=time.monotonic()-started
        report['peakRSSNativeUnits']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        path=Path(args.report);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({key:value for key,value in report.items() if key!='envelopes'}),flush=True)


if __name__=='__main__':main()
