"""Fresh, bounded input-only coordinate emission with coherent path contexts.

Legacy producer savepoints are deliberately not loaded: this compiler-only
sharing path has a different state representation. Position zero is the
current adapter scope, not variable-length last-token compilation.
"""
import argparse
import json
from pathlib import Path
import resource
import signal
import sys
import time

from direct_sympy_checkpoint import CheckpointStrings
from direct_sympy_coherent_paths import CoherentPaths
from direct_sympy_streaming_literals import streaming_literals
from direct_sympy_strings import StringCompiler


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint');parser.add_argument('artifact');parser.add_argument('report')
    parser.add_argument('--dimension',type=int,default=2)
    parser.add_argument('--max-characters',type=int,default=1024**2)
    parser.add_argument('--cas-characters',type=int,default=8*1024**2)
    parser.add_argument('--max-paths',type=int,default=64)
    parser.add_argument('--max-seconds',type=int,default=90)
    args=parser.parse_args()
    if min(args.max_characters,args.cas_characters,args.max_paths,args.max_seconds)<1:
        parser.error('Positive budgets required')
    if Path(args.artifact).resolve()==Path(args.report).resolve():
        parser.error('Artifact and diagnostic report must be different files')
    report={'coordinateCompositionComplete':False,'finalArtifactEmitted':False,
        'finalParity':False,'position':0,'dimension':args.dimension,
        'stateRestored':False,'expressionFormat':'Mathematical string',
        'budgets':{'artifactCharacters':args.max_characters,'casCharacters':args.cas_characters,
            'paths':args.max_paths,'seconds':args.max_seconds}}
    started=time.monotonic();result=1;plan=None
    def timeout(*_):raise TimeoutError('Compilation wall-clock budget exceeded')
    previous=signal.signal(signal.SIGALRM,timeout);signal.alarm(args.max_seconds)
    try:
        with CheckpointStrings(args.checkpoint,StringCompiler(max_characters=args.cas_characters)) as model:
            with streaming_literals(model) as registry:
                expression=model.coordinate(args.dimension)
                report.update(coordinateCompositionComplete=True,completedProducers=len(model.memo),
                    compilerDefinitions=len(registry.definitions),
                    storedCharacters=sum(map(len,registry.definitions)),
                    logicalCharacters=registry.size(expression),
                    compositionSeconds=time.monotonic()-started)
                report['producerGrowth']=[{'producer':key,'storedCharacters':stored,
                    'expandedCharacters':expanded,'definitions':definitions}
                    for key,stored,expanded,definitions in registry.events]
                plan=CoherentPaths(registry,max_paths=args.max_paths)
                report['artifact']=plan.write(args.artifact,expression,max_characters=args.max_characters)
                report['finalArtifactEmitted']=True
                # Artifact emission alone never claims reference parity.
                result=0
    except (ValueError,TimeoutError,MemoryError) as error:
        report['stop']=str(error) or 'Compiler memory exhausted'
    finally:
        signal.alarm(0);signal.signal(signal.SIGALRM,previous)
        if plan is not None:report['paths']=plan.stats
        report['seconds']=time.monotonic()-started
        rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report['peakRSSBytes']=rss if sys.platform=='darwin' else rss*1024
        path=Path(args.report);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
    return result


if __name__=='__main__':raise SystemExit(main())
