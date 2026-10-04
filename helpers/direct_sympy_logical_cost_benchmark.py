"""Compare compiler resident-marker and fully substituted literal costs.

Reports are diagnostics, never final model artifacts. The numerical verifier
reuses compiler-only literals; no whole expanded coordinate is emitted here.
Each run is a fresh process, with the same checkpoint and fresh CPU reference.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import struct
import subprocess
import sys
import time


def worker(args):
    from direct_sympy_checkpoint import CheckpointStrings
    from direct_sympy_parallel import ParallelBudget
    from direct_sympy_strings import StringCompiler,syntax
    from direct_sympy_streaming_literals import streaming_literals,ALIASES
    import torch
    torch.set_num_threads(1)
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Compilation time budget')))
    signal.alarm(120)
    budget=ParallelBudget(2,8*1024**3,1) if args.mode=='parallel-logical' else None
    started=time.monotonic()
    with CheckpointStrings(args.checkpoint,StringCompiler(max_characters=16*1024**2),parallel_budget=budget) as model:
        with streaming_literals(model) as registry:
            if args.mode=='marker-cost':model.compiler.expression_size=len
            result=model.coordinate(args.dimension)
            seconds=time.monotonic()-started
            logical=registry.size(result)
            growth=[{'producer':name,'residentCharacters':size,'expandedCharacters':expanded,'sharedDefinitions':count} for name,size,expanded,count in registry.events]
            definitions=list(registry.definitions)
            stats=list(model.parallel_events)
            cas=len(model.compiler.events)
            if not all(event[2:4]==('factor','simplify') for event in model.compiler.events):raise AssertionError('Missing CAS cycle')
    signal.alarm(0)
    class LazyBranches(ast.NodeTransformer):
        def visit_Call(self,node):
            node=self.generic_visit(node)
            if node.func.id!='Piecewise':return node
            result=ast.Constant(value=0)
            for pair in reversed(node.args):result=ast.IfExp(test=pair.elts[1],body=pair.elts[0],orelse=result)
            return result
    mask=2**64-1
    functions={'F64FromU64':float,'U64FromF64':int,
        'Bits64':lambda x:struct.unpack('Q',struct.pack('d',float(x)))[0],
        'Float64':lambda x:struct.unpack('d',struct.pack('Q',x))[0],
        'U64Add':lambda a,b:(a+b)&mask,'U64Mul':lambda a,b:(a*b)&mask,
        'U64And':lambda a,b:a&b,'U64Or':lambda a,b:a|b,'U64Shr':lambda a,b:a>>b,
        'And':lambda *x:all(x),'Or':lambda *x:any(x),'Not':lambda x:not x}
    programs=[compile(ast.fix_missing_locations(ast.Expression(LazyBranches().visit(syntax(text)))),'<compiler-only-literal>','eval') for text in definitions]
    index=int(ALIASES.fullmatch(result)[1])
    cases=json.loads(Path(args.reference).read_text())['cases']
    mismatches=[]
    for row in cases:
        values={f'X{i+1}':struct.unpack('e',struct.pack('H',bits))[0] for i,bits in enumerate(row['inputBits'][0])}
        for i,program in enumerate(programs):
            actual=eval(program,{'__builtins__':{},**functions},values)
            values['CompileValue'+str(i)]=lambda actual=actual:actual
        actual=values['CompileValue'+str(index)]()
        if '0x'+struct.pack('>d',actual).hex()!=row['logitF64Bits'][0][args.dimension]:mismatches.append(row['label'])
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report={'mode':args.mode,'compilationSeconds':seconds,'peakProcessRSSBytes':peak if sys.platform=='darwin' else peak*1024,
        'completedProducers':len(growth),'compilerDefinitions':len(definitions),'storedLiteralCharacters':sum(map(len,definitions)),
        'logicalOutputCharacters':logical,'CASPasses':cas,'growth':growth,'parallel':stats,
        'parity':{'cases':len(cases),'mismatches':len(mismatches),'labels':mismatches,'position':0,'dimension':args.dimension,
                  'scope':'Complete compiler-only composition; verifier reuses immutable definitions. No expanded final artifact emitted. Longer sequences verify position zero, not their last-token output.'},
        'finalArtifactEmitted':False,'finalParity':False}
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
    if mismatches:raise AssertionError(mismatches)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('checkpoint');parser.add_argument('output')
    parser.add_argument('--dimension',type=int,default=2)
    parser.add_argument('--mode',choices=['marker-cost','sequential-logical','parallel-logical'])
    parser.add_argument('--reference')
    args=parser.parse_args()
    if args.mode:
        if not args.reference:parser.error('Worker requires a fresh reference')
        worker(args);return
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    reference=root/'fresh-reference.json'
    subprocess.run([sys.executable,str(Path(__file__).with_name('capture_direct_json_reference.py')),args.checkpoint,str(reference)],check=True,capture_output=True,text=True)
    reports=[]
    for mode in ('marker-cost','sequential-logical','parallel-logical'):
        path=root/(mode+'.json')
        process=subprocess.run([sys.executable,str(Path(__file__).resolve()),args.checkpoint,str(path),'--mode',mode,'--dimension',str(args.dimension),'--reference',str(reference)],capture_output=True,text=True,timeout=150)
        (root/(mode+'.log')).write_text(process.stdout+process.stderr)
        process.check_returncode();reports.append(json.loads(path.read_text()))
        print(json.dumps({key:reports[-1][key] for key in ('mode','compilationSeconds','peakProcessRSSBytes','completedProducers','logicalOutputCharacters','parity')}),flush=True)
    helpers=Path(__file__).parent
    source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(helpers.glob('direct_sympy*.py'))}
    comparison={'sourceHashes':source_hashes,'runs':reports,
        'logicalReductionPercent':100*(1-reports[1]['logicalOutputCharacters']/reports[0]['logicalOutputCharacters']),
        'parallelTimeChangePercent':100*(reports[2]['compilationSeconds']/reports[1]['compilationSeconds']-1),
        'parallelPeakProcessRSSChangePercent':100*(reports[2]['peakProcessRSSBytes']/reports[1]['peakProcessRSSBytes']-1),
        'finalArtifactEmitted':False,'finalParity':False}
    (root/'comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')


if __name__=='__main__':main()
