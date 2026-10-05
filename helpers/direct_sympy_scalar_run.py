"""Bounded Colab compilation and parity admission for the direct scalar Rust route."""
import dataclasses
from fractions import Fraction as F
import hashlib
import json
import os
from pathlib import Path
import signal
import struct
import subprocess
import sys
import time

from direct_sympy_architecture_run import write_json


def unit(args):
    import torch
    torch.set_num_threads(1)
    from direct_sympy_architecture import architecture_plan,prune_plan,last_position_indices
    from direct_sympy_scalar import compose_scalar_operators
    from direct_sympy_strings import StringCompiler,Domain
    from direct_sympy_scan_backend import install
    from direct_sympy_parallel import ParallelBudget
    from direct_sympy_architecture_test import numeric_functions
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    started=time.monotonic();report={'length':args.unit_length,'finalArtifactParity':False}
    def deadline(*_):raise TimeoutError('Scalar architecture deadline exceeded')
    signal.signal(signal.SIGALRM,deadline);signal.alarm(args.seconds_per_length)
    try:
        plan=architecture_plan(args.checkpoint,args.unit_length,max_characters=args.max_characters)
        blocks=prune_plan(plan,last_position_indices(plan))
        compiler=StringCompiler(max_characters=args.max_characters)
        domains={f'X{i+1}':Domain(F(-65504),F(65504),-24,False) for i in range(plan.length*plan.width)}
        with install():
            program,stats=compose_scalar_operators(blocks,compiler,domains,
                ParallelBudget(args.workers,args.memory_mib*1024**2),max_bytes=args.max_characters,
                on_wave=lambda stats:write_json(root/'composition-progress.json',stats))
        descriptor={'schema':'direct-scalar-strings-v1',**dataclasses.asdict(program),'certificate':plan.finite_certificate}
        write_json(root/'scalar-program.json',descriptor)
        # Execute the persisted descriptor, never a different in-memory program.
        from direct_sympy_scalar import ScalarProgram
        reread=json.loads((root/'scalar-program.json').read_text())
        program=ScalarProgram(tuple(reread['inputs']),tuple(tuple(row) for row in reread['definitions']),tuple(reread['outputs'])).validate()
        comparisons=0;mismatches=[];functions=numeric_functions()
        corpus=json.loads(Path(args.reference).read_text())
        for case in corpus['cases']:
            if len(case['inputBits'])!=plan.length:continue
            values=[struct.unpack('e',struct.pack('H',bits))[0] for row in case['inputBits'] for bits in row]
            actual=['0x'+struct.pack('>d',v).hex() for v in program.evaluate(values,functions)]
            expected=case['logitF64Bits'][-1];comparisons+=len(expected)
            if actual!=expected:mismatches.append({'case':case['label'],'actual':actual,'expected':expected})
        report.update(composition=stats,verifiedLogits=comparisons,mismatches=mismatches,
            workingExpressionParity=bool(comparisons and not mismatches),logits=len(program.outputs),
            descriptorSha256=hashlib.sha256((root/'scalar-program.json').read_bytes()).hexdigest())
        if not comparisons:report['stop']='No matching reference cases'
        if mismatches:report['stop']='Scalar working-expression bit parity failed'
    except Exception as error:report['stop']=str(error)
    finally:signal.alarm(0)
    report['seconds']=time.monotonic()-started;write_json(root/'result.json',report)
    return int(bool(report.get('stop')))


def run(args):
    if args.unit_length:return unit(args)
    import psutil
    config=json.loads(Path(args.checkpoint,'config.json').read_text())
    context=config['max_position_embeddings'];width=config['hidden_size'];vocab=config['vocab_size']
    lengths=list(range(1,context+1)) if not args.lengths else [int(v) for v in args.lengths.split(',')]
    if len(set(lengths))!=len(lengths) or any(not 1<=n<=context for n in lengths):raise ValueError('Invalid discovered lengths')
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    reference=Path(args.reference).resolve() if args.reference else root/'reference.json'
    if not args.reference:
        subprocess.run([sys.executable,str(Path(__file__).with_name('capture_direct_json_reference.py')),args.checkpoint,str(reference),'--all-lengths'],check=True,capture_output=True)
    summary={'strategy':'pairwise-defined-scalar-strings','outputScope':'last-position-only',
        'maxAccumulatedExpressionBytes':args.max_characters,'ramBudgetBytes':args.memory_mib*1024**2,
        'results':[],'complete':False,'finalArtifactParity':False}
    for length in lengths:
        directory=root/f'length-{length}';directory.mkdir(exist_ok=True)
        command=[sys.executable,str(Path(__file__).with_name('direct_sympy_architecture_run.py')),args.checkpoint,str(directory),
            '--unit-length',str(length),'--reference',str(reference),'--workers',str(args.workers),
            '--memory-mib',str(args.memory_mib),'--max-characters',str(args.max_characters),
            '--seconds-per-length',str(args.seconds_per_length)]
        started=time.monotonic();peak=0;reason=None
        with (directory/'worker.log').open('w') as log:
            process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            observed=psutil.Process(process.pid)
            try:
                while process.poll() is None:
                    try:
                        rss=observed.memory_info().rss
                        for child in observed.children(recursive=True):
                            try:rss+=child.memory_info().rss
                            except psutil.Error:pass
                        peak=max(peak,rss)
                    except psutil.Error:pass
                    if peak>args.memory_mib*1024**2:reason='Aggregate RAM budget exceeded'
                    elif time.monotonic()-started>args.seconds_per_length+10:reason='Coordinator deadline exceeded'
                    if reason:os.killpg(process.pid,signal.SIGKILL);break
                    time.sleep(.1)
                code=process.wait()
            finally:
                if process.poll() is None:os.killpg(process.pid,signal.SIGKILL);process.wait()
        path=directory/'result.json'
        row=json.loads(path.read_text()) if path.exists() else {'length':length,'workingExpressionParity':False}
        row.update(exitCode=code,wallSeconds=time.monotonic()-started,peakObservedAggregateRSSBytes=peak)
        if reason:row['coordinatorStop']=reason
        summary['results'].append(row);write_json(root/'summary.json',summary)
        print(json.dumps({key:row.get(key) for key in ['length','verifiedLogits','workingExpressionParity','wallSeconds','stop','coordinatorStop']}),flush=True)
    ready=set(lengths)==set(range(1,context+1)) and all(row.get('workingExpressionParity') and row['exitCode']==0 for row in summary['results'])
    if ready:
        from direct_sympy_scalar import ScalarProgram
        from direct_sympy_scalar_rust import write_scalar_rust_candidate
        from direct_sympy_rust import validate_rust_candidate
        from direct_sympy_strings import StringCompiler
        programs={};certificates={}
        for length in lengths:
            row=json.loads((root/f'length-{length}'/'scalar-program.json').read_text())
            programs[length]=ScalarProgram(tuple(row['inputs']),tuple(tuple(pair) for pair in row['definitions']),tuple(row['outputs'])).validate()
            certificates[length]=row['certificate']
        try:
            compiler=StringCompiler(max_characters=args.max_characters);candidate=root/'next-token.candidate.rs'
            summary['rustEmission']=write_scalar_rust_candidate(candidate,programs,certificates,width=width,vocab=vocab,context=context,compiler=compiler,max_bytes=args.max_characters)
            summary['rustParity']=validate_rust_candidate(candidate,json.loads(reference.read_text()),width=width,vocab=vocab,context=context,timeout=max(60,args.lower_seconds))
            candidate.replace(root/'next-token.rs')
            summary.update(complete=True,finalArtifactParity=True,finalArtifact='next-token.rs',
                sha256=hashlib.sha256((root/'next-token.rs').read_bytes()).hexdigest())
        except Exception as error:summary['rustStop']=str(error)
    write_json(root/'summary.json',summary)
    return int(not summary['complete'])
