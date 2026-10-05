"""Execute full-architecture composition for the final-position logits at every discovered length.

The coordinator bounds complete worker trees by RAM and time. Working literals
are distinguished from final primitive-free artifacts, and every emitted logit
is reread and compared to an independently captured native reference.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def write_json(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def unit(args):
    import torch
    torch.set_num_threads(1)
    from direct_sympy_architecture import architecture_plan,compose_architecture,prune_plan,last_position_indices
    from direct_sympy_architecture_test import program,numeric_functions
    from direct_sympy_checkpoint import write_expression
    from direct_sympy_parallel import ParallelBudget
    from direct_sympy_strings import StringCompiler
    import struct
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    report={'length':args.unit_length,'workingExpressionsEmitted':0,'verifiedLogits':0,'mismatches':[],
        'finalArtifactEmitted':False,'finalArtifactParity':False}
    started=time.monotonic()
    def stop(*_):raise TimeoutError('Whole-architecture worker deadline exceeded')
    signal.signal(signal.SIGALRM,stop);signal.alarm(args.seconds_per_length)
    try:
        plan=architecture_plan(args.checkpoint,args.unit_length,max_characters=args.max_characters)
        indices=last_position_indices(plan)
        pruned=prune_plan(plan,indices)
        write_json(root/'plan.json',{'length':plan.length,'width':plan.width,'vocab':plan.vocab,
            'stageNames':plan.names,'weightsRead':plan.weight_reads,'operatorCount':len(plan.blocks),'selectedOutputIndices':indices,
            'originalComputedOutputs':sum(value not in b.inputs for b in plan.blocks for value in b.outputs),
            'reachableComputedOutputs':sum(value not in b.inputs for b in pruned for value in b.outputs),
            'originalInterfaceOutputSlots':sum(len(b.outputs) for b in plan.blocks),
            'reachableInterfaceOutputSlots':sum(len(b.outputs) for b in pruned)})
        def progress(stats):
            write_json(root/'composition-progress.json',stats)
        outputs,stats=compose_architecture(plan,StringCompiler(max_characters=args.max_characters),ParallelBudget(args.workers,args.memory_mib*1024**2),output_indices=indices,on_wave=progress)
        if len(outputs)!=plan.vocab:raise ValueError('Final-position logit vector is incomplete')
        report['composition']=stats
        functions=numeric_functions();codes=[];records=[]
        for index,text in enumerate(outputs):
            row,coordinate=divmod(indices[index],plan.vocab)
            target=root/f'position-{row}-logit-{coordinate}.work.expr'
            write_expression(target,text)
            restored=target.read_text().strip();codes.append(program(restored))
            records.append({'position':row,'coordinate':coordinate,'path':target.name,'characters':len(text),
                'sha256':hashlib.sha256(target.read_bytes()).hexdigest()})
        report['workingExpressionsEmitted']=len(outputs);report['expressions']=records
        corpus=json.loads(Path(args.reference).read_text())
        for case in corpus['cases']:
            if len(case['inputBits'])!=plan.length:continue
            values={f'X{i+1}':struct.unpack('e',struct.pack('H',bits))[0] for i,bits in enumerate(bit for row in case['inputBits'] for bit in row)}
            expected=case['logitF64Bits'][-1]
            for index,code in enumerate(codes):
                actual='0x'+struct.pack('>d',eval(code,{'__builtins__':{},**functions},values)).hex()
                report['verifiedLogits']+=1
                if actual!=expected[index]:report['mismatches'].append({'case':case['label'],'position':plan.length-1,'coordinate':index,'actual':actual,'expected':expected[index]})
        report['workingExpressionParity']=not report['mismatches']
        write_json(root/'result.json',report)
        if args.lower:
            # A closure failure is recorded per coordinate, never relabelled
            # as final parity or used to suppress the other output attempts.
            from direct_sympy_conversions import ConversionSession
            from direct_sympy_strings import Domain
            from fractions import Fraction as F
            import re
            import traceback
            domains={f'X{i+1}':Domain(F(-65504),F(65504),-24,False) for i in range(plan.length*plan.width)}
            report['lowering']=[]
            for index,text in enumerate(outputs):
                signal.alarm(args.lower_seconds)
                entry={'position':plan.length-1,'coordinate':index,'complete':False}
                session=ConversionSession(StringCompiler(max_characters=args.max_characters),domains,input_dtype='f16')
                try:
                    closed=session.close(text)
                    pending=sorted(set(re.findall(r'\b(R16|R32|sqrt|Silu16|Exp32)\s*\(',closed)))
                    entry['pendingPrimitives']=pending
                    if not pending:
                        # Closure alone cannot admit a final result: final
                        # scalar evaluator/bit-parity gate is still required.
                        destination=root/f'position-{entry["position"]}-logit-{entry["coordinate"]}.candidate.expr'
                        write_expression(destination,closed);entry['candidate']=destination.name
                        from direct_sympy_rust import emit_scalar
                        emit_scalar(closed,plan.length*plan.width)
                        entry['complete']=True
                    else:entry['stop']='Residual numerical primitives'
                except (ValueError,TimeoutError,RecursionError,MemoryError) as error:
                    entry['stop']=str(error) or 'Numeric closure memory budget exhausted'
                    entry['failureFrames']=[{'file':Path(frame.f_code.co_filename).name,'function':frame.f_code.co_name,'line':line}
                        for frame,line in traceback.walk_tb(error.__traceback__)][-10:]
                entry['numericClosure']={'closedOperations':session.closed,'factorSimplifyEvents':len(session.compiler.events),
                    'completedLiteralCharacters':sum(map(len,session.closed_literals)),
                    'registeredRegionCharacters':session.compiler._region_characters}
                report['lowering'].append(entry)
                write_json(root/'result.json',report)
                del session
            signal.alarm(0)
    except (ValueError,TimeoutError,RecursionError) as error:report['stop']=str(error)
    finally:
        signal.alarm(0);report['seconds']=time.monotonic()-started
        write_json(root/'result.json',report)
    return int(bool(report.get('stop') or report['mismatches'] or
        args.lower and any(not row['complete'] for row in report.get('lowering',[]))))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint');parser.add_argument('output')
    parser.add_argument('--workers',type=int,default=2);parser.add_argument('--memory-mib',type=int,default=4096)
    parser.add_argument('--max-characters',type=int,default=2*1024**3)
    parser.add_argument('--seconds-per-length',type=int,default=60)
    parser.add_argument('--lengths',help='Comma-separated lengths; default every discovered length')
    parser.add_argument('--lower',action='store_true');parser.add_argument('--lower-seconds',type=int,default=15)
    parser.add_argument('--unit-length',type=int);parser.add_argument('--reference')
    args=parser.parse_args()
    if min(args.workers,args.memory_mib,args.max_characters,args.seconds_per_length,args.lower_seconds)<1:parser.error('Positive budgets required')
    if args.unit_length:return unit(args)
    import psutil
    config=json.loads(Path(args.checkpoint,'config.json').read_text())
    lengths=list(range(1,config['max_position_embeddings']+1)) if not args.lengths else [int(v) for v in args.lengths.split(',')]
    if len(set(lengths))!=len(lengths) or any(not 1<=n<=config['max_position_embeddings'] for n in lengths):parser.error('Invalid discovered sequence lengths')
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    reference=Path(args.reference).resolve() if args.reference else root/'reference.json'
    if not args.reference:
        subprocess.run([sys.executable,str(Path(__file__).with_name('capture_direct_json_reference.py')),args.checkpoint,str(reference),'--all-lengths'],check=True,capture_output=True)
    summary={'lengths':lengths,'vocab':config['vocab_size'],'outputScope':'last-position-only','results':[],'complete':False,'finalArtifactParity':False}
    for length in lengths:
        directory=root/f'length-{length}';directory.mkdir(exist_ok=True)
        command=[sys.executable,str(Path(__file__).resolve()),args.checkpoint,str(directory),'--unit-length',str(length),
            '--reference',str(reference),'--workers',str(args.workers),'--memory-mib',str(args.memory_mib),
            '--max-characters',str(args.max_characters),'--seconds-per-length',str(args.seconds_per_length),'--lower-seconds',str(args.lower_seconds)]
        if args.lower:command.append('--lower')
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
                    allowance=args.seconds_per_length+10+(config['vocab_size']*(args.lower_seconds+2) if args.lower else 0)
                    if peak>args.memory_mib*1024**2:reason='Aggregate RAM budget exceeded'
                    elif time.monotonic()-started>allowance:reason='Coordinator deadline exceeded'
                    if reason:
                        os.killpg(process.pid,signal.SIGKILL);break
                    time.sleep(.1)
                code=process.wait()
            finally:
                if process.poll() is None:
                    os.killpg(process.pid,signal.SIGKILL);process.wait()
        path=directory/'result.json'
        result=json.loads(path.read_text()) if path.exists() else {'length':length,'workingExpressionsEmitted':0,'finalArtifactEmitted':False,'finalArtifactParity':False}
        progress_path=directory/'composition-progress.json'
        if progress_path.exists() and 'composition' not in result:result['composition']=json.loads(progress_path.read_text())
        plan_path=directory/'plan.json'
        if plan_path.exists():result['plan']=json.loads(plan_path.read_text())
        result.update({'exitCode':code,'wallSeconds':time.monotonic()-started,'peakObservedAggregateRSSBytes':peak})
        if reason:result['coordinatorStop']=reason
        summary['results'].append(result);write_json(root/'summary.json',summary)
        print(json.dumps({key:result.get(key) for key in ('length','workingExpressionsEmitted','verifiedLogits','workingExpressionParity','stop','coordinatorStop','wallSeconds','peakObservedAggregateRSSBytes')}),flush=True)
    summary['workingExpressionsComplete']=all(r.get('workingExpressionsEmitted')==config['vocab_size'] and r.get('workingExpressionParity') for r in summary['results'])
    summary['complete']=all(r.get('finalArtifactEmitted') and r.get('finalArtifactParity') for r in summary['results'])
    if args.lower and set(lengths)==set(range(1,config['max_position_embeddings']+1)):
        rows=summary['results']
        ready=all(len(r.get('lowering',[]))==config['vocab_size'] and all(e.get('complete') for e in r['lowering']) for r in rows)
        if ready:
            from direct_sympy_rust import write_rust_candidate,validate_rust_candidate
            try:
                bodies={r['length']:[(root/f'length-{r["length"]}'/e['candidate']).read_text().strip() for e in r['lowering']] for r in rows}
                candidate=root/'next-token.candidate.rs'
                summary['rustEmission']=write_rust_candidate(candidate,bodies,width=config['hidden_size'],vocab=config['vocab_size'],context=config['max_position_embeddings'],max_bytes=args.max_characters)
                summary['rustParity']=validate_rust_candidate(candidate,json.loads(reference.read_text()),width=config['hidden_size'],vocab=config['vocab_size'],context=config['max_position_embeddings'])
                candidate.replace(root/'next-token.rs')
                summary.update(complete=True,finalArtifactParity=True,finalArtifact='next-token.rs')
            except (ValueError,OSError,TimeoutError) as error:
                summary['rustStop']=str(error)
    write_json(root/'summary.json',summary)
    return int(bool(args.lower and not summary['complete']) or any(r.get('stop') or r.get('coordinatorStop') or r.get('mismatches') for r in summary['results']))


if __name__=='__main__':raise SystemExit(main())
