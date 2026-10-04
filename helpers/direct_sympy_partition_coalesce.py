"""Coalesce identical emitted functions; retain full input-coverage evidence.

Only byte-identical expressions share a body. Mathematical resemblance does
not authorize merging floating results. No incomplete tree emits a final file.
"""
import argparse
import ast
import json
from pathlib import Path

import sympy
from direct_sympy_partition_run import audit_tree,cardinality,decode
from direct_sympy_savepoints import atomic,digest_file
from direct_sympy_strings import cas_view,symbolic,syntax


def groups(directory,tree):
    directory=Path(directory);root=tree['']['domains'];covered,unfinished=audit_tree(tree,root)
    result=[];texts={};copies=0
    for key,node in sorted(tree.items()):
        if node['status']!='complete':continue
        artifact=node['artifact'];path=directory/artifact['file'];digest=digest_file(path)
        if digest!=artifact['sha256']:raise ValueError('Coalesced artifact integrity mismatch')
        text=path.read_text()
        if len(text)!=artifact['characters']:raise ValueError('Coalesced artifact size mismatch')
        if digest in texts and texts[digest]!=text:raise ValueError('Nonidentical expression under shared digest')
        texts[digest]=text;copies+=len(text)
        result.append({'sha256':digest,'domains':{name:list(bounds) for name,bounds in node['domains'].items()},'leaves':[key]})
    # Merge only adjacent rectangles with all other axes exactly equal.
    # Their union is another rectangle, with no added Half input patterns.
    changed=True
    while changed:
        changed=False
        for axis in root:
            buckets={}
            for item in result:
                key=(item['sha256'],tuple((name,tuple(bounds)) for name,bounds in sorted(item['domains'].items()) if name!=axis))
                buckets.setdefault(key,[]).append(item)
            merged=[]
            for bucket in buckets.values():
                bucket.sort(key=lambda item:item['domains'][axis][0])
                current=bucket[0]
                for other in bucket[1:]:
                    if current['domains'][axis][1]+1==other['domains'][axis][0]:
                        before=cardinality(current['domains'])+cardinality(other['domains'])
                        current={'sha256':current['sha256'],'domains':{name:list(bounds) for name,bounds in current['domains'].items()},'leaves':current['leaves']+other['leaves']}
                        current['domains'][axis][1]=other['domains'][axis][1]
                        if cardinality(current['domains'])!=before:raise ValueError('Merged rectangle changes input coverage')
                        changed=True
                    else:merged.append(current);current=other
                merged.append(current)
            result=merged
    if sum(cardinality(item['domains']) for item in result)!=covered:raise ValueError('Coalescing changes complete-domain coverage')
    by_function={}
    for item in result:by_function.setdefault(item['sha256'],[]).append(item)
    summary={'completeRegions':sum(len(item['leaves']) for item in result),'coalescedRectangles':len(result),
        'distinctExpressions':len(texts),'copiedBodyCharacters':copies,'sharedBodyCharacters':sum(map(len,texts.values())),
        'coveredInputPatterns':covered,'unfinishedInputPatterns':unfinished,'finalArtifactEmitted':False,'finalParity':False,
        'rectangles':result}
    return texts,by_function,summary


def emit(directory,tree,path,*,max_characters):
    texts,by_function,report=groups(directory,tree)
    if report['unfinishedInputPatterns']:raise ValueError('Unfinished input domain; no coalesced final artifact')
    pieces=[];characters=10
    for digest,rectangles in sorted(by_function.items()):
        guards=[]
        for rectangle in rectangles:
            conditions=[]
            for name,d in decode(rectangle['domains']).items():
                conditions.extend((f'{name} >= {repr(float(d.minimum))}',f'{name} <= {repr(float(d.maximum))}'))
            guards.append('And('+', '.join(conditions)+')')
        domain_guard=guards[0] if len(guards)==1 else 'Or('+', '.join(guards)+')'
        if len(by_function)==1:domain_guard='True' # Audited full-domain cover.
        expression=syntax(texts[digest])
        arms=expression.args if isinstance(expression,ast.Call) and expression.func.id=='Piecewise' else [ast.Tuple(elts=[expression,ast.Constant(value=True)],ctx=ast.Load())]
        for pair in arms:
            body,numeric_guard=map(ast.unparse,pair.elts)
            guard='And('+domain_guard+', '+numeric_guard+')'
            # Domain guards precede every numeric condition. Keep their
            # original evaluation order even when CAS proposes reordering.
            atom=sympy.Symbol('CoalescedBody',real=True)
            numeric_view,_=cas_view(numeric_guard)
            condition=sympy.And(symbolic(syntax(domain_guard)),symbolic(numeric_view),evaluate=False)
            sympy.simplify(sympy.factor(sympy.Piecewise((atom,condition),evaluate=False)))
            piece='(('+body+'), '+guard+')';characters+=len(piece)+2
            if characters>max_characters:raise ValueError('Coalesced final artifact character budget exceeded')
            pieces.append(piece)
    text='Piecewise('+', '.join(pieces)+')'
    if any(name in text for name in ('CompileValue','CASNumericRegion','R16(','R32(','Silu16(','sqrt(')):
        raise ValueError('Unfinished dependency in coalesced artifact')
    atomic(path,text.encode())
    report.update(finalArtifactEmitted=True,artifact={'path':str(path),'characters':len(text),'sha256':digest_file(path)})
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('state');parser.add_argument('report')
    parser.add_argument('--output');parser.add_argument('--max-characters',type=int,default=64*1024**2)
    parser.add_argument('--checkpoint')
    args=parser.parse_args();directory=Path(args.state);state=json.loads((directory/'frontier.json').read_text())
    for name,digest in state['identity']['sources'].items():
        if digest_file(Path(__file__).parent/name)!=digest:raise ValueError('Incompatible saved compiler source')
    if args.output and not args.checkpoint:parser.error('Final emission requires the reference checkpoint identity')
    if args.checkpoint:
        from direct_sympy_checkpoint import CheckpointStrings
        from direct_sympy_partition_run import encode
        from direct_sympy_strings import StringCompiler
        checkpoint=Path(args.checkpoint)
        identity={p.name:digest_file(p) for p in [checkpoint/'config.json',*sorted(checkpoint.glob('*.safetensors'))]}
        if identity!=state['identity']['checkpoint']:raise ValueError('Coalesced checkpoint identity mismatch')
        if encode(CheckpointStrings(checkpoint,StringCompiler()).domains)!=state['root']:
            raise ValueError('Coalesced root does not cover the full admitted input domain')
    report=emit(directory,state['tree'],args.output,max_characters=args.max_characters) if args.output else groups(directory,state['tree'])[2]
    atomic(args.report,json.dumps(report,indent=2).encode());print(json.dumps({k:v for k,v in report.items() if k!='rectangles'}))


if __name__=='__main__':main()
