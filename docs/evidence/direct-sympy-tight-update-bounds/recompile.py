"""Recompile old cover geometry and wider update boxes, never old proofs."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path('helpers').resolve()))
from direct_sympy_cover_regions import cover,live_identity
from direct_sympy_partition_run import audit_tree,encode,replan
from direct_sympy_checkpoint import CheckpointStrings
from direct_sympy_input_partitions import interval
from direct_sympy_strings import StringCompiler
from direct_sympy_savepoints import atomic,canonical

checkpoint='docs/evidence/direct-sympy-test-checkpoint'
old=json.loads(Path('artifacts/direct-sympy-input-partitions/dominant-square-state/frontier.json').read_text())
directory=Path('artifacts/direct-sympy-input-partitions/tight-update-state');directory.mkdir(parents=True,exist_ok=True)
manifest=directory/'frontier.json';identity=live_identity(checkpoint,2)
root=encode(CheckpointStrings(checkpoint,StringCompiler()).domains)
if not manifest.exists():
    state=replan(old,identity,root)
    state['geometryReplan']={'source':'dominant-square-state/frontier.json','numericalResultsReused':False,
        'completeLeavesReset':sum(node['status']=='complete' for node in old['tree'].values())}
    atomic(manifest,canonical(state))
else:
    state=json.loads(manifest.read_text())
    if state['identity']!=identity:raise ValueError('New proof state identity changed')
    audit_tree(state['tree'],root)
rectangles=[]
for a,b in ((-1,-1),(-1,1),(1,-1),(1,1)):
    rectangles.append(encode({'X1':interval(-65504,-8) if a<0 else interval(8,65504),
        'X2':interval(-65504,-4) if b<0 else interval(4,65504)}))
rectangles.extend(report['inputDomains'] for report in old.get('coveragePromotions',[]))
rectangles.extend(node['domains'] for node in old['tree'].values() if node['status']=='complete')
results=[]
for rectangle in rectangles:
    report=cover(checkpoint,directory,rectangle,max_seconds=30,random_cases=8192)
    results.append(report)
    atomic(directory/'recompilation.json',json.dumps(results,indent=2).encode())
    print(json.dumps({'domain':rectangle,'added':report['addedInputPatterns'],
        'covered':report.get('coveredAfter'),'stop':report.get('stop'),
        'nativeCases':report.get('parity',{}).get('cases'),'mismatches':report.get('parity',{}).get('mismatches')}),flush=True)
