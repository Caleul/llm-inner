"""Recover geometry only, then compile and validate every numerical result."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path('helpers').resolve()))
from direct_sympy_cover_regions import cover,live_identity
from direct_sympy_partition_run import audit_tree,encode,replan
from direct_sympy_checkpoint import CheckpointStrings
from direct_sympy_strings import StringCompiler
from direct_sympy_savepoints import atomic,canonical

checkpoint='docs/evidence/direct-sympy-test-checkpoint'
previous=Path('docs/evidence/direct-sympy-early-projection/frontier.json')
old=json.loads(previous.read_text())
directory=Path('artifacts/direct-sympy-input-partitions/coupled-projection-state')
directory.mkdir(parents=True,exist_ok=True);manifest=directory/'frontier.json'
identity=live_identity(checkpoint,2)
root=encode(CheckpointStrings(checkpoint,StringCompiler()).domains)
if not manifest.exists():
    state=replan(old,identity,root)
    state['geometryReplan']={'source':str(previous),'numericalResultsReused':False,
        'completeLeavesReset':sum(n['status']=='complete' for n in old['tree'].values())}
    atomic(manifest,canonical(state))
else:
    state=json.loads(manifest.read_text())
    if state['identity']!=identity:raise ValueError('Numerical identity changed; no saved result reused')
    audit_tree(state['tree'],root)

# Former admissions provide input geometry, never expressions or proofs.
rectangles=[p['inputDomains'] for p in old.get('coveragePromotions',[])]
rectangles.extend(n['domains'] for n in old['tree'].values() if n['status']=='complete')
rectangles.append({'X1':[11264,16384],'X2':[11264,15360]})
for rectangle in rectangles:
    central=rectangle==rectangles[-1]
    result=cover(checkpoint,directory,rectangle,max_seconds=120 if central else 30,
        max_characters=8388608 if central else 1048576,random_cases=8192)
    print(json.dumps({'domain':rectangle,'added':result['addedInputPatterns'],
        'covered':result['coveredAfter'],'stop':result.get('stop'),
        'nativeCases':result.get('parity',{}).get('cases'),
        'mismatches':result.get('parity',{}).get('mismatches')}),flush=True)
    if 'stop' in result and result['stop']!='Region already covered; no manifest mutation':
        raise ValueError('Fresh geometry could not be recovered: '+result['stop'])
