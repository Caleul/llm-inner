"""Recompile audited geometry, then compile both enlarged mixed sign regions."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path('helpers').resolve()))
from direct_sympy_cover_regions import cover,live_identity
from direct_sympy_partition_run import audit_tree,encode,replan
from direct_sympy_checkpoint import CheckpointStrings
from direct_sympy_strings import StringCompiler
from direct_sympy_savepoints import atomic,canonical

checkpoint='docs/evidence/direct-sympy-test-checkpoint'
previous=Path('docs/evidence/direct-sympy-norm-correlation/frontier.json')
old=json.loads(previous.read_text())
directory=Path('artifacts/direct-sympy-input-partitions/directed-residual-state')
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
    if state['identity']!=identity:raise ValueError('Numerical identity changed; no expression reused')
    audit_tree(state['tree'],root)
rectangles=[]
for x,y in ((-1,1),(1,-1)):
    rectangles.append(({'X1':[-16384,-10240] if x<0 else [10240,16384],
        'X2':[-15360,-10241] if y<0 else [10241,15360]},
        {'seconds':180,'artifactCharacters':16777216,'CASCharacters':16777216,'paths':128}))
rectangles.extend((p['inputDomains'],p['budgets']) for p in old['coveragePromotions'])
for rectangle,budgets in rectangles:
    result=cover(checkpoint,directory,rectangle,max_seconds=budgets['seconds'],
        max_characters=budgets['artifactCharacters'],cas_characters=budgets['CASCharacters'],
        max_paths=budgets['paths'],random_cases=8192)
    print(json.dumps({'domain':rectangle,'added':result['addedInputPatterns'],
        'covered':result['coveredAfter'],'characters':result.get('sharedArtifact',{}).get('characters'),
        'sha256':result.get('sharedArtifact',{}).get('sha256'),'stop':result.get('stop'),
        'nativeCases':result.get('parity',{}).get('cases'),'mismatches':result.get('parity',{}).get('mismatches')}),flush=True)
    if result.get('stop') not in (None,'Region already covered; no manifest mutation'):
        raise ValueError('Fresh cover failed: '+result['stop'])
