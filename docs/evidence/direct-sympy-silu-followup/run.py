"""Fresh targeted broad covers under the unchanged certified SiLU identity."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path('helpers').resolve()))
from direct_sympy_cover_regions import cover,live_identity
from direct_sympy_input_partitions import interval
from direct_sympy_partition_run import encode
from direct_sympy_savepoints import atomic
root=Path(__file__).resolve().parent
state=Path('artifacts/direct-sympy-input-partitions/silu-update-state')
manifest=state/'frontier.json';before=json.loads(manifest.read_text())
identity=live_identity('docs/evidence/direct-sympy-test-checkpoint',2)
assert before['identity']==identity
candidates=[]
# The two broad same-sign rectangles previously exceeded the flat budget.
# Cover larger-magnitude X2 first, keeping every lower band pending.
for sign in (-1,1):
    candidates.append(encode({'X1':interval(-1,-.03125) if sign<0 else interval(.03125,1),
        'X2':interval(-65504,-128) if sign<0 else interval(128,65504)}))
    for a,b in ((.03125,.0625),(.0625,.125)):
        candidates.append(encode({'X1':interval(-b,-a) if sign<0 else interval(a,b),
            'X2':interval(-128,-64) if sign<0 else interval(64,128)}))
reports=[]
for domains in candidates:
    report=cover('docs/evidence/direct-sympy-test-checkpoint',state,domains,max_seconds=30,random_cases=8192)
    reports.append(report);atomic(root/'covers.json',json.dumps(reports,indent=2).encode())
    print(json.dumps({'domain':domains,'added':report['addedInputPatterns'],'covered':report['coveredAfter'],
        'stop':report.get('stop'),'parity':report.get('parity')}),flush=True)
after=json.loads(manifest.read_text());assert after['identity']==identity
summary={'identityUnchanged':True,'coveredBefore':before['coveredInputPatterns'],'coveredAfter':after['coveredInputPatterns'],
    'addedInputPatterns':after['coveredInputPatterns']-before['coveredInputPatterns'],
    'unfinishedInputPatterns':after['unfinishedInputPatterns'],'totalInputPatterns':after['totalInputPatterns'],
    'nativeCases':sum(r.get('parity',{}).get('cases',0) for r in reports),
    'nativeMismatches':sum(r.get('parity',{}).get('mismatches',0) for r in reports),
    'fullCoordinateArtifactEmitted':after['finalArtifactEmitted'],'fullCoordinateParity':after['finalParity']}
atomic(root/'validation.json',json.dumps(summary,indent=2).encode());print(json.dumps(summary),flush=True)
