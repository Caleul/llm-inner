from pathlib import Path
import hashlib,json,statistics,sys,time
from unittest.mock import patch
sys.path.insert(0,str(Path('helpers').resolve()))
import direct_sympy_checkpoint as checkpoint
from direct_sympy_partition_run import _compile_region,decode
root=Path(__file__).resolve().parent
region={'X1':[-16181,-16120],'X2':[-27775,-23808]}
rows=[]
original=checkpoint.rms_component_enclosures
for index,mode in enumerate(['previous','correlated','correlated','previous','previous','correlated']):
    path=root/f'{index}-{mode}.expr'
    with patch.object(checkpoint,'rms_component_enclosures',original if mode=='correlated' else lambda *args:None):
        report=_compile_region('docs/evidence/direct-sympy-test-checkpoint',2,decode(region),path,max_characters=1048576,cas_characters=8388608,max_paths=64)
    if not report['complete']:raise ValueError(report)
    payload=path.read_bytes();sha=hashlib.sha256(payload).hexdigest()
    expected='f185e8fc560b69916e9ee164c81ba7a1d51acb4a044aafd4e6143a955bfe43a3' if mode=='correlated' else '2c5b7ffd0fe16faa4a070781d08692dd354c92cb34e02ee796e7260239e80e92'
    if sha!=expected:raise ValueError('Unexpected artifact '+sha)
    rows.append({'mode':mode,'report':report,'sha256':sha})
    (root/('coordinate.expr' if mode=='correlated' else 'previous-coordinate.expr')).write_bytes(payload)
    path.unlink()
    print(index,mode,report['seconds'],sha,flush=True)
medians={mode:statistics.median(row['report']['seconds'] for row in rows if row['mode']==mode) for mode in ['previous','correlated']}
result={'complete':True,'region':region,'rows':rows,'medians':medians,'timeSavedPercent':100*(1-medians['correlated']/medians['previous']),'sizeSavedPercent':100*(1-39787/430300),'sourceSHA256':hashlib.sha256(Path('helpers/direct_sympy_checkpoint.py').read_bytes()).hexdigest(),'scope':'Six alternating in-process trials; same region, position zero, dimension two, one token. No isolated memory measurement or full-domain result.'}
(root/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
