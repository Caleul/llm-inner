from pathlib import Path
import ast,hashlib,json,statistics,sys,time
from unittest.mock import patch
sys.path.insert(0,str(Path('helpers').resolve()))
import direct_sympy_words as words
from direct_sympy_partition_run import _compile_region,decode

class PrintedSignatures:
    def invalidate(self,node):pass
    def key(self,node):return ast.dump(node)

root=Path(__file__).resolve().parent
region={'X1':[-16181,-16120],'X2':[-27775,-23808]}
rows=[]
for index,mode in enumerate(['printed','structural','structural','printed','printed','structural']):
    path=root/f'{index}-{mode}.expr'
    start=time.monotonic()
    with patch.object(words,'StructuralSignatures',PrintedSignatures if mode=='printed' else words.StructuralSignatures):
        report=_compile_region('docs/evidence/direct-sympy-test-checkpoint',2,decode(region),path,max_characters=1048576,cas_characters=8388608,max_paths=64)
    if not report['complete']:raise ValueError(report)
    rows.append({'mode':mode,'seconds':time.monotonic()-start,'report':report,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    (root/'comparison.json').write_text(json.dumps({'complete':False,'rows':rows},indent=2)+'\n')
    print(index,mode,rows[-1]['seconds'],rows[-1]['sha256'],flush=True)
medians={mode:statistics.median(row['seconds'] for row in rows if row['mode']==mode) for mode in ['printed','structural']}
result={'complete':True,'region':region,'rows':rows,'medians':medians,'speedup':medians['printed']/medians['structural'],
    'byteIdentical':len({row['sha256'] for row in rows})==1,
    'sourceSHA256':hashlib.sha256(Path('helpers/direct_sympy_words.py').read_bytes()).hexdigest(),
    'scope':'One emitted position-zero, dimension-two, one-token region; alternating in-process trials; other validation processes may run concurrently. Not full-coordinate speed or parity.'}
if not result['byteIdentical']:raise ValueError('Different emitted expressions')
payload=Path(rows[-1]['report']['artifact']['path']).read_bytes()
(root/'coordinate.expr').write_bytes(payload)
result['durableArtifact']={'file':'coordinate.expr','characters':len(payload),'sha256':hashlib.sha256(payload).hexdigest()}
for row in rows:
    path=Path(row['report']['artifact']['path'])
    if hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:raise ValueError('Trial artifact readback mismatch')
    path.unlink()
result['trialArtifactsRemovedAfterReadback']=True
(root/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
