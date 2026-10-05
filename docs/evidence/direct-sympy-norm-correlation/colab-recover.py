"""Validate the existing Colab artifact after increasing Clang parser depth."""
import json,os,subprocess,sys,time,traceback
from pathlib import Path
root=Path('/content/llm-inner-norm-correlation');os.chdir(root)
sys.path.insert(0,str(root/'helpers'))
from direct_sympy_cover_regions import live_identity
from direct_sympy_input_partitions import interval
from direct_sympy_region_parity import verify_region
from direct_sympy_savepoints import digest_file
import torch
torch.set_num_threads(1)
out=root/'results';old=json.loads((out/'report.json').read_text())
checkpoint=str(root/'docs/evidence/direct-sympy-test-checkpoint')
if not old['terminal']:raise RuntimeError('Original driver has not terminated; no recovery dispatched')
if old['compilerIdentity']!=live_identity(checkpoint,2):raise RuntimeError('Numerical identity changed')
artifact=out/'mixed.expr'
if digest_file(artifact)!=old['mixedCompilation']['artifact']['sha256']:raise RuntimeError('Artifact changed')
report={'terminal':False,'compilerIdentity':old['compilerIdentity'],'originalFailure':'report.json',
    'artifactSHA256':digest_file(artifact),'recompiled':False,'fullCoordinateParity':False,'exhaustive':[]}
def save():
    p=out/'recovery.tmp';p.write_text(json.dumps(report,indent=2)+'\n');p.replace(out/'recovery.json')
try:
    save();report['corpusParity']=verify_region(checkpoint,2,artifact,{'X1':interval(-2,-.0625),'X2':interval(.0625,1)},random_cases=8192);save()
    if report['corpusParity']['mismatches']:raise RuntimeError('Actual-file corpus parity failed')
    for x,y in ((-1,1),(1,-1)):
        name='exhaustive-'+str(x)+'-'+str(y);started=time.monotonic()
        with (out/(name+'.log')).open('w') as log:
            result=subprocess.run([sys.executable,'docs/evidence/direct-sympy-norm-correlation/exhaustive.py',
                '--artifact',str(artifact),'--x-sign',str(x),'--y-sign',str(y),'--report',str(out/(name+'.json'))],stdout=log,stderr=subprocess.STDOUT)
        report['exhaustive'].append({'report':name+'.json','exitCode':result.returncode,'seconds':time.monotonic()-started});save()
        if result.returncode:raise RuntimeError('Exhaustive actual-file parity failed')
    report['terminal']=True;report['success']=True;save()
except BaseException:
    report['terminal']=True;report['success']=False;report['error']=traceback.format_exc();save();raise
