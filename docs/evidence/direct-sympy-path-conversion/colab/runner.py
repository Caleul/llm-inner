import pathlib,sys,subprocess,json,time,resource
root=pathlib.Path(__file__).parent;report={'runs':[],'coordinateComplete':False}
def run(name,args):
 start=time.monotonic()
 with (root/(name+'.log')).open('w') as log:result=subprocess.run(args,cwd=root,stdout=log,stderr=subprocess.STDOUT)
 report['runs'].append({'name':name,'seconds':time.monotonic()-start,'exitCode':result.returncode,'childrenPeakRSSKiB':resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss});(root/'report.json').write_text(json.dumps(report,indent=2));return result.returncode
sys.path.insert(0,str(root/'helpers'))
from direct_sympy_checkpoint import CheckpointStrings
from direct_sympy_strings import StringCompiler
from direct_sympy_savepoints import ProducerSavepoints
old='/content/llm-inner-envelope-diagnostic/state'
with CheckpointStrings(root/'docs/evidence/direct-sympy-test-checkpoint',StringCompiler(max_characters=8*1024**3)) as model:
 with ProducerSavepoints(old,model,2,compressed=True) as store:
  try:store.restore(model)
  except ValueError as error:
   assert str(error).startswith('Incompatible savepoint:');assert not model.memo
   report['oldStateRejectedBeforeMutation']=True
  else:raise RuntimeError('Changed numerical compiler accepted old state')
for name in ('direct_sympy_path_conversion_test.py','direct_sympy_savepoints_test.py'):
 if run(name,[sys.executable,'helpers/'+name]):raise RuntimeError('Numerical gate failed')
if run('cuda',[sys.executable,'helpers/direct_sympy_cuda_validation.py']):raise RuntimeError('CUDA proof gate failed')
if run('compile',[sys.executable,'helpers/direct_sympy_envelope_diagnostic.py','docs/evidence/direct-sympy-test-checkpoint','state','diagnostic.json','--fresh','--max-characters',str(8*1024**3),'--max-seconds','600']):raise RuntimeError('Diagnostic process failed')
if run('parity',[sys.executable,'helpers/direct_sympy_frontier_validation.py','state','docs/evidence/direct-sympy-test-checkpoint','docs/evidence/direct-sympy-offset-inputs.json']):raise RuntimeError('Saved producer parity failed')
report['terminal']='complete';(root/'report.json').write_text(json.dumps(report,indent=2));print('PATH_CONVERSION_COMPLETE',flush=True)
