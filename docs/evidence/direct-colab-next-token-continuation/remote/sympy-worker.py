import os,sys,json,time,signal,re,struct,resource
from pathlib import Path
root=Path('/content/llm-inner-final-logits-eligible-candidates');os.chdir(root);sys.path.insert(0,str(root/'helpers'))
import torch
torch.set_num_threads(1)
from direct_sympy_architecture import architecture_plan,compose_architecture,prune_plan,last_position_indices
from direct_sympy_parallel import ParallelBudget
from direct_sympy_strings import StringCompiler,Domain
from direct_sympy_architecture_test import program,numeric_functions
from direct_sympy_conversions import ConversionSession
from fractions import Fraction as F
report={'outputScope':'last-position-only','coordinate':0,'length':1,'artifactComplete':False,'finalParity':False,'compilations':[]};start=time.monotonic()
def record():
 report['seconds']=time.monotonic()-start;report['peakRSSBytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
 (root/'sympy-result.json').write_text(json.dumps(report,indent=2))
def deadline(*args):raise TimeoutError('CPU composition/lowering deadline reached')
signal.signal(signal.SIGALRM,deadline)
try:
 plan=architecture_plan(str(root/'docs/evidence/direct-sympy-test-checkpoint'),1,max_characters=512*1024**2)
 selected=(last_position_indices(plan)[0],);pruned=prune_plan(plan,selected)
 report['selectedOutputIndices']=selected;report['reachableProducers']=sum(v not in b.inputs for b in pruned for v in b.outputs)
 outputs=None;reference=json.loads((root/'docs/evidence/direct-sympy-envelope-sync-reference.json').read_text());cases=[c for c in reference['cases'] if len(c['inputBits'])==1]
 for workers in [1,2]:
  compiler=StringCompiler(max_characters=512*1024**2);t=time.monotonic();signal.alarm(90)
  def progress(stats):
   (root/f'sympy-wave-{workers}.json').write_text(json.dumps(stats,indent=2))
  outputs,stats=compose_architecture(plan,compiler,ParallelBudget(workers,4*1024**3),output_indices=selected,on_wave=progress)
  code=program(outputs[0]);mismatches=[]
  for case in cases:
   values={f'X{i+1}':struct.unpack('e',struct.pack('H',bits))[0] for i,bits in enumerate(case['inputBits'][0])}
   value=eval(code,{'__builtins__':{},**numeric_functions()},values);actual='0x'+struct.pack('>d',value).hex();expected=case['logitF64Bits'][-1][0]
   if actual!=expected:mismatches.append({'case':case['label'],'actual':actual,'expected':expected})
  report['compilations'].append({'workers':workers,'seconds':time.monotonic()-t,'characters':len(outputs[0]),'stats':stats,'comparisons':len(cases),'mismatches':mismatches});record()
  if mismatches:raise ValueError('Working-coordinate parity diverged')
 signal.alarm(120);t=time.monotonic()
 domains={f'X{i+1}':Domain(F(-65504),F(65504),-24,False) for i in range(plan.width)}
 session=ConversionSession(compiler,domains,input_dtype='f16');closed=session.close(outputs[0]);signal.alarm(0)
 pending=sorted(set(re.findall(r'\b(R16|R32|sqrt|Silu16|Exp32|CASNumericRegion[0-9]+|CompileValue[0-9]+)\s*\(',closed)))
 report['lowering']={'seconds':time.monotonic()-t,'characters':len(closed),'pending':pending}
 (root/'first-coordinate.work.expr').write_text(closed+'\n')
 if pending:raise ValueError('Residual primitives remain; candidate not admitted as final')
 # Artifact parity must follow a complete primitive audit and literal reread.
 report['primitiveClosure']=True
except Exception as error:report['stop']=str(error)
finally:signal.alarm(0);record();print(json.dumps({k:v for k,v in report.items() if k!='compilations'}),flush=True)
