import json,subprocess,time,os,psutil
from pathlib import Path
root=Path('/content/llm-inner-final-logits-eligible-candidates');node=str(root/'node-v22.22.2-linux-x64/bin/node')
results=[]
for label,args in [('cofactor-final',['--test','dist/test/direct-json-cofactor-rounds.test.js','dist/test/direct-json-cofactor-parallel.test.js']),('native-final',['--test','--test-name-pattern=pipeline reports|refuses emission','dist/test/direct-json-model.test.js'])]:
 t=time.monotonic();peak=0
 with (root/(label+'.log')).open('w') as log:
  p=subprocess.Popen([node,*args],cwd=root,env={**os.environ,'LLM_INNER_DIRECT_PYTHON':'unused-target-snapshot','LLM_INNER_DIRECT_JSON_CHECKPOINT':str(root/'docs/evidence/direct-sympy-test-checkpoint'),'LLM_INNER_DIRECT_JSON_SNAPSHOT':str(root/'target-discovery.json')},stdout=log,stderr=subprocess.STDOUT)
  while p.poll() is None:
   try:q=psutil.Process(p.pid);peak=max(peak,sum(x.memory_info().rss for x in [q,*q.children(recursive=True)]))
   except psutil.NoSuchProcess:pass
   time.sleep(.2)
 row={'label':label,'exitCode':p.returncode,'seconds':time.monotonic()-t,'peakTreeRSSBytes':peak};results.append(row)
 (root/'regression-result.json').write_text(json.dumps(results,indent=2))
 if p.returncode:break
