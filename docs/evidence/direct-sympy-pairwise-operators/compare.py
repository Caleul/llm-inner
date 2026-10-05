import json,subprocess,time,sys
from pathlib import Path
import psutil
root=Path(__file__).resolve().parents[3]
out=root/'docs/evidence/direct-sympy-pairwise-operators'
results=[]
for workers,label in [(1,'sequential'),(2,'paired')]:
 command=[sys.executable,'helpers/direct_sympy_checkpoint.py','docs/evidence/direct-sympy-test-checkpoint',str(out/label),'--workers',str(workers),'--memory-mib','4096','--max-characters','536870912','--max-seconds','45']
 start=time.monotonic();peak=0
 with (out/(label+'.log')).open('w') as log:
  process=subprocess.Popen(command,cwd=root,stdout=log,stderr=subprocess.STDOUT)
  observed=psutil.Process(process.pid)
  while process.poll() is None:
   try:
    children=observed.children(recursive=True)
    total=observed.memory_info().rss
    for child in children:
     try:total+=child.memory_info().rss
     except psutil.Error:pass
    peak=max(peak,total)
   except psutil.Error:pass
   time.sleep(.1)
  code=process.wait()
 growth=(out/(label+'.growth.tsv')).read_text().splitlines()[1:]
 records=json.loads((out/(label+'.parallel.json')).read_text())
 results.append({'label':label,'workers':workers,'wallSeconds':time.monotonic()-start,'peakAggregateRSSBytes':peak,'exitCode':code,'completedDistinctProducers':len(growth),'lastProducer':growth[-1].split('\t')[0] if growth else None,'operatorPairMerges':sum(r.get('operatorPairMerges',0) for r in records),'finalArtifactEmitted':False})
 print(json.dumps(results[-1]),flush=True)
 # These prefixes are incomplete, not final artifacts; keep only their hashes
 # in the final evidence index rather than committing ten-megabyte strings.
(out/'comparison.json').write_text(json.dumps(results,indent=2)+'\n')
