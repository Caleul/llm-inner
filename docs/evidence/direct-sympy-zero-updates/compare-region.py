from pathlib import Path
import hashlib,json,subprocess,sys,time
root=Path.cwd();output=root/'docs/evidence/direct-sympy-zero-updates'
rows=[]
for name,helpers in [('before',root/'artifacts/direct-sympy-input-partitions/constant-cells-baseline/helpers'),('after',root/'helpers')]:
 start=time.monotonic();artifact=output/(name+'-small-region.expr')
 args=['/private/tmp/llm-inner-pytorch/bin/python',str(helpers/'direct_sympy_partition_run.py'),'--worker',str(root/'docs/evidence/direct-sympy-test-checkpoint'),str(artifact),'2','1048576','8388608','64',json.dumps({'X1':[-2,2],'X2':[-2,2]})]
 try:p=subprocess.run(args,capture_output=True,text=True,timeout=90)
 except subprocess.TimeoutExpired:rows.append({'version':name,'complete':False,'stop':'90s worker limit'});continue
 (output/(name+'-small-region.log')).write_text(p.stdout+p.stderr)
 if p.returncode:raise RuntimeError(p.stderr[-2000:])
 row=json.loads(p.stdout);row['version']=name;row['wallSeconds']=time.monotonic()-start;rows.append(row)
 if row['complete']:assert hashlib.sha256(artifact.read_bytes()).hexdigest()==row['artifact']['sha256']
(output/'small-region-comparison.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows))
