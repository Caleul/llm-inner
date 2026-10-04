from pathlib import Path
import hashlib,json,tarfile
root=Path('/content/llm-inner-input-partitions')
report=json.loads((root/'comparison-job.json').read_text())
if report['state']!='COMPLETED' or not report['identicalArtifacts']:raise RuntimeError('Comparison not terminal and identical')
for row in report['runs']:
 p=Path(row['artifact']['path'])
 if hashlib.sha256(p.read_bytes()).hexdigest()!=row['artifact']['sha256']:raise RuntimeError('Artifact identity mismatch')
archive=root/'comparison-evidence.tar.gz'
with tarfile.open(archive,'w:gz') as stream:
 for name in ('comparison-job.json','comparison-preparation.json','compare.py','prepare_comparison.py','comparison-launch.log','preparation.log'):
  stream.add(root/name,arcname=name)
 stream.add(root/'results',arcname='results')
print(json.dumps({'path':str(archive),'bytes':archive.stat().st_size,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest()}))
