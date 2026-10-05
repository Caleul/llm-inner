"""Create the inspected source/checkpoint archive; never contact Colab."""
import gzip,hashlib,json,tarfile
from pathlib import Path
root=Path.cwd();folder=Path(__file__).resolve().parent
path=Path('/private/tmp/llm-inner-h100-512-20261005.tar.gz')
sources=sorted((root/'helpers').glob('*.py'))+sorted((root/'docs/evidence/direct-sympy-test-checkpoint').iterdir())
with path.open('wb')as output:
 with gzip.GzipFile(fileobj=output,mode='wb',mtime=0,filename='')as compressed:
  with tarfile.open(fileobj=compressed,mode='w')as archive:
   for source in sources:
    if not source.is_file():continue
    member=archive.gettarinfo(str(source),arcname=str(source.relative_to(root)))
    member.mtime=0;member.uid=member.gid=0;member.uname=member.gname=''
    with source.open('rb')as stream:archive.addfile(member,stream)
manifest={'path':str(path),'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
 'executionDispatched':False,'requestedGPU':'H100','expressionAndConditionBudgetBytes':512*1024**2}
(folder/'source-archive.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest))
