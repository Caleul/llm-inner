"""Upload archive and prepare helper first; execute this through colab_cli -f."""
import json,os,subprocess,sys
from pathlib import Path
root=Path('/content/llm-inner-h100-512-20261005')
archive=Path('/content/llm-inner-h100-512-20261005.tar.gz')
manifest=json.loads(Path('/content/llm-inner-h100-512-source-archive.json').read_text())
if root.exists():raise RuntimeError('Job already exists; inspect it before any retry')
marker=root.with_suffix('.launch.json')
with marker.open('x')as stream:
 json.dump({'state':'DISPATCHING','archiveSHA256':manifest['sha256']},stream)
with root.with_suffix('.preparation-launch.log').open('w')as log:
 process=subprocess.Popen([sys.executable,'/content/direct_sympy_colab_logits_prepare.py',str(archive),manifest['sha256'],str(root),'--gpu','H100'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
marker.write_text(json.dumps({'state':'DISPATCHED','preparationPID':process.pid,'archiveSHA256':manifest['sha256']}))
print(marker.read_text())
