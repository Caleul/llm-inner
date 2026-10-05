"""Package exact compiler sources/target/reference for colab_cli exec -f.

Local work is limited to file I/O and compression. Compilation runs remotely.
The generated file is temporary transport, not a compiled model artifact.
"""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import tarfile

ROOT=Path(__file__).resolve().parent.parent
BOOTSTRAP="import base64,hashlib,io,json,os,shutil,subprocess,sys,tarfile,urllib.request\nfrom pathlib import Path\nroot=Path('/content/llm-inner-final-logits');root.mkdir(exist_ok=True)\npayload=base64.b64decode(PAYLOAD)\nif hashlib.sha256(payload).hexdigest()!=ARCHIVE_SHA:raise ValueError('Compilation bundle hash mismatch')\nwith tarfile.open(fileobj=io.BytesIO(payload),mode='r:gz') as archive:\n for member in archive.getmembers():\n  if member.name.startswith('/') or '..' in Path(member.name).parts or not member.isfile():raise ValueError('Unsafe compiler bundle member')\n archive.extractall(root)\nos.chdir(root)\nmissing=[]\nfor name in ['sympy','psutil','safetensors','numpy']:\n try:__import__(name)\n except ImportError:missing.append(name)\nif missing:subprocess.run([sys.executable,'-m','pip','install',*missing],check=True)\nnode=shutil.which('node')\nif node is None or int(subprocess.check_output([node,'--version'],text=True).strip().lstrip('v').split('.')[0])<22:\n url='https://nodejs.org/dist/v22.22.2/node-v22.22.2-linux-x64.tar.xz'\n data=urllib.request.urlopen(url).read()\n with tarfile.open(fileobj=io.BytesIO(data),mode='r:xz') as archive:archive.extractall(root/'node-runtime')\n os.environ['PATH']=str(root/'node-runtime/node-v22.22.2-linux-x64/bin')+':'+os.environ['PATH']\nimport torch,psutil\nresource={'device':torch.cuda.get_device_name() if torch.cuda.is_available() else None,'CPU':os.cpu_count(),'RAMBytes':psutil.virtual_memory().total,'GPUBytes':torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0,'cuda':torch.version.cuda,'node':subprocess.check_output(['node','--version'],text=True).strip(),'bundleSha256':ARCHIVE_SHA}\n(root/'bundle.json').write_text(json.dumps(resource,indent=2)+'\\n')\nlog=(root/'coordinator.log').open('w')\nprocess=subprocess.Popen([sys.executable,'helpers/direct_colab_final_logits.py'],cwd=root,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)\n(root/'coordinator-pid.json').write_text(json.dumps({'pid':process.pid})+'\\n')\nprint(json.dumps({'compilationStarted':True,'pid':process.pid,'resources':resource}),flush=True)\n"


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshot');parser.add_argument('reference');parser.add_argument('output')
    args=parser.parse_args();buffer=io.BytesIO()
    files=[p for folder in ['helpers','dist/src','numeric-profiles','docs/evidence/direct-sympy-test-checkpoint']
        for p in (ROOT/folder).glob('*') if p.is_file() and p.suffix in ('.py','.mjs','.js','.rs','.cpp','.bin','.json','.safetensors')]
    files.append(ROOT/'package.json')
    with tarfile.open(fileobj=buffer,mode='w:gz') as archive:
        for path in sorted(files):archive.add(path,arcname=str(path.relative_to(ROOT)))
        archive.add(args.snapshot,arcname='target-discovery.json')
        archive.add(args.reference,arcname='source-reference.json')
    payload=buffer.getvalue();sha=hashlib.sha256(payload).hexdigest()
    destination=Path(args.output);destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text('PAYLOAD='+repr(base64.b64encode(payload).decode())+'\nARCHIVE_SHA='+repr(sha)+'\n'+BOOTSTRAP)
    print(json.dumps({'path':str(destination.resolve()),'archiveSha256':sha,'archiveBytes':len(payload),'sourceFiles':len(files)}))


if __name__=='__main__':main()
