"""Prepare and start a fresh Colab job through colab_cli exec -f.

Upload this script and its source archive via Access Broker. No remote service
is accessed directly by this entry point; it runs inside the allocated VM.
The archive SHA256 must be passed explicitly. Existing jobs are never replayed.
"""
import argparse,hashlib,json,os,shutil,subprocess,sys,tarfile,time
from pathlib import Path

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('archive');parser.add_argument('sha256');parser.add_argument('root')
    parser.add_argument('--gpu',default='H100',choices=['H100','A100']);args=parser.parse_args()
    archive=Path(args.archive);root=Path(args.root)
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=args.sha256:raise ValueError('Uploaded archive integrity mismatch')
    root.mkdir(parents=True,exist_ok=False)
    with tarfile.open(archive)as stream:
        for entry in stream.getmembers():
            if entry.name.startswith('/')or '..'in Path(entry.name).parts or not(entry.isfile()or entry.isdir()):raise ValueError('Unsafe source archive member')
        stream.extractall(root)
    status={'state':'PREPARING','requestedGPU':args.gpu,'archiveSHA256':args.sha256,'startedAt':time.time()}
    def save():
        path=root/'preparation.json';temporary=root/'preparation.tmp';temporary.write_text(json.dumps(status,indent=2));os.replace(temporary,path)
    save()
    with (root/'dependency-setup.log').open('w')as log:
        subprocess.run([sys.executable,'-m','pip','install','--quiet','sympy==1.14.0','transformers==5.5.0','safetensors==0.8.0','numpy==2.2.6','psutil==7.0.0'],stdout=log,stderr=subprocess.STDOUT,check=True)
        if not shutil.which('clang++'):
            subprocess.run(['apt-get','update','-qq'],stdout=log,stderr=subprocess.STDOUT,check=True)
            subprocess.run(['apt-get','install','-y','-qq','clang'],stdout=log,stderr=subprocess.STDOUT,check=True)
    import torch,psutil,sympy
    hardware={'cpuCount':os.cpu_count(),'ramTotalBytes':psutil.virtual_memory().total,'ramAvailableBytes':psutil.virtual_memory().available,
        'torch':str(torch.__version__),'sympy':sympy.__version__,'cuda':torch.version.cuda,'cudaAvailable':torch.cuda.is_available()}
    if hardware['cudaAvailable']:
        gpu=torch.cuda.get_device_properties(0);hardware.update(gpu=gpu.name,gpuVRAMBytes=gpu.total_memory)
    status['resources']=hardware;save()
    if not hardware['cudaAvailable']or args.gpu not in hardware.get('gpu',''):
        status.update(state='BLOCKED',error='Allocated accelerator does not match request');save();print(json.dumps(status));return 1
    environment={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','LLM_INNER_DIRECT_JSON_CHECKPOINT':str(root/'docs/evidence/direct-sympy-test-checkpoint')}
    for helper in ('coherent_paths','artifact_budget','projection_sign'):
        with (root/(helper+'-tests.log')).open('w')as log:
            result=subprocess.run([sys.executable,'helpers/direct_sympy_'+helper+'_test.py'],cwd=root,env=environment,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            status.update(state='FAILED',error=f'Numerical preflight failed: {helper}');save();print(json.dumps(status));return 1
    command=[sys.executable,'helpers/direct_sympy_logits_run.py',environment['LLM_INNER_DIRECT_JSON_CHECKPOINT'],str(root/'logits-results'),
        '--require-gpu',args.gpu,'--expression-bytes',str(512*1024**2),'--workers',str(min(4,os.cpu_count()or 1))]
    with (root/'controller.log').open('w')as log:
        process=subprocess.Popen(command,cwd=root,env=environment,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    status.update(state='DISPATCHED',controllerPID=process.pid,controllerCommand=command);save();print(json.dumps(status))
    return 0
if __name__=='__main__':raise SystemExit(main())
