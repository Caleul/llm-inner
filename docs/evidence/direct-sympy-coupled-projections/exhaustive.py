"""Stream all central-region Half inputs through the actual emitted file."""
import argparse,json,struct,subprocess,sys,tempfile,time
from pathlib import Path
sys.path.insert(0,str(Path('helpers').resolve()))
import numpy as np
import torch
from transformers import AutoModelForCausalLM
from direct_sympy_conversions_test import cpp
from direct_sympy_strings import syntax
from direct_sympy_savepoints import atomic,digest_file
from direct_sympy_cover_regions import live_identity

checkpoint='docs/evidence/direct-sympy-test-checkpoint'
directory=Path('artifacts/direct-sympy-input-partitions/central-guard-diagnosis')
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--artifact',default=str(directory/'coupled.expr'))
parser.add_argument('--report',default=str(directory/'exhaustive-parity.json'))
parser.add_argument('--x-sign',type=int,choices=(-1,1),default=1)
parser.add_argument('--y-sign',type=int,choices=(-1,1),default=1)
args=parser.parse_args()
artifact=Path(args.artifact);text=artifact.read_text()
identity=live_identity(checkpoint,2);artifact_sha=digest_file(artifact)
assert not any(s in text for s in ('CompileValue','R16(','R32(','sqrt(','Silu16('))
torch.set_num_threads(1)
model=AutoModelForCausalLM.from_pretrained(checkpoint,dtype=torch.float16,attn_implementation='eager').eval()
total=5121*4097;batch=4096;started=time.monotonic();failures=0;first=[]
with tempfile.TemporaryDirectory() as temporary:
    root=Path(temporary);source=root/'native.cpp';binary=root/'native'
    source.write_text('''#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cfenv>
template<class T,class U>T word(U x){T y;std::memcpy(&y,&x,sizeof(y));return y;}
double candidate(double X1,double X2){return '''+cpp(syntax(text))+''';}
int main(){if(std::fesetround(FE_TONEAREST))return 2;
 uint32_t n;uint16_t in[8192];uint64_t out[4096];
 while(std::fread(&n,4,1,stdin)==1){
  if(n>4096||std::fread(in,4,n,stdin)!=n)return 3;
  for(unsigned i=0;i<n;i++)out[i]=word<uint64_t>(candidate(word<_Float16>(in[2*i]),word<_Float16>(in[2*i+1])));
  if(std::fwrite(out,8,n,stdout)!=n||std::fflush(stdout))return 4;
 }return std::ferror(stdin)?5:0;}
''')
    built=subprocess.run(['clang++','-O3','-ffp-contract=off','-std=c++17',str(source),'-o',str(binary)],capture_output=True,text=True)
    if built.returncode:raise ValueError(built.stderr[:2000])
    process=subprocess.Popen([str(binary)],stdin=subprocess.PIPE,stdout=subprocess.PIPE)
    try:
        for offset in range(0,total,batch):
            indices=np.arange(offset,min(offset+batch,total),dtype=np.uint32)
            bits=np.column_stack((11264+indices//4097,11264+indices%4097)).astype(np.uint16)
            if args.x_sign<0:bits[:,0]|=0x8000
            if args.y_sign<0:bits[:,1]|=0x8000
            process.stdin.write(struct.pack('I',len(bits))+bits.tobytes());process.stdin.flush()
            raw=process.stdout.read(8*len(bits))
            if len(raw)!=8*len(bits):raise ValueError('Native output incomplete')
            actual=np.frombuffer(raw,dtype=np.uint64)
            with torch.inference_mode():
                expected=model(inputs_embeds=torch.from_numpy(bits.view(np.float16)).unsqueeze(1),use_cache=False).logits[:,0,2].double().numpy().view(np.uint64)
            errors=np.flatnonzero(actual!=expected);failures+=len(errors)
            for i in errors[:max(0,8-len(first))]:
                first.append({'inputBits':bits[i].tolist(),'actual':f'{int(actual[i]):016x}','expected':f'{int(expected[i]):016x}'})
            if offset//batch%256==0 or offset+len(bits)==total:
                report={'artifact':str(artifact),'sha256':artifact_sha,'characters':len(text),'compilerIdentity':identity,
                    'verifierSHA256':digest_file(Path(__file__)),'cases':offset+len(bits),'totalCases':total,
                    'mismatches':failures,'firstMismatches':first,'seconds':time.monotonic()-started,
                    'batchSize':batch,'position':0,'tokens':1,'dimension':2,
                    'complete':offset+len(bits)==total,'inputDomains':{'X1':[-2,-.0625] if args.x_sign<0 else [.0625,2],
                        'X2':[-1,-.0625] if args.y_sign<0 else [.0625,1]},
                    'scope':'Every Half pair in the declared central region; not full-coordinate or multiple-token parity.'}
                atomic(args.report,json.dumps(report,indent=2).encode())
                print(json.dumps({key:report[key] for key in ('cases','totalCases','mismatches','seconds','complete')}),flush=True)
        process.stdin.close()
        if process.wait()!=0:raise ValueError('Native verification failed')
    finally:
        if process.poll() is None:process.terminate();process.wait()
        process.stdout.close()
        if not process.stdin.closed:process.stdin.close()
assert live_identity(checkpoint,2)==identity and digest_file(artifact)==artifact_sha
raise SystemExit(1 if failures else 0)
