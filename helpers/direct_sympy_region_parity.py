"""Native readback parity of emitted regions against fresh CPU checkpoint output.

This verifier never participates in the generated function. Region parity
is explicitly position-zero, one-token parity, not final last-token parity.
"""
import argparse
import itertools
import json
from pathlib import Path
import random
import struct
import subprocess
import tempfile

import torch
from transformers import AutoModelForCausalLM
from direct_sympy_conversions_test import cpp
from direct_sympy_input_partitions import rank,value
from direct_sympy_partition_run import decode,audit_tree
from direct_sympy_savepoints import atomic,digest_file
from direct_sympy_strings import syntax


def sample_bits(domains,random_cases=256):
    """Seeded corners/random inputs plus both zeros on each admitted axis."""
    if type(random_cases) is not int or random_cases<0:raise ValueError('Nonnegative sample count required')
    names=sorted(domains,key=lambda name:int(name[1:]));rng=random.Random(9183)
    if not names:raise ValueError('Input coordinates required')
    endpoints=[[rank(domains[name].minimum),rank(domains[name].maximum)] for name in names]
    vectors=list(itertools.product(*endpoints)) if len(names)<=10 else [tuple(point[i] for point in endpoints) for i in range(2)]
    vectors.extend(tuple(rng.randrange(a,b+1) for a,b in endpoints) for _ in range(random_cases))
    bits=[tuple(abs(index)|(0x8000 if index<0 else 0) for index in row) for row in vectors]
    # Both zeros belong to the same numeric region but are different IEEE inputs.
    for sign in (0,0x8000):
        if all(a<=0<=b for a,b in endpoints):bits.append((sign,)*len(names))
    # A zero on one coordinate must also be tested beside nonzero inputs.
    # Two boundary anchors keep this linear for high-dimensional models.
    seen=set(bits)
    def append(row):
        row=tuple(row)
        if row not in seen:bits.append(row);seen.add(row)
    anchors=[tuple(abs(point[i])|(0x8000 if point[i]<0 else 0) for point in endpoints) for i in (0,1)]
    for axis,(a,b) in enumerate(endpoints):
        if not a<=0<=b:continue
        for anchor in anchors:
            for sign in (0,0x8000):
                row=list(anchor);row[axis]=sign;append(row)
    if all(a<=0<=b for a,b in endpoints):
        if len(names)<=10:
            for row in itertools.product((0,0x8000),repeat=len(names)):append(row)
        else:
            for axis in range(len(names)):
                row=[0]*len(names);row[axis]=0x8000;append(row)
    return names,bits


def verify_region(checkpoint,dimension,artifact,domains,*,random_cases=256):
    names,bits=sample_bits(domains,random_cases)
    matrix=torch.tensor([[float(struct.unpack('>e',struct.pack('>H',item))[0]) for item in row] for row in bits],dtype=torch.float16).unsqueeze(1)
    model=AutoModelForCausalLM.from_pretrained(checkpoint,dtype=torch.float16,attn_implementation='eager').eval()
    if model.config.hidden_size!=len(names) or not 0<=dimension<model.config.vocab_size:raise ValueError('Reference coordinate geometry mismatch')
    with torch.inference_mode():expected=model(inputs_embeds=matrix,use_cache=False).logits[:,0,dimension].double().tolist()
    artifact=Path(artifact);text=artifact.read_text()
    if any(name in text for name in ('CompileValue','CASNumericRegion','R16(','R32(','Silu16(','sqrt(')):
        raise ValueError('Unfinished numeric dependency in emitted region')
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory);source=root/'native.cpp';binary=root/'native'
        parameters=','.join('double '+name for name in names)
        declarations=';'.join('unsigned v'+str(i) for i in range(len(names)))+';'
        fmt=' '.join('%u' for _ in names)
        refs=','.join('&v'+str(i) for i in range(len(names)))
        inputs=','.join('word<_Float16>(uint16_t(v'+str(i)+'))' for i in range(len(names)))
        source.write_text('''#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cfenv>
#include <cmath>
template<class T,class U>T word(U x){T y;std::memcpy(&y,&x,sizeof(y));return y;}
'''+f'double candidate({parameters}){{return '+cpp(syntax(text))+';}\n'+
            'int main(){if(std::fesetround(FE_TONEAREST))return 2;'+declarations+
            f'while(std::scanf("{fmt}",{refs})=={len(names)})'+
            '{auto result=candidate('+inputs+');std::printf("%016llx\\n",(unsigned long long)word<uint64_t>(result));}}')
        built=subprocess.run(['clang++','-O3','-ffp-contract=off','-std=c++17',str(source),'-o',str(binary)],capture_output=True,text=True)
        if built.returncode:raise ValueError('Native region validation build failed: '+built.stderr[:2000])
        executed=subprocess.run([str(binary)],input=''.join(' '.join(map(str,row))+'\n' for row in bits),check=True,capture_output=True,text=True)
    actual=executed.stdout.splitlines()
    if len(actual)!=len(expected):raise ValueError('Native output corpus incomplete')
    mismatches=[{'inputBits':bits[i],'actual':actual[i],'expected':struct.pack('>d',number).hex()} for i,number in enumerate(expected) if actual[i]!=struct.pack('>d',number).hex()]
    return {'artifact':str(artifact),'sha256':digest_file(artifact),'characters':len(text),'position':0,'tokens':1,
        'dimension':dimension,'inputCorpusVersion':2,'verifierSHA256':digest_file(Path(__file__)),
        'cases':len(bits),'mismatches':len(mismatches),'firstMismatches':mismatches[:8],
        'scope':'Emitted coordinate in this input region only; not full-domain or variable-length last-token parity.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('checkpoint');parser.add_argument('state');parser.add_argument('report')
    args=parser.parse_args();torch.set_num_threads(1)
    directory=Path(args.state);state=json.loads((directory/'frontier.json').read_text())
    covered,unfinished=audit_tree(state['tree'],state['root']);identity=state['identity']
    actual={p.name:digest_file(p) for p in [Path(args.checkpoint)/'config.json',*sorted(Path(args.checkpoint).glob('*.safetensors'))]}
    if actual!=identity['checkpoint']:raise ValueError('Reference checkpoint identity mismatch')
    reports=[]
    for node in state['tree'].values():
        if node['status']!='complete':continue
        artifact=directory/node['artifact']['file']
        if digest_file(artifact)!=node['artifact']['sha256']:raise ValueError('Emitted region integrity mismatch')
        reports.append(verify_region(args.checkpoint,identity['dimension'],artifact,decode(node['domains'])))
    result={'regions':reports,'inputCorpusVersion':2,'verifierSHA256':digest_file(Path(__file__)),
        'compilerIdentity':identity,'completedRegions':len(reports),'coveredInputPatterns':covered,
        'unfinishedInputPatterns':unfinished,'cases':sum(r['cases'] for r in reports),
        'mismatches':sum(r['mismatches'] for r in reports),'fullCoordinateParity':False}
    atomic(args.report,json.dumps(result,indent=2).encode());print(json.dumps(result),flush=True)
    return 1 if result['mismatches'] or not reports else 0


if __name__=='__main__':raise SystemExit(main())
