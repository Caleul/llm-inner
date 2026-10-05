"""Validate the actual elementary root JSON on CUDA; never a model runtime.

CPU IEEE F32 root/reciprocal are diagnostic oracles. Each CUDA operation is
an eager kernel, so no fusion can change the declared arithmetic order.
"""
import argparse
import hashlib
import json
import struct
import time
from pathlib import Path

import numpy as np
import torch


def evaluate(node,values,memo):
    key=id(node)
    if key in memo:return memo[key]
    tag,dtype,*args=node
    if tag=='input':answer=values[args[0]]
    elif tag=='constant':
        bits=int(args[0],0)
        if dtype=='f64':answer=struct.unpack('>d',bits.to_bytes(8,'big'))[0]
        elif dtype=='u64' and bits<2**63:answer=bits
        else:raise ValueError('Unsupported CUDA certificate constant')
    else:
        operands=[evaluate(x,values,memo) for x in args]
        if tag=='reinterpret':
            if not isinstance(operands[0],torch.Tensor):
                source=torch.tensor(operands[0],device='cuda',dtype=torch.int64 if dtype=='f64' else torch.float64)
            else:source=operands[0]
            answer=source.view(torch.float64 if dtype=='f64' else torch.int64)
        else:
            operations={'add':lambda a,b:a+b,'sub':lambda a,b:a-b,'mul':lambda a,b:a*b,
                'div':lambda a,b:torch.div(a,b) if isinstance(a,torch.Tensor) or isinstance(b,torch.Tensor) else a/b,'and':lambda a,b:a&b,'or':lambda a,b:a|b,
                'xor':lambda a,b:a^b,'shl':lambda a,b:a<<b,'shr':lambda a,b:a>>b}
            if tag not in operations or len(operands)!=2:raise ValueError('Unsupported CUDA certificate operation')
            answer=operations[tag](*operands)
    memo[key]=answer;return answer


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('expressions');parser.add_argument('output');parser.add_argument('--batch',type=int,default=131072)
    args=parser.parse_args()
    if args.batch<1:raise ValueError('Positive batch size required')
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no silent CPU fallback')
    payload=Path(args.expressions).read_bytes();expressions=json.loads(payload)
    torch.cuda.reset_peak_memory_stats();started=time.monotonic();tested=0;mismatches={'root':0,'inverse':0}
    for exponent in (127,128):
        for begin in range(0,2**23,args.batch):
            end=min(2**23,begin+args.batch)
            bits=np.arange(begin,end,dtype=np.uint32)|(exponent<<23)
            source=bits.view(np.float32)
            expected_root=np.sqrt(source).astype(np.float32)
            expected_inverse=(np.float32(1)/expected_root).astype(np.float32)
            X1=torch.from_numpy(source.astype(np.float64)).to('cuda')
            for name,expected in [('root',expected_root),('inverse',expected_inverse)]:
                actual=evaluate(expressions[name],{'X1':X1},{}).cpu().numpy()
                mismatches[name]+=int(np.count_nonzero(actual.view(np.uint64)!=expected.astype(np.float64).view(np.uint64)))
            tested+=len(source)
    torch.cuda.synchronize()
    report={'device':torch.cuda.get_device_name(),'cuda':torch.version.cuda,'expressionSha256':hashlib.sha256(payload).hexdigest(),
        'normalizedInputs':tested,'comparedOutputs':tested*2,'mismatches':mismatches,'seconds':time.monotonic()-started,
        'peakGPUAllocatedBytes':torch.cuda.max_memory_allocated(),'purpose':'Primitive numerical validation only; not final artifact parity'}
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
    if any(mismatches.values()):raise SystemExit(1)


if __name__=='__main__':main()
