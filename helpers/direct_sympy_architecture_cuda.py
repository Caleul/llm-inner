"""Numerical validation of CPU-built complete architecture templates on CUDA.

The compiler remains on CPU. This diagnostic never becomes the final runtime,
and it compares the selected logits with independently captured native values.
"""
import ast
import json
import os
from pathlib import Path
import struct
import time

import torch


def validate(plans,cases):
    if not torch.cuda.is_available():raise RuntimeError('NVIDIA CUDA runtime required')
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    torch.use_deterministic_algorithms(True);torch.cuda.reset_peak_memory_stats()
    device='cuda';results=[];started=time.monotonic()
    class Selectors(ast.NodeTransformer):
        def visit_Call(self,node):
            node=self.generic_visit(node)
            if isinstance(node.func,ast.Name) and node.func.id=='Piecewise':
                value=ast.Constant(float('nan'))
                for arm in reversed(node.args):
                    body,condition=arm.elts
                    value=ast.Call(ast.Name('Select',ast.Load()),[condition,body,value],[])
                return ast.copy_location(value,node)
            return node
    def array(value,dtype=torch.float64):return torch.as_tensor(value,dtype=dtype,device=device)
    def conjunction(*values):
        result=array(True,torch.bool)
        for value in values:result=torch.logical_and(result,array(value,torch.bool))
        return result
    functions={'R32':lambda x:array(x,torch.float32).double(),'R16':lambda x:array(x,torch.float16).double(),
        'sqrt':lambda x:torch.sqrt(array(x)),'Silu16':lambda x:torch.nn.functional.silu(array(x,torch.float16)).double(),
        'Exp32':lambda x:torch.exp(array(x,torch.float32)).double(),'And':conjunction,
        'Select':lambda c,a,b:torch.where(array(c,torch.bool),array(a),array(b))}
    for plan in plans:
        length=plan['length'];indices=plan.get('outputIndices',list(range(length*plan['vocab'])));batch=[case for case in cases if len(case['inputBits'])==length]
        values=torch.tensor([[struct.unpack('e',struct.pack('H',bits))[0] for row in case['inputBits'] for bits in row] for case in batch],dtype=torch.float64,device=device)
        start=time.monotonic()
        for block in plan['blocks']:
            inputs={name:values[:,i] for i,name in enumerate(block['inputs'])};outputs=[]
            for expression in block['outputs']:
                tree=Selectors().visit(ast.parse(expression,mode='eval'))
                code=compile(ast.fix_missing_locations(tree),'<CUDA architecture>','eval')
                value=eval(code,{'__builtins__':{},**functions},inputs)
                outputs.append(array(value).expand(len(batch)))
            values=torch.stack(outputs,dim=1)
        torch.cuda.synchronize()
        actual=values.cpu().contiguous().view(torch.int64).tolist();mismatches=[]
        if values.shape[1]!=len(indices):raise AssertionError('Incomplete selected CUDA logit vector')
        for row,case in zip(actual,batch):
            all_expected=[word for token in case['logitF64Bits'] for word in token]
            expected=[all_expected[index] for index in indices]
            for index,(value,wanted) in enumerate(zip(row,expected)):
                observed=f'0x{value&((1<<64)-1):016x}'
                if observed!=wanted:mismatches.append({'case':case['label'],'position':indices[index]//plan['vocab'],'coordinate':indices[index]%plan['vocab'],'actual':observed,'expected':wanted})
        results.append({'length':length,'cases':len(batch),'logits':len(batch)*len(indices),'outputIndices':indices,
            'mismatches':mismatches,'seconds':time.monotonic()-start})
    record={'device':torch.cuda.get_device_name(),'cuda':torch.version.cuda,'cpuCount':os.cpu_count(),
        'totalRAMBytes':os.sysconf('SC_PAGE_SIZE')*os.sysconf('SC_PHYS_PAGES'),
        'totalGPUBytes':torch.cuda.get_device_properties(0).total_memory,'peakGPUAllocatedBytes':torch.cuda.max_memory_allocated(),
        'results':results,'seconds':time.monotonic()-started,'finalArtifactParity':False,
        'scope':'Numerical complete-architecture templates on CUDA; symbolic composition and primitive lowering remain CPU work'}
    print(json.dumps(record),flush=True)
    if any(item['mismatches'] for item in results):raise AssertionError('CUDA whole-architecture parity failed')
    return record
