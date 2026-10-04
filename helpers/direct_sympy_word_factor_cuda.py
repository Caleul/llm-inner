"""CUDA numeric parity for modular expressions factored on the CPU.

The validation buffers represent raw 64-bit words, not model activations.
Signed int64 GPU add/multiply retain the same low 64 bits as unsigned
modular operations. Floating forward/reference computation stays on CPU.
"""
import argparse
import ast
import json
import time

import torch
from direct_sympy_strings import StringCompiler,syntax


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples',type=int,default=1048576)
    parser.add_argument('--batch',type=int,default=65536)
    args=parser.parse_args()
    if args.samples<1 or args.batch<1:parser.error('Positive sample and batch counts required')
    if not torch.cuda.is_available():raise RuntimeError('CUDA GPU required; no CPU fallback reported as CUDA')
    started=time.monotonic()
    sources=[
        'U64Add(U64Mul(Bits64(X1), 3), U64Mul(Bits64(X1), 7))',
        'U64Add(U64Mul(Bits64(X1), 18446744073709551615), Bits64(X1))',
        'U64Add(U64Mul(Bits64(X1), Bits64(X2)), U64Mul(Bits64(X1), Bits64(X3)))',
        'U64Add(U64Mul(Bits64(X1), Bits64(X1)), U64Mul(Bits64(X1), Bits64(X2)))',
    ]
    programs=[];sizes=[]
    class SignedConstants(ast.NodeTransformer):
        def visit_Constant(self,node):
            if type(node.value) is int and 2**63<=node.value<2**64:return ast.copy_location(ast.Constant(value=node.value-2**64),node)
            return node
    for source in sources:
        result=StringCompiler().stabilize(source,{})
        programs.append(tuple(compile(ast.fix_missing_locations(SignedConstants().visit(ast.Expression(syntax(text)))),'modular CUDA','eval') for text in (source,result)))
        sizes.append((len(source),len(result)))
    functions={'Bits64':lambda x:x,'U64Add':lambda a,b:a+b,'U64Mul':lambda a,b:a*b}
    generator=torch.Generator(device='cuda').manual_seed(917)
    boundaries=torch.tensor([0,1,-1,2**63-1,-2**63,2**62,-2**62,65535],dtype=torch.int64,device='cuda')
    comparisons=0;cpu_boundary_comparisons=0;torch.cuda.reset_peak_memory_stats()
    for start in range(0,args.samples,args.batch):
        size=min(args.batch,args.samples-start)
        values=torch.randint(-2**63,2**63-1,(size,3),dtype=torch.int64,device='cuda',generator=generator)
        if start==0:
            n=min(len(boundaries),size)
            for column in range(3):values[:n,column]=boundaries.roll(column)[:n]
        inputs={f'X{i+1}':values[:,i] for i in range(3)}
        for original,factored in programs:
            a=eval(original,{'__builtins__':{},**functions},inputs)
            b=eval(factored,{'__builtins__':{},**functions},inputs)
            if not torch.equal(torch.as_tensor(a,dtype=torch.int64,device='cuda').expand(size),torch.as_tensor(b,dtype=torch.int64,device='cuda').expand(size)):
                raise AssertionError('CUDA modular parity mismatch')
            if start==0:
                mask=2**64-1
                cpu_functions={'Bits64':lambda x:x&mask,'U64Add':lambda a,b:(a+b)&mask,'U64Mul':lambda a,b:(a*b)&mask}
                rows=values[:n].tolist();observed=torch.as_tensor(a,dtype=torch.int64,device='cuda').expand(size)[:n].tolist()
                for row,actual in zip(rows,observed):
                    expected=eval(original,{'__builtins__':{},**cpu_functions},{f'X{i+1}':value for i,value in enumerate(row)})
                    expected=expected if expected<2**63 else expected-2**64
                    if actual!=expected:raise AssertionError('CUDA/CPU integer boundary mismatch')
                    cpu_boundary_comparisons+=1
            comparisons+=size
    torch.cuda.synchronize()
    print(json.dumps({'device':torch.cuda.get_device_name(),'cuda':torch.version.cuda,
        'samples':args.samples,'comparisons':comparisons,'mismatches':0,
        'cpuBoundaryComparisons':cpu_boundary_comparisons,
        'peakGPUAllocatedBytes':torch.cuda.max_memory_allocated(),'seconds':time.monotonic()-started,
        'expressionCharacters':sizes,'scope':'Modular integer arithmetic only; not checkpoint forward or complete-coordinate parity'}))


if __name__=='__main__':main()
