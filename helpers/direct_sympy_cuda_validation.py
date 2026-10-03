"""CUDA only for numerical parity batches; symbolic algebra stays on CPU.

This validation admits the exact tested domains, never arbitrary Torch
kernels or a different reduction tree. Timings include warm-up separately
and report both resident-device and transfer-inclusive measurements.
"""
import json
import statistics
import time
import torch


def main():
    if not torch.cuda.is_available():raise RuntimeError('CUDA validation requires an NVIDIA CUDA runtime')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    bits=torch.arange(65536,dtype=torch.int32)
    bits=bits[(bits&0x7c00)!=0x7c00].to(torch.int16)
    x=bits.view(torch.float16).to(torch.float64)
    y=torch.tensor([0,0x8000,1,0x8001,0x03ff,0x83ff,0x0400,0x8400,0x3555,0xb555,0x3c01,0xbc01,0x7bff,0xfbff],dtype=torch.int32).to(torch.int16).view(torch.float16).to(torch.float64)
    def products(a,b):return (a[:,None]*b[None,:]).to(torch.float32).to(torch.float64)
    expected=products(x,y);cx=x.cuda();cy=y.cuda()
    for _ in range(3):products(cx,cy)
    torch.cuda.synchronize()
    def median(action,cuda=False):
        samples=[]
        for _ in range(15):
            if cuda:torch.cuda.synchronize()
            start=time.perf_counter();action()
            if cuda:torch.cuda.synchronize()
            samples.append(time.perf_counter()-start)
        return statistics.median(samples)
    cpu_time=median(lambda:products(x,y))
    gpu_time=median(lambda:products(cx,cy),True)
    end_to_end=median(lambda:products(x.cuda(),y.cuda()).cpu(),True)
    actual=products(cx,cy).cpu()
    product_mismatches=(actual.view(torch.int64)!=expected.view(torch.int64)).sum().item()
    small=x[(x>=-3/128)&(x<=3/128)].to(torch.float16)
    reference=torch.nn.functional.silu(small)
    silu_cuda=torch.nn.functional.silu(small.cuda()).cpu()
    silu_mismatches=(reference.view(torch.int16)!=silu_cuda.view(torch.int16)).sum().item()
    report={'device':torch.cuda.get_device_name(0),'torch':str(torch.__version__),'cuda':torch.version.cuda,'cpuThreads':torch.get_num_threads(),
        'halfProducts':{'cases':expected.numel(),'mismatches':product_mismatches,'medianCpuSeconds':cpu_time,'medianCudaSeconds':gpu_time,
            'medianCudaWithTransfersSeconds':end_to_end,'samples':15,'warmup':3},
        'silu':{'cases':small.numel(),'mismatches':silu_mismatches,'domain':[-3/128,3/128],'cudaAdmitted':silu_mismatches==0},
        'cudaPeakAllocatedBytes':torch.cuda.max_memory_allocated()}
    print(json.dumps(report,indent=2))
    if product_mismatches or silu_mismatches:raise RuntimeError('CUDA numerical parity failed; kernel not admitted')


if __name__=='__main__':main()
