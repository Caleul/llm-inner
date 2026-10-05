import json,os,time
import torch
if not torch.cuda.is_available():raise RuntimeError('CUDA GPU required')
torch.set_num_threads(1)
torch.backends.cuda.matmul.allow_tf32=False
torch.use_deterministic_algorithms(True)
WEIGHTS = {'v': [[0.0030536651611328125, 0.00511932373046875], [-0.0226287841796875, 0.0184478759765625]], 'o': [[0.032928466796875, -0.0225982666015625], [0.0203399658203125, 0.006168365478515625]]}
bits=torch.arange(65536,dtype=torch.int32)
bits=bits[(bits&0x7c00)!=0x7c00].to(torch.int16)
x=bits.view(torch.float16)
# Every finite Half is visited on each axis, including both zero signs.
values=torch.stack((x,x.roll(733)),dim=1).to(torch.float64)
def projection(inputs,name):
    weight=torch.tensor(WEIGHTS[name],dtype=torch.float64,device=inputs.device)
    lanes=[]
    for lane in range(4):
        accumulator=torch.zeros((len(inputs),2),dtype=torch.float64,device=inputs.device)
        for column in range(lane,2,4):
            product=(inputs[:,column,None]*weight[None,:,column]).to(torch.float32).to(torch.float64)
            accumulator=(accumulator+product).to(torch.float32).to(torch.float64)
        lanes.append(accumulator)
    left=(lanes[0]+lanes[1]).to(torch.float32).to(torch.float64)
    right=(lanes[2]+lanes[3]).to(torch.float32).to(torch.float64)
    return (left+right).to(torch.float32).to(torch.float16).to(torch.float64)
def ordered(inputs):return projection(projection(inputs,'v'),'o')
torch.cuda.reset_peak_memory_stats()
start=time.perf_counter();cpu=ordered(values);cpu_seconds=time.perf_counter()-start
gpu_values=values.cuda();ordered(gpu_values);torch.cuda.synchronize()
start=time.perf_counter();gpu=ordered(gpu_values);torch.cuda.synchronize();gpu_seconds=time.perf_counter()-start
gpu=gpu.cpu()
mismatches=int((cpu.view(torch.int64)!=gpu.view(torch.int64)).sum())
record={'device':torch.cuda.get_device_name(),'cuda':torch.version.cuda,'cpuCount':os.cpu_count(),
 'totalRAMBytes':os.sysconf('SC_PAGE_SIZE')*os.sysconf('SC_PHYS_PAGES'),
 'totalGPUBytes':torch.cuda.get_device_properties(0).total_memory,'peakGPUAllocatedBytes':torch.cuda.max_memory_allocated(),
 'pairs':len(values),'comparedOutputs':cpu.numel(),'mismatches':mismatches,
 'cpuSeconds':cpu_seconds,'residentGPUSeconds':gpu_seconds,
 'scope':'Actual checkpoint V/O weights, original four-lane ordered numerical projections; not full-coordinate artifact parity'}
print(json.dumps(record))
if mismatches:raise AssertionError('CUDA projection parity failed; GPU backend not admitted')
