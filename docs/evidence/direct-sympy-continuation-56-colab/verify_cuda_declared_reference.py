import json,os,time,platform,statistics,torch,math,struct
from pathlib import Path
torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
def f64(a):return a.to(torch.float64) if isinstance(a,torch.Tensor) else float(a)
def Bits64(a):return a.view(torch.int64)
def Float64(a):return a.view(torch.float64)
def U64And(a,b):return a & b
def U64Or(a,b):return a | b
def U64Add(a,b):return a+b
def U64Mul(a,b):return a*b
def U64Shr(a,b):
    if type(b) is not int or not 0<=b<64:raise ValueError('Constant unsigned shift required')
    mask=((1<<(64-b))-1);mask=(mask+(1<<63))%(1<<64)-(1<<63)
    return (a>>b)&mask
def compiled(X1):
    n0 = Bits64(X1)
    n1 = U64And(n0, 4503599627370495)
    n2 = U64Or(n1, 4607182418800017408)
    n3 = Float64(n2)
    n4 = f64(n3) - f64(1.5)
    n5 = f64(n4) + f64(18.5850823052632)
    n6 = Bits64(X1)
    n7 = U64And(n6, 4503599627370495)
    n8 = U64Or(n7, 4607182418800017408)
    n9 = Float64(n8)
    n10 = f64(n9) - f64(1.5)
    n11 = f64(n10) + f64(1.664726217594286)
    n12 = f64(n11) * f64(n11)
    n13 = f64(n12) + f64((-0.018266192753258678))
    n14 = f64(n5) * f64(n13)
    n15 = Bits64(X1)
    n16 = U64And(n15, 4503599627370495)
    n17 = U64Or(n16, 4607182418800017408)
    n18 = Float64(n17)
    n19 = f64(n18) - f64(1.5)
    n20 = f64(n19) + f64(3.8155673924431786)
    n21 = f64(n20) * f64(n20)
    n22 = f64(n21) + f64((-1.5052409305796959))
    n23 = f64(n14) * f64(n22)
    n24 = f64(13.367174015436346) * f64(n23)
    n25 = Bits64(X1)
    n26 = U64And(n25, 4503599627370495)
    n27 = U64Or(n26, 4607182418800017408)
    n28 = Float64(n27)
    n29 = f64(n28) - f64(1.5)
    n30 = f64(n29) + f64(72.88886838020784)
    n31 = Bits64(X1)
    n32 = U64And(n31, 4503599627370495)
    n33 = U64Or(n32, 4607182418800017408)
    n34 = Float64(n33)
    n35 = f64(n34) - f64(1.5)
    n36 = f64(n35) + f64(1.8598888910976752)
    n37 = f64(n36) * f64(n36)
    n38 = f64(n37) + f64((-0.05583983254715131))
    n39 = f64(n30) * f64(n38)
    n40 = Bits64(X1)
    n41 = U64And(n40, 4503599627370495)
    n42 = U64Or(n41, 4607182418800017408)
    n43 = Float64(n42)
    n44 = f64(n43) - f64(1.5)
    n45 = f64(n44) + f64(5.991012208537561)
    n46 = f64(n45) * f64(n45)
    n47 = f64(n46) + f64((-6.507242687878663))
    n48 = f64(n39) * f64(n47)
    n49 = f64(n24) / f64(n48)
    n50 = Bits64(X1)
    n51 = U64Shr(n50, 52)
    n52 = U64And(n51, 2047)
    n53 = U64Add(n52, 1)
    n54 = U64And(n53, 1)
    n55 = f64(n54) * f64(0.4142135623730951)
    n56 = f64(1.0) + f64(n55)
    n57 = f64(n49) * f64(n56)
    n58 = Bits64(n57)
    n59 = U64Add(n58, 268435455)
    n60 = U64And(n59, -536870912)
    n61 = Bits64(X1)
    n62 = U64Shr(n61, 52)
    n63 = U64And(n62, 2047)
    n64 = U64Add(n63, 1023)
    n65 = U64Shr(n64, 1)
    n66 = U64Add(n65, -1023)
    n67 = U64Mul(n66, 4503599627370496)
    n68 = U64Add(n60, n67)
    n69 = Float64(n68)
    return n69

metadata={'expressionSHA256': 'd71236cdccd68965a02b876160f9f814a27263f3caf690636e83c7450b74e654', 'expressionCharacters': 1048, 'inputOccurrences': 8, 'operations': 70}

def sample():
    generator=torch.Generator().manual_seed(9183)
    bits=torch.cat((torch.randint(1,0x7f800000,(8192,),generator=generator,dtype=torch.int32),torch.tensor([1,8388607,8388608,0x7f7fffff],dtype=torch.int32)))
    return bits.view(torch.float32).double()

def cpu_check():
    x=sample();expected=torch.tensor([struct.unpack("f",struct.pack("f",math.sqrt(float(v))))[0] for v in x],dtype=torch.float64).view(torch.int64);actual=compiled(x).view(torch.int64)
    mismatches=int(torch.count_nonzero(actual!=expected));assert mismatches==0,(mismatches,'CPU translation')
    return len(x)

if os.environ.get('LLM_INNER_CUDA_TRANSLATION_ONLY'):
    print(json.dumps({'cpuTranslationCases':cpu_check(),'mismatches':0,'expression':metadata}));raise SystemExit(0)

result={'expression':metadata,'cpuTranslationCases':cpu_check(),'platform':platform.platform(),'torch':str(torch.__version__),'cpuCount':os.cpu_count(),'ramBytes':os.sysconf('SC_PAGE_SIZE')*os.sysconf('SC_PHYS_PAGES'),'cudaAvailable':torch.cuda.is_available(),'finalCoordinateParity':False,'scope':'Emitted positive F32 sqrt CUDA expression vs the same CPU expression and F64 sqrt rounded to F32; not PyTorch Float32 sqrt, model/coordinate GPU parity, or CAS speedup','torchFloat32SqrtReferenceCompatible':False}
print(json.dumps(result),flush=True)
if not result['cudaAvailable']:
    Path('/content/rational-sqrt-report.json').write_text(json.dumps(result,indent=2));raise RuntimeError('CUDA unavailable')
result['gpu']=torch.cuda.get_device_name();free,total=torch.cuda.mem_get_info();result['gpuBytes']=total
budget=min(512*1024**2,free//4);tile=min(262144,max(1024,budget//(20*metadata['operations'])))
result.update(memoryBudgetBytes=budget,tileValues=tile,cases=0,mismatches=0,cudaVsCPUExpressionMismatches=0,kernelMilliseconds=0.0,transferAndKernelSeconds=0.0)
# Compare the same straight-line expression on one CPU thread and CUDA.
x=sample().repeat(32);xgpu=x.cuda();compiled(xgpu);torch.cuda.synchronize()
cpu=[];gpu=[];transfer=[]
for _ in range(5):
    before=time.perf_counter();compiled(x);cpu.append(time.perf_counter()-before)
    start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);start.record();compiled(xgpu);end.record();end.synchronize();gpu.append(start.elapsed_time(end)/1000)
    before=time.perf_counter();compiled(x.cuda()).cpu();torch.cuda.synchronize();transfer.append(time.perf_counter()-before)
result['numericalBenchmark']={'values':len(x),'trials':5,'cpuThreads':1,'cpuMedianSeconds':statistics.median(cpu),'cudaResidentMedianSeconds':statistics.median(gpu),'cudaIncludingTransfersMedianSeconds':statistics.median(transfer),'scope':'Elementwise emitted sqrt expression; not factor/simplify or model speedup'}
del x,xgpu;torch.cuda.reset_peak_memory_stats()
started=time.perf_counter()
def validate(x):
    expected=torch.sqrt(x).float().double().view(torch.int64)
    cpu_expression=compiled(x).view(torch.int64)
    before=time.perf_counter();device=x.cuda();a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record();actual=compiled(device);b.record();b.synchronize();result['kernelMilliseconds']+=a.elapsed_time(b)
    words=actual.view(torch.int64).cpu();result['transferAndKernelSeconds']+=time.perf_counter()-before
    result['mismatches']+=int(torch.count_nonzero(words!=expected));result['cudaVsCPUExpressionMismatches']+=int(torch.count_nonzero(words!=cpu_expression));result['cases']+=len(x)
    if torch.cuda.max_memory_allocated()>budget:raise RuntimeError('Numerical tile memory budget exceeded')
for exponent in (127,128):
    for first in range(0,8388608,tile):
        bits=torch.arange(first,min(first+tile,8388608),dtype=torch.int32)|(exponent<<23);validate(bits.view(torch.float32).double())
    print(json.dumps({'completedExponent':exponent,'cases':result['cases'],'mismatches':result['mismatches']}),flush=True)
for first in range(1,8388608,tile):
    bits=torch.arange(first,min(first+tile,8388608),dtype=torch.int32);validate(bits.view(torch.float32).double())
exponents=torch.arange(1,255,dtype=torch.int32).unsqueeze(1);mantissas=torch.tensor([0,1,2,4194303,4194304,8388606,8388607],dtype=torch.int32).unsqueeze(0);bits=((exponents<<23)|mantissas).flatten();validate(bits.view(torch.float32).double())
result.update(seconds=time.perf_counter()-started,cudaPeakAllocatedBytes=torch.cuda.max_memory_allocated(),cudaPeakReservedBytes=torch.cuda.max_memory_reserved())
assert result['cases']==25167601
Path('/content/rational-sqrt-report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if result['mismatches'] or result['cudaVsCPUExpressionMismatches']:raise RuntimeError('CUDA expression differs from declared references')
