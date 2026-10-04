import json,os,time,platform,statistics,torch
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
    x=sample();expected=torch.sqrt(x.float()).double().view(torch.int64);actual=compiled(x).view(torch.int64)
    mismatches=int(torch.count_nonzero(actual!=expected));assert mismatches==0,(mismatches,'CPU translation')
    return len(x)


import math,struct
x=sample();actual=compiled(x);float_reference=torch.sqrt(x.float()).double();double_reference=torch.sqrt(x).float().double()
scalar_reference=torch.tensor([struct.unpack('f',struct.pack('f',math.sqrt(float(value))))[0] for value in x],dtype=torch.float64)
report={'platform':platform.platform(),'torch':str(torch.__version__),'cpuCapability':torch.backends.cpu.get_cpu_capability(),'cases':len(x),'expression':metadata,'compiledVsScalar':int(torch.count_nonzero(actual.view(torch.int64)!=scalar_reference.view(torch.int64))),'floatTorchVsScalar':int(torch.count_nonzero(float_reference.view(torch.int64)!=scalar_reference.view(torch.int64))),'doubleTorchVsScalar':int(torch.count_nonzero(double_reference.view(torch.int64)!=scalar_reference.view(torch.int64))),'cudaAvailable':torch.cuda.is_available(),'examples':[]}
for index in torch.nonzero(actual.view(torch.int64)!=float_reference.view(torch.int64)).flatten()[:8]:
 i=int(index);report['examples'].append({'input':float(x[i]).hex(),'compiled':float(actual[i]).hex(),'torchFloat':float(float_reference[i]).hex(),'scalar':float(scalar_reference[i]).hex()})
if torch.cuda.is_available():
 report['gpu']=torch.cuda.get_device_name();cuda=compiled(x.cuda()).cpu();report['cudaVsCompiledCPU']=int(torch.count_nonzero(cuda.view(torch.int64)!=actual.view(torch.int64)));report['cudaVsScalar']=int(torch.count_nonzero(cuda.view(torch.int64)!=scalar_reference.view(torch.int64)))
Path('/content/rational-sqrt-diagnostic.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
