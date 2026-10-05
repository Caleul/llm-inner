import json,subprocess
from pathlib import Path
import mpmath as mp
import sympy as sp
folder=Path(__file__).resolve().parent;variants=[]
with mp.workdps(100):
 z=sp.Symbol('z')
 for n,low,high in ((4,'1','1.625'),(4,'1.03125','1.6875'),(4,'1.046875','1.6640625')):
  center=(mp.mpf(low)+mp.mpf(high))/2;radius=(mp.mpf(high)-mp.mpf(low))/2
  nodes=[radius*mp.cos(mp.pi*i/(2*n))for i in range(2*n+1)]
  matrix=mp.matrix([[x**j for j in range(n+1)]+[-mp.sqrt(x+center)*x**j for j in range(1,n+1)]for x in nodes])
  solution=mp.lu_solve(matrix,mp.matrix([mp.sqrt(x+center)for x in nodes]))
  numerator=[solution[j]for j in range(n+1)];denominator=[mp.mpf(1)]+[solution[n+j]for j in range(1,n+1)]
  poles=sp.nroots(sp.Poly.from_list([sp.Float(mp.nstr(c,100),100)for c in reversed(denominator)],z),n=70,maxsteps=500)
  fractions=[]
  for pole in poles:
   r=mp.mpf(str(pole));residue=sum(numerator[i]*r**i for i in range(n+1))/sum(i*denominator[i]*r**(i-1)for i in range(1,n+1));fractions.append((float(residue),float(-r)))
  fractions.sort(key=lambda row:abs(row[0]/row[1]),reverse=True)
  variants.append({'degree':n,'interval':[float(low),float(high)],'center':float(center),'constant':float(numerator[-1]/denominator[-1]),'fractions':fractions})
(folder/'middle-coefficients.json').write_text(json.dumps(variants,indent=2)+'\n')
functions=[]
for i,v in enumerate(variants):
 expr=repr(v['constant'])
 for a,b in v['fractions']:expr=f'(({expr})+({a!r}/(z+{b!r})))'
 functions.append(f'double f{i}(double m,unsigned p){{double z=m-{v["center"]!r};return ({expr})*(p?1.4142135623730951:1.0);}}')
source='''#include <cstdint>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <cfenv>
template<class T,class U>T word(U x){T y;std::memcpy(&y,&x,sizeof(y));return y;}
'''+ '\n'.join(functions)+'\nint main(){std::fesetround(FE_TONEAREST);double (*fs[])(double,unsigned)={'+','.join(f'f{i}'for i in range(len(variants)))+'};unsigned ranges[][2]={'+','.join('{'+str(int((v['interval'][0]-1)*2**23))+','+str(min(2**23-1,int((v['interval'][1]-1)*2**23)))+'}'for v in variants)+'''};
for(unsigned i=0;i<sizeof(fs)/sizeof(fs[0]);i++){uint64_t cases=0,fail=0,ties=0;for(unsigned m=ranges[i][0];m<=ranges[i][1];m++)for(unsigned p=0;p<2;p++){
 double x=word<float>(uint32_t((127u<<23)|m));double raw=fs[i](x,p);
 fail+=word<uint32_t>(float(raw))!=word<uint32_t>(float(std::sqrt(x*(p?2.0:1.0))));
 ties+=(word<uint64_t>(raw)&UINT64_C(536870911))==UINT64_C(268435456);cases++;
}std::printf("kernel=%u cases=%llu mismatches=%llu midpoints=%llu\\n",i,(unsigned long long)cases,(unsigned long long)fail,(unsigned long long)ties);}}
'''
(folder/'middle-probe.cpp').write_text(source);binary='/private/tmp/llm-inner-regional-root-middle-probe'
subprocess.run(['clang++','-O3','-ffp-contract=off','-std=c++17',str(folder/'middle-probe.cpp'),'-o',binary],check=True)
r=subprocess.run([binary],check=True,capture_output=True,text=True);(folder/'middle-probe.log').write_text(r.stdout);print(r.stdout)
