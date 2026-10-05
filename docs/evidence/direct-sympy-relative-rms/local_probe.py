from pathlib import Path
import mpmath as mp,sympy as sp,json,subprocess
root=Path('docs/evidence/direct-sympy-relative-rms')
variants=[]
with mp.workdps(100):
 z=sp.Symbol('z');center=mp.mpf('1.046875');radius=mp.mpf('0.015625')
 for n in (2,3):
  nodes=[radius*mp.cos(mp.pi*i/(2*n))for i in range(2*n+1)]
  A=mp.matrix([[x**j for j in range(n+1)]+[-mp.sqrt(x+center)*x**j for j in range(1,n+1)]for x in nodes])
  s=mp.lu_solve(A,mp.matrix([mp.sqrt(x+center)for x in nodes]))
  num=[s[j]for j in range(n+1)];den=[mp.mpf(1)]+[s[n+j]for j in range(1,n+1)]
  poles=sp.nroots(sp.Poly.from_list([sp.Float(mp.nstr(c,100),100)for c in reversed(den)],z),n=70,maxsteps=500)
  fractions=[]
  for pole in poles:
   r=mp.mpf(str(pole));res=sum(num[i]*r**i for i in range(n+1))/sum(i*den[i]*r**(i-1)for i in range(1,n+1))
   fractions.append((float(res),float(-r)))
  fractions.sort(key=lambda v:abs(v[0]/v[1]),reverse=True)
  variants.append({'degree':n,'constant':float(num[-1]/den[-1]),'fractions':fractions})
(root/'local-coefficients.json').write_text(json.dumps(variants,indent=2)+'\n')
funcs=[]
for v in variants:
 p=repr(v['constant'])
 for a,b in v['fractions']:p=f'(({p})+({a!r}/(z+{b!r})))'
 funcs.append(f'double f{v["degree"]}(double m,unsigned parity){{double z=m-1.046875;return ({p})*(parity?1.4142135623730951:1.0);}}')
src='''#include <cstdint>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <cfenv>
#include <initializer_list>
template<class T,class U>T word(U x){T y;std::memcpy(&y,&x,sizeof(y));return y;}
'''+ '\n'.join(funcs)+'''
int main(){std::fesetround(FE_TONEAREST);unsigned degree=2;
for(auto f:{f2,f3}){unsigned cases=0,fail=0,ties=0;
for(unsigned m=262144;m<=524288;m++)for(unsigned p=0;p<2;p++){
 double x=double(word<float>(uint32_t((127u<<23)|m)));double raw=f(x,p);
 bool mismatch=word<uint32_t>(float(raw))!=word<uint32_t>(float(std::sqrt(x*(p?2.0:1.0))));
 bool tie=(word<uint64_t>(raw)&UINT64_C(536870911))==UINT64_C(268435456);
 if(mismatch||tie)std::printf("bad %u %u %u %u %u\\n",degree,m,p,mismatch,tie);
 fail+=mismatch;ties+=tie;cases++;}
std::printf("summary %u %u %u %u\\n",degree,cases,fail,ties);degree++;}}
'''
(root/'local-probe.cpp').write_text(src);binary=Path('/private/tmp/llm-inner-local-root-probe')
subprocess.run(['clang++','-O3','-ffp-contract=off','-std=c++17',str(root/'local-probe.cpp'),'-o',str(binary)],check=True)
r=subprocess.run([str(binary)],check=True,capture_output=True,text=True)
(root/'local-probe.log').write_text(r.stdout)
print('\n'.join(line for line in r.stdout.splitlines()if line.startswith('summary')))
