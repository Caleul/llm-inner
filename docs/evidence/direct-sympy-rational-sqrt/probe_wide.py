import sys,json,subprocess,tempfile
from pathlib import Path
import mpmath as mp
import sympy as sp
sys.path.insert(0,str(Path('helpers').resolve()))
from direct_sympy_conversions_test import cpp
from direct_sympy_strings import StringCompiler,Domain,syntax
from direct_sympy_conversions import ConversionSession,FiniteSource,lower_finite_conversion
from direct_sympy_words import simplify_words
from fractions import Fraction as F
mp.mp.dps=100
root=Path('artifacts/direct-sympy-input-partitions/rational-sqrt/wide')
root.mkdir(exist_ok=True)
for n in (6,):
 nodes=[mp.mpf('1.5')*mp.cos(mp.pi*i/(2*n)) for i in range(2*n+1)]
 a=mp.matrix([[z**j for j in range(n+1)]+[-mp.sqrt(2*(z+mp.mpf('2.5')))*z**j for j in range(1,n+1)] for z in nodes]);b=mp.matrix([mp.sqrt(2*(z+mp.mpf('2.5'))) for z in nodes]);sol=mp.lu_solve(a,b)
 numerator=[sol[j] for j in range(n+1)];denominator=[mp.mpf(1)]+[sol[n+j] for j in range(1,n+1)]
 maxerror=max(abs(mp.polyval(list(reversed(numerator)),z)/mp.polyval(list(reversed(denominator)),z)-mp.sqrt(2*(z+mp.mpf('2.5')))) for z in [mp.mpf(3*i)/1000-mp.mpf('1.5') for i in range(1001)])
 z=sp.Symbol('z');groups=[]
 for coefficients in (numerator,denominator):
  roots=sorted(sp.nroots(sp.Poly.from_list([sp.Float(mp.nstr(c,100),100) for c in reversed(coefficients)],z),n=70,maxsteps=500),reverse=True)
  factors=[]
  if n%2:factors.append((float(-roots.pop()),))
  for i in range(0,len(roots),2):
   shift=-(roots[i]+roots[i+1])/2;tail=roots[i]*roots[i+1]-shift*shift
   factors.append(tuple(map(float,(shift,tail))))
  groups.append(factors)
 coefficient=float(numerator[-1]/denominator[-1])
 raw='Bits64(X999999997)';m=f'Float64(U64Add(U64And({raw},9007199254740991),4607182418800017408))';ztext=f'(({m}) - 2.5)'
 def product(factors):
  parts=[f'(({ztext}) + {f[0]!r})' if len(f)==1 else f'((({ztext}) + {f[0]!r}) ** 2 + ({f[1]!r}))' for f in factors]
  out=parts[0]
  for part in parts[1:]:out=f'(({out}) * ({part}))'
  return out
 seed=f'(({coefficient!r} * ({product(groups[0])})) / ({product(groups[1])}))'
 exponent=f'U64And(U64Shr({raw},52),2047)';p=seed
 session=ConversionSession(StringCompiler(),{'X1':Domain(F(2)**-149,F(3.4028234663852886e38),-149,True)})
 candidate=session.compiler.substitute(p,'X999999997','X1',session.domains)
 rounded=lower_finite_conversion(candidate,'R32',FiniteSource(0.9,3.0,-53),session.compiler,session.domains,no_odd_f32_ties=True)
 target=f'U64Shr(U64Add({exponent},1022),1)';adjustment=f'U64Mul(U64Add({target},18446744073709550593),4503599627370496)';adjustment=session.compiler.substitute(adjustment,'X999999997','X1',session.domains)
 result=simplify_words(f'Float64(U64Add(Bits64({rounded}),{adjustment}))',session.compiler,session.domains)
 (root/f'candidate-{n}.expr').write_text(result)
 header='#include <cstdint>\n#include <cstring>\n#include <cmath>\n#include <cstdio>\n#include <initializer_list>\ntemplate<class T,class U>T word(U value){T result;std::memcpy(&result,&value,sizeof(result));return result;}\n'
 c=root/f'candidate-{n}.cpp';binary=root/f'candidate-{n}'
 c.write_text(header+'double compiled(double X1){return '+cpp(syntax(result))+';}\ndouble approximate(double X1){return '+cpp(syntax(candidate))+';}\n'+'''int main(){unsigned cases=0,mismatches=0,ties=0;for(unsigned b=0;b<8388608;b++)for(unsigned e=127;e<129;e++){double x=word<float>(uint32_t((e<<23)|b));uint64_t p=word<uint64_t>(approximate(x));ties+=(p&UINT64_C(536870911))==UINT64_C(268435456);if(word<uint64_t>(compiled(x))!=word<uint64_t>(double(float(std::sqrt(x))))){if(mismatches<8)std::printf("mismatch=%a\\n",x);mismatches++;}cases++;}for(unsigned b=1;b<8388608;b++){double x=word<float>(uint32_t(b));mismatches+=word<uint64_t>(compiled(x))!=word<uint64_t>(double(float(std::sqrt(x))));cases++;}for(unsigned e=1;e<=254;e++)for(unsigned b:{0u,1u,2u,4194303u,4194304u,8388606u,8388607u}){double x=word<float>(uint32_t((e<<23)|b));mismatches+=word<uint64_t>(compiled(x))!=word<uint64_t>(double(float(std::sqrt(x))));cases++;}std::printf("cases=%u mismatches=%u ties=%u\\n",cases,mismatches,ties);return mismatches||ties?1:0;}''')
 subprocess.run(['clang++','-O3','-ffp-contract=off','-std=c++17',str(c),'-o',str(binary)],check=True)
 run=subprocess.run([str(binary)],capture_output=True,text=True,timeout=120)
 report={'degree':n,'maxInterpolationError':str(maxerror),'coefficient':coefficient,'factors':groups,'inputOccurrences':result.count('X1'),'characters':len(result),'exitCode':run.returncode,'native':run.stdout}
 (root/f'candidate-{n}.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
