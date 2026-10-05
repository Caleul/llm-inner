#include <cstdint>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <cfenv>
template<class T,class U>T word(U x){T y;std::memcpy(&y,&x,sizeof(y));return y;}
double f0(double m,unsigned p){double z=m-1.375;return (((((((((10.497874058371051)+(-376.0442911765619/(z+45.10733469861704))))+(-4.215191385696203/(z+5.4350185108224265))))+(-0.41741505225839626/(z+2.3209845535576568))))+(-0.05153593636688096/(z+1.5518421097128303))))*(p?1.4142135623730951:1.0);}
double f1(double m,unsigned p){double z=m-1.390625;return (((((((((10.56315855446766)+(-383.1172224793218/(z+45.671534992193116))))+(-4.296110039636687/(z+5.503592872701465))))+(-0.42586091672596127/(z+2.349683751626839))))+(-0.05264980863531023/(z+1.5700352068902454))))*(p?1.4142135623730951:1.0);}
double f2(double m,unsigned p){double z=m-1.515625;return (((((((((10.99927028612513)+(-432.48370404191843/(z+49.5133234289682))))+(-4.841004117977899/(z+5.963514971813368))))+(-0.47757748408173384/(z+2.5490162199256945))))+(-0.058666199681106014/(z+1.7083062929103427))))*(p?1.4142135623730951:1.0);}
double f3(double m,unsigned p){double z=m-1.53125;return (((((((((11.062680649586998)+(-440.0242487257263/(z+50.08761928378408))))+(-4.9275340059310855/(z+6.033422660407277))))+(-0.48667768438519676/(z+2.578174017147084))))+(-0.059877234960313865/(z+1.7266106160348669))))*(p?1.4142135623730951:1.0);}
int main(){std::fesetround(FE_TONEAREST);double (*fs[])(double,unsigned)={f0,f1,f2,f3};unsigned ranges[][2]={{0,6291456},{262144,6291456},{262144,8388607},{524288,8388607}};
for(unsigned i=0;i<sizeof(fs)/sizeof(fs[0]);i++){uint64_t cases=0,fail=0,ties=0;for(unsigned m=ranges[i][0];m<=ranges[i][1];m++)for(unsigned p=0;p<2;p++){
 double x=word<float>(uint32_t((127u<<23)|m));double raw=fs[i](x,p);
 fail+=word<uint32_t>(float(raw))!=word<uint32_t>(float(std::sqrt(x*(p?2.0:1.0))));
 ties+=(word<uint64_t>(raw)&UINT64_C(536870911))==UINT64_C(268435456);cases++;
}std::printf("kernel=%u cases=%llu mismatches=%llu midpoints=%llu\n",i,(unsigned long long)cases,(unsigned long long)fail,(unsigned long long)ties);}}
