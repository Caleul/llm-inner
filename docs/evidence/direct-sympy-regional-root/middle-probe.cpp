#include <cstdint>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <cfenv>
template<class T,class U>T word(U x){T y;std::memcpy(&y,&x,sizeof(y));return y;}
double f0(double m,unsigned p){double z=m-1.3125;return (((((((((10.26961342316249)+(-352.0731056594946/(z+43.17020647328195))))+(-3.949980852102898/(z+5.20291359370326))))+(-0.3920750181688043/(z+2.220581723885396))))+(-0.04855891165065189/(z+1.482528139728993))))*(p?1.4142135623730951:1.0);}
double f1(double m,unsigned p){double z=m-1.359375;return (((((((((10.450208827566483)+(-370.9730120141941/(z+44.70160165214271))))+(-4.16169735164847/(z+5.387359610060382))))+(-0.41300404655780115/(z+2.2994202929922345))))+(-0.05113691803064475/(z+1.535363214678897))))*(p?1.4142135623730951:1.0);}
double f2(double m,unsigned p){double z=m-1.35546875;return (((((((((10.43999777995661)+(-369.8975830992561/(z+44.615455438834054))))+(-4.150956227250936/(z+5.377462316167927))))+(-0.41228757032525376/(z+2.294716394288787))))+(-0.051105444051296745/(z+1.531408298458699))))*(p?1.4142135623730951:1.0);}
int main(){std::fesetround(FE_TONEAREST);double (*fs[])(double,unsigned)={f0,f1,f2};unsigned ranges[][2]={{0,5242880},{262144,5767168},{393216,5570560}};
for(unsigned i=0;i<sizeof(fs)/sizeof(fs[0]);i++){uint64_t cases=0,fail=0,ties=0;for(unsigned m=ranges[i][0];m<=ranges[i][1];m++)for(unsigned p=0;p<2;p++){
 double x=word<float>(uint32_t((127u<<23)|m));double raw=fs[i](x,p);
 fail+=word<uint32_t>(float(raw))!=word<uint32_t>(float(std::sqrt(x*(p?2.0:1.0))));
 ties+=(word<uint64_t>(raw)&UINT64_C(536870911))==UINT64_C(268435456);cases++;
}std::printf("kernel=%u cases=%llu mismatches=%llu midpoints=%llu\n",i,(unsigned long long)cases,(unsigned long long)fail,(unsigned long long)ties);}}
