#include <cstdint>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <cfenv>
#include <initializer_list>
template<class T,class U>T word(U x){T y;std::memcpy(&y,&x,sizeof(y));return y;}
double f2(double m,unsigned parity){double z=m-1.046875;return (((((5.115760004435847)+(-42.49736561224015/(z+10.962622678643505))))+(-0.3455137406727387/(z+1.599440032849224))))*(parity?1.4142135623730951:1.0);}
double f3(double m,unsigned parity){double z=m-1.046875;return (((((((7.162069704907793)+(-118.63538018768996/(z+21.141638478994864))))+(-1.2378066366763951/(z+2.6929054335936766))))+(-0.08742419447549843/(z+1.2896398163904605))))*(parity?1.4142135623730951:1.0);}
int main(){std::fesetround(FE_TONEAREST);unsigned degree=2;
for(auto f:{f2,f3}){unsigned cases=0,fail=0,ties=0;
for(unsigned m=262144;m<=524288;m++)for(unsigned p=0;p<2;p++){
 double x=double(word<float>(uint32_t((127u<<23)|m)));double raw=f(x,p);
 bool mismatch=word<uint32_t>(float(raw))!=word<uint32_t>(float(std::sqrt(x*(p?2.0:1.0))));
 bool tie=(word<uint64_t>(raw)&UINT64_C(536870911))==UINT64_C(268435456);
 if(mismatch||tie)std::printf("bad %u %u %u %u %u\n",degree,m,p,mismatch,tie);
 fail+=mismatch;ties+=tie;cases++;}
std::printf("summary %u %u %u %u\n",degree,cases,fail,ties);degree++;}}
