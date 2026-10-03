#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdio>
template<class T,class U>T bits(U x){T y;std::memcpy(&y,&x,sizeof(y));return y;}
int main(){unsigned errors=0,ties=0;double maxerror=0;for(uint32_t i=0;i<16777216;i++){double x=bits<float>(uint32_t((127u<<23)+i));double m=x<2?x:x/2;double z=m-1.5;double seed=(1.224744871391589+z*(1.0294870565343948+z*0.17419039766421152))/(1+z*(0.507247903505105+z*0.02869157717198823));double p=0.5*(seed+m/seed)*(x<2?1:1.4142135623730951);auto raw=bits<uint64_t>(p);ties+=(raw&536870911)==268435456;if(bits<uint32_t>(float(p))!=bits<uint32_t>(std::sqrt(float(x)))){if(errors<8)std::printf("mismatch %u x=%.17g p=%.17g ref=%.17g\n",i,x,p,double(std::sqrt(float(x))));errors++;}maxerror=std::fmax(maxerror,std::fabs(p-std::sqrt(x)));}std::printf("cases=16777216 mismatches=%u ties=%u maxAbsError=%.17g\n",errors,ties,maxerror);return errors||ties?1:0;}
