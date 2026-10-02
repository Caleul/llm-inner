"""Execute a saved RMS-inverse or first V-projection string against native arithmetic.

This is a producer certificate, not output-coordinate/model parity. sqrt is
still present in this prefix and its native interpretation is explicit here.
"""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from direct_sympy_strings import syntax
from direct_sympy_conversions_test import cpp


def main():
    if len(sys.argv) not in (3,4):raise ValueError("Usage: prefix checkpoint [inverse|v0]")
    prefix,checkpoint=sys.argv[1:3]
    producer=sys.argv[3] if len(sys.argv)==4 else "inverse"
    if producer not in ("inverse","v0"):raise ValueError("Unknown producer validation")
    config=json.loads((Path(checkpoint)/"config.json").read_text())
    if config["hidden_size"]!=2:raise ValueError("This native prefix validation fixture requires width two")
    expected="double(inverse)"
    if producer=="v0":
        from direct_sympy_checkpoint import CheckpointStrings
        from direct_sympy_strings import StringCompiler
        with CheckpointStrings(checkpoint,StringCompiler()) as model:
            gamma=[model.weight("model.layers.0.input_layernorm.weight",i) for i in range(2)]
            weights=[model.weight("model.layers.0.self_attn.v_proj.weight",0,i) for i in range(2)]
        expected="double(static_cast<_Float16>((float(0.0f+p0)+float(0.0f+p1))+float(0.0f+0.0f)))"
        recipe="\n".join(f"double n{i}=static_cast<_Float16>(float({('x','y')[i]}*double(inverse))); double h{i}=static_cast<_Float16>(float(n{i}*{gamma[i]})); float p{i}=float(h{i}*{weights[i]});" for i in range(2))
    else:recipe=""
    expression=Path(prefix).read_text()
    if "R32(" in expression or "R16(" in expression:raise ValueError("Residual conversion in saved producer prefix")
    with tempfile.TemporaryDirectory(prefix="sympy-inverse-prefix-") as directory:
        source=Path(directory)/"proof.cpp";binary=Path(directory)/"proof"
        source.write_text("""
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <cfenv>
#include <cfloat>
#include <initializer_list>
template<class T,class U>T word(U value){static_assert(sizeof(T)==sizeof(U));T result;std::memcpy(&result,&value,sizeof(result));return result;}
double candidate(double X1,double X2){return """+cpp(syntax(expression))+""";}
int main(){
 static_assert(FLT_EVAL_METHOD==0);if(std::fesetround(FE_TONEAREST))return 2;
 uint64_t compared=0,failures=0;
 auto check=[&](uint16_t first,uint16_t second){
  double x=word<_Float16>(first),y=word<_Float16>(second);
  float a=float(x)*float(x),b=float(y)*float(y);
  float sum=a+b,mean=sum/2.0f,variance=mean+float("""+repr(config["rms_norm_eps"])+""");
  float root=std::sqrt(variance),inverse=1.0f/root;
  """+recipe+"""
  compared++;if(word<uint64_t>(candidate(x,y))!=word<uint64_t>("""+expected+"""))failures++;
 };
 for(uint32_t bits=0;bits<65536;bits++)if((bits&0x7c00)!=0x7c00)
  for(uint16_t second:{uint16_t(0),uint16_t(0x8000),uint16_t(1),uint16_t(0x8001),uint16_t(0x3c00),uint16_t(0xbc00),uint16_t(0x7bff),uint16_t(0xfbff)})check(bits,second);
 uint32_t random=0x584ecdb3;
 for(unsigned i=0;i<20000;i++){
  random=1664525*random+1013904223;uint16_t a=random&0xffff;
  random=1664525*random+1013904223;uint16_t b=random&0xffff;
  if((a&0x7c00)==0x7c00)a^=0x0400;
  if((b&0x7c00)==0x7c00)b^=0x0400;
  check(a,b);
 }
 std::printf("Native saved-prefix certificate: cases=%llu mismatches=%llu; sqrt remains, finalParity=false\\n",(unsigned long long)compared,(unsigned long long)failures);
 return failures?1:0;
}
""")
        subprocess.run(["clang++","-O3","-ffp-contract=off","-std=c++17",str(source),"-o",str(binary)],check=True,capture_output=True)
        result=subprocess.run([str(binary)],capture_output=True,text=True,timeout=120)
        print("producer="+producer+"; "+result.stdout,end="")
        if result.returncode:raise ValueError(result.stdout+result.stderr)


if __name__=="__main__":main()
