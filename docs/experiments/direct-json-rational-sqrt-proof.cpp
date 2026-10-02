// Isolated certificate experiment; the production root is unchanged.
// clang++ -O3 -ffp-contract=off -std=c++17 FILE -o PROOF
// Normalize positive normal F32 values into [1,4), covering both exponent
// parities. Exact power-of-two covariance must be documented before admission.
#include <cmath>
#include <cfenv>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <limits>
#ifdef __FAST_MATH__
#error This certificate requires exact IEEE operations without fast math
#endif
static uint32_t bits(float x) {
  uint32_t result; std::memcpy(&result, &x, sizeof result); return result;
}
static float value(uint32_t x) {
  float result; std::memcpy(&result, &x, sizeof result); return result;
}
int main() {
  static_assert(sizeof(float)==4 && sizeof(double)==8);
  static_assert(std::numeric_limits<float>::is_iec559 &&
                std::numeric_limits<double>::is_iec559);
  if(std::fegetround()!=FE_TONEAREST) return 2;
  constexpr double a=0x1.b8b124936d913p+0, b=0x1.d07b36ff85ce5p+1, c=0x1.166c4d8faa3bcp+2;
  uint64_t tested=0, mismatches=0; uint32_t first=0;
  for(uint32_t exponent=127; exponent<=128; ++exponent) {
    for(uint32_t fraction=0; fraction<0x800000; ++fraction) {
      const float x=value((exponent<<23)|fraction);
      const double m=value((127u<<23)|fraction);
      double y=(a+b*m)/(c+m);
      for(int step=0; step<2; ++step) y=.5*(y+m/y);
      y*=exponent==127 ? 1.0 : 0x1.6a09e667f3bcdp+0;
      const float actual=static_cast<float>(y), reference=std::sqrt(x);
      ++tested;
      if(bits(actual)!=bits(reference)) {
        if(!mismatches) first=bits(x);
        ++mismatches;
      }
    }
  }
  std::printf("{\"tested\":%llu,\"mismatches\":%llu,\"firstMismatch\":%u}\n",
              static_cast<unsigned long long>(tested),
              static_cast<unsigned long long>(mismatches), first);
  return mismatches ? 1 : 0;
}
