#include <simd_stl/numeric/BasicSimd.h>
#include <string>

int main() {
    simd_stl::numeric::basic_simd<simd_stl::arch::CpuFeature::SSE> simd;
   // //std::string str;
   // str = str + str;

    simd = simd + simd;

    return 0;
}
