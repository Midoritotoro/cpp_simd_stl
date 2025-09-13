#include <simd_stl/numeric/BasicSimd.h>
#include <string>


int main() {
    simd_stl::numeric::basic_simd<simd_stl::arch::CpuFeature::SSE> simd;
    simd = simd + simd;


    return 0;
}
