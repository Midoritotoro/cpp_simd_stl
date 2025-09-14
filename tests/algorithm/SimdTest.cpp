#include <simd_stl/numeric/BasicSimd.h>
#include <string>

void testSse() {
    simd_stl::numeric::basic_simd<simd_stl::arch::CpuFeature::SSE2, int> simd(1,2,3,4);
    simd = simd + simd;
    auto other = simd_stl::numeric::basic_simd<simd_stl::arch::CpuFeature::SSE, int>(simd);
}

int main() {
    testSse();
    return 0;
}
