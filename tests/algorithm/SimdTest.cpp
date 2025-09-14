#include <simd_stl/numeric/BasicSimd.h>
#include <string>

void testSse() {
    simd_stl::numeric::basic_simd<simd_stl::arch::CpuFeature::SSE2, int> simd;
    simd = simd + simd;
}

int main() {
    testSse();
    return 0;
}
