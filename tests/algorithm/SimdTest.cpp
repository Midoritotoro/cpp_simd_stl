#include <simd_stl/numeric/BasicSimd.h>
#include <string>

void testSse() {
    simd_stl::numeric::basic_simd<simd_stl::arch::CpuFeature::SSE2, int> simd(51);
    simd = simd + simd;
    auto other = simd_stl::numeric::basic_simd<simd_stl::arch::CpuFeature::SSE2>
        ::safeCast<simd_stl::numeric::basic_simd<simd_stl::arch::CpuFeature::SSE, float>>(simd);

    int array[4];  
    simd.storeUnaligned(array);

    for (int i = 0; i < 4; ++i)
        std::cout << array[i] << " ";
}

int main() {
    testSse();
    return 0;
}
