#pragma once 

#include <numeric>

#include <simd_stl/compatibility/CompilerDetection.h>
#include <simd_stl/compatibility/CallingConventions.h>

#include <simd_stl/SimdStlNamespace.h>
#include <simd_stl/compatibility/SimdCompatibility.h>

#if (defined(simd_stl_cpp_clang) || defined(simd_stl_cpp_gnu)) && !defined(simd_stl_cpp_msvc)
#  include <cpuid.h>
#endif // (defined(simd_stl_cpp_clang) || defined(simd_stl_cpp_gnu)) && !defined(simd_stl_cpp_msvc)


__SIMD_STL_ARCH_NAMESPACE_BEGIN

void cpuid(
	uint32 regs[4],
	uint32 leaf) noexcept
{
#if (defined(simd_stl_cpp_clang) || defined(simd_stl_cpp_gnu)) && !defined(simd_stl_cpp_msvc)
	__get_cpuid(leaf, regs, regs + 1, regs + 2, regs + 3);
#else
	__cpuid((int*)regs, leaf);
#endif // (defined(simd_stl_cpp_clang) || defined(simd_stl_cpp_gnu)) && !defined(simd_stl_cpp_msvc)
}

void cpuidex(
    uint32 regs[4],
    uint32 leaf, 
    uint32 subleaf) noexcept
{
#if defined(simd_stl_cpp_msvc) || \
    (defined(simd_stl_cpp_clang) && simd_stl_cpp_clang >= 1810) || \
    (defined(simd_stl_cpp_gnu) && simd_stl_cpp_gnu >= 1100)
    __cpuidex((int*)regs, leaf, subleaf);
#else
    uint32* eax = &regs[0], *ebx = &regs[1], *ecx = &regs[2], *edx = &regs[3];
    __asm__ __volatile__(
        "cpuid"
        : "=a"(*eax), "=b"(*ebx), "=c"(*ecx), "=d"(*edx)
        : "a"(leaf), "c"(subleaf)
    );
#endif
}

__SIMD_STL_ARCH_NAMESPACE_END
