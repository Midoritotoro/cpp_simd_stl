#pragma once

#include <bitset>
#include <vector>

#include <array>

#include <simd_stl/Types.h>
#include <simd_stl/arch/CpuId.h>

#include <simd_stl/compatibility/Nodiscard.h>
#include <simd_stl/compatibility/Inline.h>

__SIMD_STL_ARCH_NAMESPACE_BEGIN

class ProcessorFeatures
{
public:
    simd_stl_nodiscard simd_stl_always_inline static bool SSE()         noexcept;
    simd_stl_nodiscard simd_stl_always_inline static bool SSE2()        noexcept;
    simd_stl_nodiscard simd_stl_always_inline static bool SSE3()        noexcept;
    simd_stl_nodiscard simd_stl_always_inline static bool SSSE3()       noexcept;
    simd_stl_nodiscard simd_stl_always_inline static bool SSE41()       noexcept;
    simd_stl_nodiscard simd_stl_always_inline static bool SSE42()       noexcept;
    simd_stl_nodiscard simd_stl_always_inline static bool AVX()         noexcept;
    simd_stl_nodiscard simd_stl_always_inline static bool AVX2()        noexcept;
    simd_stl_nodiscard simd_stl_always_inline static bool AVX512F()     noexcept;
    simd_stl_nodiscard simd_stl_always_inline static bool AVX512BW()    noexcept;
    simd_stl_nodiscard simd_stl_always_inline static bool AVX512PF()    noexcept;
    simd_stl_nodiscard simd_stl_always_inline static bool AVX512ER()    noexcept;
    simd_stl_nodiscard simd_stl_always_inline static bool AVX512CD()    noexcept;
    simd_stl_nodiscard simd_stl_always_inline static bool AVX512VL()    noexcept;
    
    simd_stl_nodiscard simd_stl_always_inline static bool POPCNT()      noexcept;
private:
    class ProcessorFeaturesInternal
    {
    public:
        ProcessorFeaturesInternal() noexcept;

        bool _sse      : 1 = false;
        bool _sse2     : 1 = false;
        bool _sse3     : 1 = false;
        bool _sse41    : 1 = false;
        bool _sse42    : 1 = false;
        bool _ssse3    : 1 = false;
        
        bool _avx      : 1 = false;
        bool _avx2     : 1 = false;

        bool _avx512f  : 1 = false;
        bool _avx512bw : 1 = false;
        bool _avx512pf : 1 = false;
        bool _avx512er : 1 = false;
        bool _avx512cd : 1 = false;
        bool _avx512vl : 1 = false;

        bool _popcnt   : 2 = false;
    };

    static inline ProcessorFeaturesInternal _processorFeaturesInternal;
};

ProcessorFeatures::ProcessorFeaturesInternal::ProcessorFeaturesInternal() noexcept {
    std::array<uint32, 4> registers;

    cpuid(registers.data(), 0);
    const auto leafCount = registers[0];
 
    if (leafCount >= 1) {
        memset(registers.data(), 0, registers.size() * sizeof(uint32));
        cpuidex(registers.data(), 1, 0); // 0 - eax, 1 - ebx, 2 - ecx, 3 - edx

        const auto leaf1Ecx = registers[2];
        const auto leaf1Edx = registers[3];

        _sse    = (leaf1Edx >> 25) & 1;
        _sse2   = (leaf1Edx >> 26) & 1;
        
        _sse3   = (leaf1Ecx & 1);
        _ssse3  = (leaf1Ecx >> 9) & 1;
        _sse41  = (leaf1Ecx >> 19) & 1;
        _sse42  = (leaf1Ecx >> 20) & 1;
        
        _popcnt = (leaf1Ecx >> 23) & 1;
        _avx    = (leaf1Ecx >> 28) & 1;
    }

    if (leafCount >= 7) {
        memset(registers.data(), 0, registers.size() * sizeof(uint32));
        cpuidex(registers.data(), 7, 0); // 0 - eax, 1 - ebx, 2 - ecx, 3 - edx

        const auto leaf7Ebx = registers[1];
        
        _avx2       = (leaf7Ebx >> 5) & 1;

        _avx512f    = (leaf7Ebx >> 16) & 1;
        _avx512bw   = (leaf7Ebx >> 30) & 1;
        _avx512pf   = (leaf7Ebx >> 26) & 1;
        _avx512er   = (leaf7Ebx >> 27) & 1;
        _avx512cd   = (leaf7Ebx >> 28) & 1;
        _avx512vl   = (leaf7Ebx >> 31) & 1;
    }
}

bool ProcessorFeatures::SSE() noexcept {
    return _processorFeaturesInternal._sse;
}

bool ProcessorFeatures::SSE2() noexcept {
    return _processorFeaturesInternal._sse2;
}

bool ProcessorFeatures::SSE3() noexcept {
    return _processorFeaturesInternal._sse3;
}

bool ProcessorFeatures::SSSE3() noexcept {
    return _processorFeaturesInternal._ssse3;
}

bool ProcessorFeatures::SSE41() noexcept {
    return _processorFeaturesInternal._sse41;
}

bool ProcessorFeatures::SSE42() noexcept {
    return _processorFeaturesInternal._sse42;
}

bool ProcessorFeatures::AVX() noexcept {
    return _processorFeaturesInternal._avx;
}

bool ProcessorFeatures::AVX2() noexcept {
    return _processorFeaturesInternal._avx2;
}

bool ProcessorFeatures::AVX512F() noexcept {
    return _processorFeaturesInternal._avx512f;
}

bool ProcessorFeatures::AVX512BW() noexcept {
    return _processorFeaturesInternal._avx512bw;
}

bool ProcessorFeatures::AVX512PF() noexcept {
    return _processorFeaturesInternal._avx512pf;
}

bool ProcessorFeatures::AVX512ER() noexcept {
    return _processorFeaturesInternal._avx512er;
}

bool ProcessorFeatures::AVX512CD() noexcept {
    return _processorFeaturesInternal._avx512cd;
}

bool ProcessorFeatures::AVX512VL() noexcept {
    return _processorFeaturesInternal._avx512vl;
}

bool ProcessorFeatures::POPCNT() noexcept {
    return _processorFeaturesInternal._popcnt;
}

__SIMD_STL_ARCH_NAMESPACE_END
