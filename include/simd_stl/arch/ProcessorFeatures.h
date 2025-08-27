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
    simd_stl_nodiscard simd_stl_always_inline static bool SSE()        noexcept;
    simd_stl_nodiscard simd_stl_always_inline static bool SSE2()       noexcept;
    simd_stl_nodiscard simd_stl_always_inline static bool SSE3()       noexcept;
    simd_stl_nodiscard simd_stl_always_inline static bool SSSE3()      noexcept;
    simd_stl_nodiscard simd_stl_always_inline static bool SSE41()      noexcept;
    simd_stl_nodiscard simd_stl_always_inline static bool SSE42()      noexcept;
    simd_stl_nodiscard simd_stl_always_inline static bool AVX()        noexcept;
    simd_stl_nodiscard simd_stl_always_inline static bool AVX2()       noexcept;
    simd_stl_nodiscard simd_stl_always_inline static bool AVX512F()    noexcept;
    simd_stl_nodiscard simd_stl_always_inline static bool AVX512BW()   noexcept;
    simd_stl_nodiscard simd_stl_always_inline static bool AVX512PF()   noexcept;
    simd_stl_nodiscard simd_stl_always_inline static bool AVX512ER()   noexcept;
    simd_stl_nodiscard simd_stl_always_inline static bool AVX512CD()   noexcept;
    simd_stl_nodiscard simd_stl_always_inline static bool AVX512VL()   noexcept;
private:
    class ProcessorFeaturesInternal
    {
    public:
        ProcessorFeaturesInternal() noexcept;

        std::bitset<32> _leaf1EcxBitset;
        std::bitset<32> _leaf1EdxBitset;
        std::bitset<32> _leaf7EbxBitset;
    };

    static inline ProcessorFeaturesInternal _processorFeaturesInternal;
};

ProcessorFeatures::ProcessorFeaturesInternal::ProcessorFeaturesInternal() noexcept {
    std::array<uint32, 4>               registers;
    std::vector<std::array<uint32, 4>>  data;

    cpuid(registers.data(), 0);
    const auto leafNumber = registers[0];

    for (int i = 0; i <= leafNumber; ++i) {
        cpuidex(registers.data(), i, 0);
        data.push_back(registers);
    }
 
    if (leafNumber >= 1) {
        _leaf1EcxBitset = data[1][2]; // leaf 1
        _leaf1EdxBitset = data[1][3]; // leaf 1
    }

    if (leafNumber >= 7)
        _leaf7EbxBitset = data[7][1]; // leaf 7
}

bool ProcessorFeatures::SSE() noexcept {
    return _processorFeaturesInternal._leaf1EdxBitset[25];
}

bool ProcessorFeatures::SSE2() noexcept {
    return _processorFeaturesInternal._leaf1EdxBitset[26];
}

bool ProcessorFeatures::SSE3() noexcept {
    return _processorFeaturesInternal._leaf1EcxBitset[0];
}

bool ProcessorFeatures::SSSE3() noexcept {
    return _processorFeaturesInternal._leaf1EcxBitset[9];
}

bool ProcessorFeatures::SSE41() noexcept {
    return _processorFeaturesInternal._leaf1EcxBitset[19];
}

bool ProcessorFeatures::SSE42() noexcept {
    return _processorFeaturesInternal._leaf1EcxBitset[20];
}

bool ProcessorFeatures::AVX() noexcept {
    return _processorFeaturesInternal._leaf1EcxBitset[28];
}

bool ProcessorFeatures::AVX2() noexcept {
    return _processorFeaturesInternal._leaf7EbxBitset[5];
}

bool ProcessorFeatures::AVX512F() noexcept {
    return _processorFeaturesInternal._leaf7EbxBitset[16];
}

bool ProcessorFeatures::AVX512BW() noexcept {
    return _processorFeaturesInternal._leaf7EbxBitset[30];
}

bool ProcessorFeatures::AVX512PF() noexcept {
    return _processorFeaturesInternal._leaf7EbxBitset[26];
}

bool ProcessorFeatures::AVX512ER() noexcept {
    return _processorFeaturesInternal._leaf7EbxBitset[27];
}

bool ProcessorFeatures::AVX512CD() noexcept {
    return _processorFeaturesInternal._leaf7EbxBitset[28];
}

bool ProcessorFeatures::AVX512VL() noexcept {
    return _processorFeaturesInternal._leaf7EbxBitset[31];
}

__SIMD_STL_ARCH_NAMESPACE_END
