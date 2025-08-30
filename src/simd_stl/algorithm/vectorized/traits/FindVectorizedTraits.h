#pragma once 

#include <simd_stl/compatibility/SimdCompatibility.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

SIMD_STL_DECLARE_CPU_FEATURE_GUARDED_CLASS(
    template <arch::CpuFeature feature> class FindTraits,
    feature,
    "simd_stl::algorithm",
    arch::CpuFeature::SSE2, arch::CpuFeature::AVX2, arch::CpuFeature::AVX512F
);


template <>
class FindTraits<arch::CpuFeature::SSE2> {
public:
    static constexpr bool useMaskedLoadForTail = false;
    static constexpr size_t portionSize = 0xF;

    simd_stl_declare_const_function simd_stl_always_inline static __m128i LoadAligned(const void* address) noexcept {
        return _mm_load_si128(reinterpret_cast<const __m128i*>(address));
    }

    simd_stl_declare_const_function simd_stl_always_inline static __m128i LoadUnaligned(const void* address) noexcept {
        return _mm_loadu_si128(reinterpret_cast<const __m128i*>(address));
    }

    simd_stl_declare_const_function simd_stl_always_inline static __m128i Set(const uint8 value) noexcept {
        return _mm_set1_epi8(value);
    }

    simd_stl_declare_const_function simd_stl_always_inline static __m128i Set(const uint16 value) noexcept {
        return _mm_set1_epi16(value);
    }

    simd_stl_declare_const_function simd_stl_always_inline static __m128i Set(const uint32 value) noexcept {
        return _mm_set1_epi32(value);
    }

    simd_stl_declare_const_function simd_stl_always_inline static __m128i Set(const uint64 value) noexcept {
        return _mm_set1_epi64x(value);
    }

    

    template <size_t singleElementSize>
    simd_stl_declare_const_function simd_stl_always_inline static __m128i Compare(
        const __m128i left,
        const __m128i right) noexcept
    {
        static_assert(
            singleElementSize == 1 || singleElementSize == 2 || singleElementSize == 4 || singleElementSize == 8,
            "base::algorithm::FindTraits<arch::CpuFeature::AVX2>::Compare: Unsupported element size");

        if      constexpr (singleElementSize == 1)
            return _mm_cmpeq_epi8(left, right);
        else if constexpr (singleElementSize == 2) 
            return _mm_cmpeq_epi16(left, right);
        else if constexpr (singleElementSize == 4)
            return _mm_cmpeq_epi32(left, right);
        else if constexpr (singleElementSize == 8)
            return _mm_cmpeq_epi64(left, right);

        return {};
    }
};

template <>
class FindTraits<arch::CpuFeature::AVX2> {
public:
    static constexpr bool useMaskedLoadForTail = true;
    static constexpr size_t portionSize = 0x1F;

    simd_stl_declare_const_function simd_stl_always_inline static __m256i LoadAligned(const void* address) noexcept {
        return _mm256_load_si256(reinterpret_cast<const __m256i*>(address));
    }

    simd_stl_declare_const_function simd_stl_always_inline static __m256i LoadUnaligned(const void* address) noexcept {
        return _mm256_lddqu_si256(reinterpret_cast<const __m256i*>(address));
    }

    simd_stl_declare_const_function simd_stl_always_inline static __m256i Set(const uint8 value) noexcept {
        return _mm256_set1_epi8(value);
    }

    simd_stl_declare_const_function simd_stl_always_inline static __m256i Set(const uint16 value) noexcept {
        return _mm256_set1_epi16(value);
    }

    simd_stl_declare_const_function simd_stl_always_inline static __m256i Set(const uint32 value) noexcept {
        return _mm256_set1_epi32(value);
    }

    simd_stl_declare_const_function simd_stl_always_inline static __m256i Set(const uint64 value) noexcept {
        return _mm256_set1_epi64x(value);
    }

    template <size_t singleElementSize>
    simd_stl_declare_const_function simd_stl_always_inline static __m256i Compare(
        const __m256i left,
        const __m256i right) noexcept
    {
        static_assert(
            singleElementSize == 1 || singleElementSize == 2 || singleElementSize == 4 || singleElementSize == 8,
            "base::algorithm::FindTraits<arch::CpuFeature::AVX2>::Compare: Unsupported element size");

        if      constexpr (singleElementSize == 1) 
            return _mm256_cmpeq_epi8(left, right);
        else if constexpr (singleElementSize == 2) 
            return _mm256_cmpeq_epi16(left, right);
        else if constexpr (singleElementSize == 4) 
            return _mm256_cmpeq_epi32(left, right);
        else if constexpr (singleElementSize == 8)
            return _mm256_cmpeq_epi64(left, right);

        return {};
    }
};

template <>
class FindTraits<arch::CpuFeature::AVX512F> {
public:
    static constexpr bool useMaskedLoadForTail = true;
    static constexpr size_t portionSize = 0x3F;

    simd_stl_declare_const_function simd_stl_always_inline static __m512i LoadAligned(const void* address) noexcept {
        return _mm512_load_si512(reinterpret_cast<const __m512i*>(address));
    }

    simd_stl_declare_const_function simd_stl_always_inline static __m512i LoadUnaligned(const void* address) noexcept {
        return _mm512_loadu_si512(reinterpret_cast<const __m512i*>(address));
    }

    simd_stl_declare_const_function simd_stl_always_inline static __m512i Set(const uint8 value) noexcept {
        return _mm512_set1_epi8(value);
    }

    simd_stl_declare_const_function simd_stl_always_inline static __m512i Set(const uint16 value) noexcept {
        return _mm512_set1_epi16(value);
    }

    simd_stl_declare_const_function simd_stl_always_inline static __m512i Set(const uint32 value) noexcept {
        return _mm512_set1_epi32(value);
    }

    simd_stl_declare_const_function simd_stl_always_inline static __m512i Set(const uint64 value) noexcept {
        return _mm512_set1_epi64(value);
    }

    template <size_t singleElementSize>
    simd_stl_declare_const_function simd_stl_always_inline static auto Compare(
        const __m512i left,
        const __m512i right) noexcept
    {
        static_assert(
            singleElementSize == 1 || singleElementSize == 2 || singleElementSize == 4 || singleElementSize == 8,
            "base::algorithm::FindTraits<arch::CpuFeature::AVX512F>::Compare: Unsupported element size");

        if      constexpr (singleElementSize == 1)
            return _mm512_cmpeq_epi8_mask(left, right);
        else if constexpr (singleElementSize == 2)
            return _mm512_cmpeq_epi16_mask(left, right);
        else if constexpr (singleElementSize == 4)
            return _mm512_cmpeq_epi32_mask(left, right);
        else if constexpr (singleElementSize == 8)
            return _mm512_cmpeq_epi64_mask(left, right);

        return {};
    }
};

__SIMD_STL_ALGORITHM_NAMESPACE_END
