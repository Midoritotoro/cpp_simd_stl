#pragma once 

#include <src/simd_stl/numeric/SimdMemoryAccess.h>


__SIMD_STL_NUMERIC_NAMESPACE_BEGIN

template <typename _PointerType_>
using __deduce_pointer_address_type = std::conditional_t<
    std::is_pointer_v<_PointerType_> || std::is_same_v<std::decay_t<_PointerType_>, std::nullptr_t>,
    uintptr_t, _PointerType_>;

template <typename _PointerType_>
constexpr __deduce_pointer_address_type<_PointerType_> pointerAddress(_PointerType_ pointer) noexcept {
    if constexpr (std::is_same_v<std::decay_t<_PointerType_>, std::nullptr_t>)
        return 0;
    else if constexpr (std::is_pointer_v<_PointerType_>)
        return reinterpret_cast<uintptr>(pointer);
    else
        return pointer;
}

template <arch::CpuFeature _SimdGeneration_>
class SimdElementAccess;

template <>
class SimdElementAccess<arch::CpuFeature::SSE2> {
    using _Cast_            = SimdCast<arch::CpuFeature::SSE2>;
    using _MemoryAccess_    = SimdMemoryAccess<arch::CpuFeature::SSE2>;
public:
    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline void insert(
        _VectorType_&       vector,
        const uint8         position,
        const _DesiredType_ value) noexcept
    {
        if constexpr (is_epi64_v<_DesiredType_> || is_epu64_v<_DesiredType_>) {
#if defined(simd_stl_processor_x86_64)
            auto vectorValue = _mm_cvtsi64_si128(value);
#else
            union {
                __m128i vec;
                int64   num;
            } convert;

            convert.num = value;
            auto vectorValue = _mm_loadl_epi64(&convert.vec);
#endif
            if (position == 0) {
                vectorValue = _mm_unpacklo_epi64(vectorValue, vectorValue);
                vector = _Cast_::template cast<__m128i, _VectorType_>(
                    _mm_unpackhi_epi64(vectorValue, _Cast_::template cast<_VectorType_, __m128i>(vector)));
            }
            else
                vector = _Cast_::template cast<__m128i, _VectorType_>(_mm_unpacklo_epi64(
                    _Cast_::template cast<_VectorType_, __m128i>(vector), vectorValue));
        }
        else if constexpr (is_epi32_v<_DesiredType_> || is_epu32_v<_DesiredType_>) {
            const auto broad = _mm_set1_epi32(value);
            const int32 maskArray[8] = { 0,0,0,0,-1,0,0,0 };

            const auto mask = _mm_loadu_si128((__m128i const*)(maskArray + 4 - (position & 3))); // FFFFFFFF at index position
            vector = _Cast_::template cast<__m128i, _VectorType_>(
                _mm_or_si128(_mm_and_si128(mask, broad), 
                    _mm_andnot_si128(mask, _Cast_::template cast<_VectorType_, __m128i>(vector))));
        }
        else if constexpr (is_epi16_v<_DesiredType_> || is_epu16_v<_DesiredType_>) {
            // _mm_insert_epi16 ������� ����� ������� ����������
            switch (position) {
                case 0:
                    vector = _Cast_::template cast<__m128i, _VectorType_>(_mm_insert_epi16(vector, value, 0));
                case 1:
                    vector = _Cast_::template cast<__m128i, _VectorType_>(_mm_insert_epi16(vector, value, 1));
                case 2:
                    vector = _Cast_::template cast<__m128i, _VectorType_>(_mm_insert_epi16(vector, value, 2));
                case 3:
                    vector = _Cast_::template cast<__m128i, _VectorType_>(_mm_insert_epi16(vector, value, 3));
                case 4:
                    vector = _Cast_::template cast<__m128i, _VectorType_>(_mm_insert_epi16(vector, value, 4));
                case 5:
                    vector = _Cast_::template cast<__m128i, _VectorType_>(_mm_insert_epi16(vector, value, 5));
                case 6:
                    vector = _Cast_::template cast<__m128i, _VectorType_>(_mm_insert_epi16(vector, value, 6));
                case 7:
                    vector = _Cast_::template cast<__m128i, _VectorType_>(_mm_insert_epi16(vector, value, 7));
            }

        }
        else if constexpr (is_epi8_v<_DesiredType_> || is_epu8_v<_DesiredType_>) {
            const int8 maskArray[32] = { 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 };
            const auto broad = _mm_set1_epi8(value);

            const auto mask = _mm_loadu_si128((__m128i const*)(maskArray + 16 - (position & 0x0F))); // FF at index position
            vector = _mm_or_si128(_mm_and_si128(mask, broad), _mm_andnot_si128(mask, vector));
        }
        else if constexpr (is_pd_v<_DesiredType_>) {
            const auto broadcasted = _mm_set_sd(value);

            vector = (position == 0)
                ? _mm_shuffle_pd(broadcasted, vector, 2)
                : _mm_shuffle_pd(vector, broadcasted, 0);
        }
        else if constexpr (is_ps_v<_DesiredType_>) {
            const int32 maskArray[8] = { 0,0,0,0,-1,0,0,0 };

            const auto broadcasted = _mm_set1_ps(value);
            const auto mask = _mm_loadu_ps((float const*)(maskArray + 4 - (position & 3))); // FFFFFFFF at index position

            vector = _mm_or_ps(
                _mm_and_ps(mask, broadcasted),
                _mm_andnot_ps(mask, vector));
        }
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _DesiredType_ extract(
        _VectorType_    vector,
        const uint8     where) noexcept
    {
        if constexpr (is_epi16_v<_DesiredType_> || is_epu16_v<_DesiredType_>) {
            // _mm_extract_epi16 ������� ����� ������� ����������
            switch (where) {
                case 0:
                    return static_cast<_DesiredType_>(_mm_extract_epi16(vector, 0));
                case 1:
                    return static_cast<_DesiredType_>(_mm_extract_epi16(vector, 1));
                case 2:
                    return static_cast<_DesiredType_>(_mm_extract_epi16(vector, 2));
                case 3:
                    return static_cast<_DesiredType_>(_mm_extract_epi16(vector, 3));
                case 4:
                    return static_cast<_DesiredType_>(_mm_extract_epi16(vector, 4));
                case 5:
                    return static_cast<_DesiredType_>(_mm_extract_epi16(vector, 5));
                case 6:
                    return static_cast<_DesiredType_>(_mm_extract_epi16(vector, 6));
                case 7:
                    return static_cast<_DesiredType_>(_mm_extract_epi16(vector, 7));
            }
        }
        else {
            _DesiredType_ array[sizeof(_VectorType_) / sizeof(_DesiredType_)];
            _MemoryAccess_::template storeUnaligned<_DesiredType_>(array, vector);

            return static_cast<_DesiredType_>(array[where]);
        }
    }

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ constructZero() noexcept {
        if      constexpr (std::is_same_v<_VectorType_, __m128d>)
            return _mm_setzero_pd();
        else if constexpr (std::is_same_v<_VectorType_, __m128i>)
            return _mm_setzero_si128();
        else if constexpr (std::is_same_v<_VectorType_, __m128>)
            return _mm_setzero_ps();
    }

    template <
        typename _VectorType_,
        typename _DesiredType_>
    static simd_stl_always_inline _VectorType_ broadcast(_DesiredType_ value) noexcept {
        if constexpr (is_epi64_v<_DesiredType_> || is_epu64_v<_DesiredType_>)
            return _Cast_::template cast<__m128i, _VectorType_>(_mm_set1_epi64x(pointerAddress(value)));
        else if constexpr (is_epi32_v<_DesiredType_> || is_epu32_v<_DesiredType_>)
            return _Cast_::template cast<__m128i, _VectorType_>(_mm_set1_epi32(pointerAddress(value)));
        else if constexpr (is_epi16_v<_DesiredType_> || is_epu16_v<_DesiredType_>)
            return _Cast_::template cast<__m128i, _VectorType_>(_mm_set1_epi16(value));
        else if constexpr (is_epi8_v<_DesiredType_> || is_epu8_v<_DesiredType_>)
            return _Cast_::template cast<__m128i, _VectorType_>(_mm_set1_epi8(value));
        else if constexpr (is_ps_v<_DesiredType_>)
            return _Cast_::template cast<__m128, _VectorType_>(_mm_set1_ps(value));
        else if constexpr (is_pd_v<_DesiredType_>)
            return _Cast_::template cast<__m128d, _VectorType_>(_mm_set1_pd(value));
    }
};

template <>
class SimdElementAccess<arch::CpuFeature::SSE3>:
    public SimdElementAccess<arch::CpuFeature::SSE2>
{};

template <>
class SimdElementAccess<arch::CpuFeature::SSSE3> :
    public SimdElementAccess<arch::CpuFeature::SSE3>
{};

template <>
class SimdElementAccess<arch::CpuFeature::SSE41> :
    public SimdElementAccess<arch::CpuFeature::SSSE3>
{};

template <>
class SimdElementAccess<arch::CpuFeature::SSE42>:
    public SimdElementAccess<arch::CpuFeature::SSE41>
{};

__SIMD_STL_NUMERIC_NAMESPACE_END
