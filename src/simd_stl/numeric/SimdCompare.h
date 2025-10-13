#pragma once 

#include <src/simd_stl/numeric/SimdCast.h>
#include <src/simd_stl/numeric/SimdConvert.h>

#include <src/simd_stl/type_traits/OperatorWrappers.h>


__SIMD_STL_NUMERIC_NAMESPACE_BEGIN

template <arch::CpuFeature _SimdGeneration_>
class SimdCompare;

template <>
class SimdCompare<arch::CpuFeature::SSE2> {
    using _Cast_        = SimdCast<arch::CpuFeature::SSE2>;
    using _Convert_     = SimdConvert<arch::CpuFeature::SSE2>;
public:
    template <
        typename _DesiredType_,
        class    _CompareType_,
        typename _VectorType_>
    static simd_stl_always_inline int compare(
        _VectorType_ left,
        _VectorType_ right) noexcept
    {
        if constexpr (std::is_same_v<_CompareType_, type_traits::equal_to<>>)
            return compareEqual<_DesiredType_>(left, right);

        else if constexpr (std::is_same_v<_CompareType_, type_traits::not_equal_to<>>)
            return (~compareEqual<_DesiredType_>(left, right));

        else if constexpr (std::is_same_v<_CompareType_, type_traits::less<>>)
            return compareLess<_DesiredType_>(left, right);

        else if constexpr (std::is_same_v<_CompareType_, type_traits::less_equal<>>)
            return ~compareGreater<_DesiredType_>(left, right);

        else if constexpr (std::is_same_v<_CompareType_, type_traits::greater<>>)
            return compareGreater<_DesiredType_>(right, left);

        else if constexpr (std::is_same_v<_CompareType_, type_traits::greater_equal<>>)
            return ~compareLess<_DesiredType_>(right, left);
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline int compareEqual(
        _VectorType_ left,
        _VectorType_ right) noexcept
    {
        if constexpr (is_epi64_v<_DesiredType_> || is_epu64_v<_DesiredType_>) {
            const auto equalMask = _mm_cmpeq_epi32(
                _Cast_::template cast<_VectorType_, __m128i>(left),
                _Cast_::template cast<_VectorType_, __m128i>(right));

            // Меняем местами младшие и старшие 64 бита. 
            // int64 temp = equalMask[0];
            // equalMask[0] = equalMask[1];
            // equalMask[1] = temp;

            const auto rotatedMask = _mm_shuffle_epi32(equalMask, 0xB1);

            const auto combinedMask = _mm_and_si128(equalMask, rotatedMask);
            const auto signMask = _mm_srai_epi32(combinedMask, 31);

            const auto finalMask = _mm_shuffle_epi32(signMask, 0xF5);
            return _Convert_::template convertToMask<_DesiredType_>(finalMask);
        }

        else if constexpr (is_epi32_v<_DesiredType_> || is_epu32_v<_DesiredType_>)
            return _Convert_::template convertToMask<_DesiredType_>(_mm_cmpeq_epi32(
                _Cast_::template cast<_VectorType_, __m128i>(left),
                _Cast_::template cast<_VectorType_, __m128i>(right)));

        else if constexpr (is_epi16_v<_DesiredType_> || is_epu16_v<_DesiredType_>)
            return _Convert_::template convertToMask<_DesiredType_>(_mm_cmpeq_epi16(
                _Cast_::template cast<_VectorType_, __m128i>(left),
                _Cast_::template cast<_VectorType_, __m128i>(right)));

        else if constexpr (is_epi8_v<_DesiredType_> || is_epu8_v<_DesiredType_>)
            return _Convert_::template convertToMask<_DesiredType_>(_mm_cmpeq_epi8(
                _Cast_::template cast<_VectorType_, __m128i>(left),
                _Cast_::template cast<_VectorType_, __m128i>(right)));

        else if constexpr (is_ps_v<_DesiredType_> || is_ps_v<_DesiredType_>)
            return _Convert_::template convertToMask<_DesiredType_>(_mm_cmpeq_ps(
                _Cast_::template cast<_VectorType_, __m128>(left),
                _Cast_::template cast<_VectorType_, __m128>(right)));

        else if constexpr (is_pd_v<_DesiredType_> || is_pd_v<_DesiredType_>)
            return _Convert_::template convertToMask<_DesiredType_>(_mm_cmpeq_pd(
                _Cast_::template cast<_VectorType_, __m128d>(left),
                _Cast_::template cast<_VectorType_, __m128d>(right)));
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline int compareLess(
        _VectorType_ left,
        _VectorType_ right) noexcept
    {
        if constexpr (is_epi64_v<_DesiredType_> || is_epu64_v<_DesiredType_>) {
            const auto leftToInteger    = _Cast_::template cast<_VectorType_, __m128i>(left);
            const auto rightToInteger   = _Cast_::template cast<_VectorType_, __m128i>(right);

            const auto diff64           = _mm_sub_epi64(leftToInteger, rightToInteger);

            const auto xorMask          = _mm_xor_si128(leftToInteger, rightToInteger);     // left ^ right
            const auto leftAndNotRight  = _mm_andnot_si128(rightToInteger, leftToInteger);  // left & ~right
            const auto diffAndNotXor    = _mm_andnot_si128(xorMask, diff64);                // diff & ~(left ^ right)

            const auto combinedMask     = _mm_or_si128(leftAndNotRight, diffAndNotXor);

            const auto signBits32       = _mm_srai_epi32(combinedMask, 31);
            const auto signBits64       = _mm_shuffle_epi32(signBits32, 0xF5);

            return _Convert_::template convertToMask<_DesiredType_>(signBits64);
        }
        else if constexpr (is_epi32_v<_DesiredType_> || is_epu32_v<_DesiredType_>)
            return _Convert_::template convertToMask<_DesiredType_>(_mm_cmplt_epi32(
                _Cast_::template cast<_VectorType_, __m128i>(left),
                _Cast_::template cast<_VectorType_, __m128i>(right)));

        else if constexpr (is_epi16_v<_DesiredType_> || is_epu16_v<_DesiredType_>)
            return _Convert_::template convertToMask<_DesiredType_>(_mm_cmplt_epi16(
                _Cast_::template cast<_VectorType_, __m128i>(left),
                _Cast_::template cast<_VectorType_, __m128i>(right)));

        else if constexpr (is_epi8_v<_DesiredType_> || is_epu8_v<_DesiredType_>)
            return _Convert_::template convertToMask<_DesiredType_>(_mm_cmplt_epi8(
                _Cast_::template cast<_VectorType_, __m128i>(left),
                _Cast_::template cast<_VectorType_, __m128i>(right)));

        else if constexpr (is_ps_v<_DesiredType_> || is_ps_v<_DesiredType_>)
            return _Convert_::template convertToMask<_DesiredType_>(_mm_cmplt_ps(
                _Cast_::template cast<_VectorType_, __m128>(left),
                _Cast_::template cast<_VectorType_, __m128>(right)));

        else if constexpr (is_pd_v<_DesiredType_> || is_pd_v<_DesiredType_>)
            return _Convert_::template convertToMask<_DesiredType_>(_mm_cmplt_pd(
                _Cast_::template cast<_VectorType_, __m128d>(left),
                _Cast_::template cast<_VectorType_, __m128d>(right)));
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline int compareGreater(
        _VectorType_ left,
        _VectorType_ right) noexcept
    {
        if constexpr (is_epi64_v<_DesiredType_> || is_epu64_v<_DesiredType_>) {
            const auto leftToInteger = _Cast_::template cast<_VectorType_, __m128i>(left);
            const auto rightToInteger = _Cast_::template cast<_VectorType_, __m128i>(right);

            const auto signBitMask = _mm_set1_epi32(0x80000000);
            const auto leftUnsigned = _mm_xor_si128(leftToInteger, signBitMask);
            const auto rightUnsigned = _mm_xor_si128(rightToInteger, signBitMask);

            const auto equalityMask = _mm_cmpeq_epi32(leftToInteger, rightToInteger);
            const auto greaterMask = _mm_cmpgt_epi32(leftUnsigned, rightUnsigned);

            const auto greaterHiMask = _mm_shuffle_epi32(greaterMask, 0xA0);
            const auto equalAndGreater = _mm_and_si128(equalityMask, greaterHiMask);

            const auto combinedMask = _mm_or_si128(greaterMask, equalAndGreater);
            const auto finalMask = _mm_shuffle_epi32(combinedMask, 0xF5);

            return _Convert_::template convertToMask<_DesiredType_>(finalMask);
        }
        else if constexpr (is_epi32_v<_DesiredType_> || is_epu32_v<_DesiredType_>)
            return _Convert_::template convertToMask<_DesiredType_>(_mm_cmpgt_epi32(
                _Cast_::template cast<_VectorType_, __m128i>(left),
                _Cast_::template cast<_VectorType_, __m128i>(right)));

        else if constexpr (is_epi16_v<_DesiredType_> || is_epu16_v<_DesiredType_>)
            return _Convert_::template convertToMask<_DesiredType_>(_mm_cmpgt_epi16(
                _Cast_::template cast<_VectorType_, __m128i>(left),
                _Cast_::template cast<_VectorType_, __m128i>(right)));

        else if constexpr (is_epi8_v<_DesiredType_> || is_epu8_v<_DesiredType_>)
            return _Convert_::template convertToMask<_DesiredType_>(_mm_cmpgt_epi8(
                _Cast_::template cast<_VectorType_, __m128i>(left),
                _Cast_::template cast<_VectorType_, __m128i>(right)));

        else if constexpr (is_ps_v<_DesiredType_> || is_ps_v<_DesiredType_>)
            return _Convert_::template convertToMask<_DesiredType_>(_mm_cmpgt_ps(
                _Cast_::template cast<_VectorType_, __m128>(left),
                _Cast_::template cast<_VectorType_, __m128>(right)));

        else if constexpr (is_pd_v<_DesiredType_> || is_pd_v<_DesiredType_>)
            return _Convert_::template convertToMask<_DesiredType_>(_mm_cmpgt_pd(
                _Cast_::template cast<_VectorType_, __m128d>(left),
                _Cast_::template cast<_VectorType_, __m128d>(right)));
    }
};

template <>
class SimdCompare<arch::CpuFeature::SSE3> :
    public SimdCompare<arch::CpuFeature::SSE2>
{};

template <>
class SimdCompare<arch::CpuFeature::SSSE3> :
    public SimdCompare<arch::CpuFeature::SSE3>
{};

template <>
class SimdCompare<arch::CpuFeature::SSE41> :
    public SimdCompare<arch::CpuFeature::SSSE3>
{};

template <>
class SimdCompare<arch::CpuFeature::SSE42> :
    public SimdCompare<arch::CpuFeature::SSE41>
{};

__SIMD_STL_NUMERIC_NAMESPACE_END