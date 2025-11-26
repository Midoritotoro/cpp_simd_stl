#pragma once 

#include <src/simd_stl/numeric/IntrinBitcast.h>
#include <src/simd_stl/numeric/SimdConvert.h>

#include <src/simd_stl/type_traits/OperatorWrappers.h>


__SIMD_STL_NUMERIC_NAMESPACE_BEGIN

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_>
class _SimdCompareImplementation;

template <>
class _SimdCompareImplementation<arch::CpuFeature::SSE2, xmm128> {
    static constexpr auto _Generation   = arch::CpuFeature::SSE2;
    using _RegisterPolicy               = xmm128;

    template <class _DesiredType_>
    using _Simd_mask_type = type_traits::__deduce_simd_mask_type<_Generation, _DesiredType_, _RegisterPolicy>;
public:
    template <
        typename _DesiredType_,
        class    _CompareType_,
        typename _VectorType_>
    static simd_stl_always_inline _Simd_mask_type<_DesiredType_> _CompareMask(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept
    {
        if constexpr (std::is_same_v<_CompareType_, type_traits::equal_to<>>)
            return _CompareEqual<_DesiredType_>(_Left, _Right);

        else if constexpr (std::is_same_v<_CompareType_, type_traits::not_equal_to<>>)
            return ~_CompareEqual<_DesiredType_>(_Left, _Right);

        else if constexpr (std::is_same_v<_CompareType_, type_traits::less<>>)
            return _CompareLess<_DesiredType_>(_Left, _Right);

        else if constexpr (std::is_same_v<_CompareType_, type_traits::less_equal<>>)
            return ~_CompareGreater<_DesiredType_>(_Left, _Right);

        else if constexpr (std::is_same_v<_CompareType_, type_traits::greater<>>)
            return _CompareGreater<_DesiredType_>(_Right, _Left);

        else if constexpr (std::is_same_v<_CompareType_, type_traits::greater_equal<>>)
            return ~_CompareLess<_DesiredType_>(_Right, _Left);
    }

    template <
        typename _DesiredType_,
        class    _CompareType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _Compare(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept
    {
        if constexpr (std::is_same_v<_CompareType_, type_traits::equal_to<>>)
            return _CompareEqual<_DesiredType_>(_Left, _Right);

        else if constexpr (std::is_same_v<_CompareType_, type_traits::not_equal_to<>>)
            return ~_CompareEqual<_DesiredType_>(_Left, _Right);

        else if constexpr (std::is_same_v<_CompareType_, type_traits::less<>>)
            return _CompareLess<_DesiredType_>(_Left, _Right);

        else if constexpr (std::is_same_v<_CompareType_, type_traits::less_equal<>>)
            return ~_CompareGreater<_DesiredType_>(_Left, _Right);

        else if constexpr (std::is_same_v<_CompareType_, type_traits::greater<>>)
            return _CompareGreater<_DesiredType_>(_Right, _Left);

        else if constexpr (std::is_same_v<_CompareType_, type_traits::greater_equal<>>)
            return ~_CompareLess<_DesiredType_>(_Right, _Left);
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _CompareEqual(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept
    {
        if constexpr (_Is_epi64_v<_DesiredType_> || _Is_epu64_v<_DesiredType_>) {
            const auto _EqualMask = _mm_cmpeq_epi32(_IntrinBitcast<__m128i>(_Left), _IntrinBitcast<__m128i>(_Right));

            // int64 temp = equalMask[0];
            // equalMask[0] = equalMask[1];
            // equalMask[1] = temp;
            const auto _RotatedMask = _mm_shuffle_epi32(_EqualMask, 0xB1);
            const auto _CombinedMask = _mm_and_si128(_EqualMask, _RotatedMask);

            const auto _SignMask = _mm_srai_epi32(_CombinedMask, 31);
            return _IntrinBitcast<_VectorType_>(_mm_shuffle_epi32(_SignMask, 0xF5));
        }

        else if constexpr (_Is_epi32_v<_DesiredType_> || _Is_epu32_v<_DesiredType_>)
            return _IntrinBitcast<_VectorType_>(_mm_cmpeq_epi32(
                _IntrinBitcast<__m128i>(_Left), _IntrinBitcast<__m128i>(_Right)));

        else if constexpr (_Is_epi16_v<_DesiredType_> || _Is_epu16_v<_DesiredType_>)
            return _IntrinBitcast<_VectorType_>(_mm_cmpeq_epi16(
                _IntrinBitcast<__m128i>(_Left), _IntrinBitcast<__m128i>(_Right)));

        else if constexpr (_Is_epi8_v<_DesiredType_> || _Is_epu8_v<_DesiredType_>)
            return _IntrinBitcast<_VectorType_>(_mm_cmpeq_epi8(
                _IntrinBitcast<__m128i>(_Left), _IntrinBitcast<__m128i>(_Right)));

        else if constexpr (_Is_ps_v<_DesiredType_> || _Is_ps_v<_DesiredType_>)
            return _IntrinBitcast<_VectorType_>(_mm_cmpeq_ps(
                _IntrinBitcast<__m128>(_Left), _IntrinBitcast<__m128>(_Right)));

        else if constexpr (_Is_pd_v<_DesiredType_> || _Is_pd_v<_DesiredType_>)
            return _IntrinBitcast<_VectorType_>(_mm_cmpeq_pd(
                _IntrinBitcast<__m128d>(_Left), _IntrinBitcast<__m128d>(_Right)));
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _CompareLess(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept
    {
        if constexpr (_Is_epi64_v<_DesiredType_> || _Is_epu64_v<_DesiredType_>) {
            const auto _LeftToInteger   = _IntrinBitcast<__m128i>(_Left);
            const auto _RightToInteger  = _IntrinBitcast<__m128i>(_Right);

            const auto _Difference64    = _mm_sub_epi64(_LeftToInteger, _RightToInteger);

            const auto _XorMask          = _mm_xor_si128(_LeftToInteger, _RightToInteger);      // left ^ right
            const auto _LeftAndNotRight  = _mm_andnot_si128(_RightToInteger, _LeftToInteger);   // left & ~right
            const auto _DifferenceAndNotXor = _mm_andnot_si128(_XorMask, _Difference64);        // diff & ~(left ^ right)

            const auto _CombinedMask     = _mm_or_si128(_LeftAndNotRight, _DifferenceAndNotXor);

            const auto _SignBits32       = _mm_srai_epi32(_CombinedMask, 31);
            const auto _SignBits64       = _mm_shuffle_epi32(_SignBits32, 0xF5);

            return _IntrinBitcast<_VectorType_>(_SignBits64);
        }

        else if constexpr (_Is_epi32_v<_DesiredType_> || _Is_epu32_v<_DesiredType_>)
            return _IntrinBitcast<_VectorType_>(_mm_cmplt_epi32(
                _IntrinBitcast<__m128i>(_Left), _IntrinBitcast<__m128i>(_Right)));

        else if constexpr (_Is_epi16_v<_DesiredType_> || _Is_epu16_v<_DesiredType_>)
            return _IntrinBitcast<_VectorType_>(_mm_cmplt_epi16(
                _IntrinBitcast<__m128i>(_Left), _IntrinBitcast<__m128i>(_Right)));

        else if constexpr (_Is_epi8_v<_DesiredType_> || _Is_epu8_v<_DesiredType_>)
            return _IntrinBitcast<_VectorType_>(_mm_cmplt_epi8(
                _IntrinBitcast<__m128i>(_Left), _IntrinBitcast<__m128i>(_Right)));

        else if constexpr (_Is_ps_v<_DesiredType_>)
            return _IntrinBitcast<_VectorType_>(_mm_cmplt_ps(
                _IntrinBitcast<__m128>(_Left), _IntrinBitcast<__m128>(_Right)));

        else if constexpr (_Is_pd_v<_DesiredType_>)
            return _IntrinBitcast<_VectorType_>(_mm_cmplt_pd(
                _IntrinBitcast<__m128d>(_Left), _IntrinBitcast<__m128d>(_Right)));
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _CompareGreater(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept
    {
        if constexpr (_Is_epi64_v<_DesiredType_> || _Is_epu64_v<_DesiredType_>) {
            const auto _LeftToInteger   = _IntrinBitcast<__m128i>(_Left);
            const auto _RightToInteger  = _IntrinBitcast<__m128i>(_Right);

            const auto _SignBitMask = _mm_set1_epi32(0x80000000);
            const auto _LeftUnsigned = _mm_xor_si128(_LeftToInteger, _SignBitMask);
            const auto _RightUnsigned = _mm_xor_si128(_RightToInteger, _SignBitMask);

            const auto _EqualityMask = _mm_cmpeq_epi32(_LeftToInteger, _RightToInteger);
            const auto _GreaterMask = _mm_cmpgt_epi32(_LeftUnsigned, _RightUnsigned);

            const auto _GreaterHiMask = _mm_shuffle_epi32(_GreaterMask, 0xA0);
            const auto _EqualAndGreater = _mm_and_si128(_EqualityMask, _GreaterHiMask);

            const auto _CombinedMask = _mm_or_si128(_GreaterMask, _EqualAndGreater);

            return _IntrinBitcast<_VectorType_>(_mm_shuffle_epi32(_CombinedMask, 0xF5));
        }
        else if constexpr (_Is_epi32_v<_DesiredType_> || _Is_epu32_v<_DesiredType_>)
            return _IntrinBitcast<_VectorType_>(_mm_cmpgt_epi32(
                _IntrinBitcast<__m128i>(_Left), _IntrinBitcast<__m128i>(_Right)));

        else if constexpr (_Is_epi16_v<_DesiredType_> || _Is_epu16_v<_DesiredType_>)
            return _IntrinBitcast<_VectorType_>(_mm_cmpgt_epi16(
                _IntrinBitcast<__m128i>(_Left), _IntrinBitcast<__m128i>(_Right)));

        else if constexpr (_Is_epi8_v<_DesiredType_> || _Is_epu8_v<_DesiredType_>)
            return _IntrinBitcast<_VectorType_>(_mm_cmpgt_epi8(
                _IntrinBitcast<__m128i>(_Left), _IntrinBitcast<__m128i>(_Right)));

        else if constexpr (_Is_ps_v<_DesiredType_>)
            return _IntrinBitcast<_VectorType_>(_mm_cmpgt_ps(
                _IntrinBitcast<__m128>(_Left), _IntrinBitcast<__m128>(_Right)));

        else if constexpr (_Is_pd_v<_DesiredType_>)
            return _IntrinBitcast<_VectorType_>(_mm_cmpgt_pd(
                _IntrinBitcast<__m128d>(_Left), _IntrinBitcast<__m128d>(_Right)));
    }
};

template <>
class _SimdCompareImplementation<arch::CpuFeature::SSE3, xmm128> :
    public _SimdCompareImplementation<arch::CpuFeature::SSE2, xmm128>
{};

template <>
class _SimdCompareImplementation<arch::CpuFeature::SSSE3, xmm128> :
    public _SimdCompareImplementation<arch::CpuFeature::SSE3, xmm128>
{};

template <>
class _SimdCompareImplementation<arch::CpuFeature::SSE41, xmm128> :
    public _SimdCompareImplementation<arch::CpuFeature::SSSE3, xmm128>
{};

template <>
class _SimdCompareImplementation<arch::CpuFeature::SSE42, xmm128> :
    public _SimdCompareImplementation<arch::CpuFeature::SSE41, xmm128>
{};

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_,
    class               _DesiredType_,
    class               _CompareType_,
    class               _VectorType_>
simd_stl_nodiscard simd_stl_always_inline _VectorType_ _SimdCompare(
    _VectorType_ _Left,
    _VectorType_ _Right) noexcept
{
    _VerifyRegisterPolicy(_SimdGeneration_, _RegisterPolicy_);
    return _SimdCompareImplementation<_SimdGeneration_, _RegisterPolicy_>
        ::template _Compare<_DesiredType_, _CompareType_>(_Left, _Right);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_,
    class               _DesiredType_,
    class               _CompareType_,
    class               _VectorType_>
simd_stl_nodiscard simd_stl_always_inline type_traits::__deduce_simd_mask_type<_SimdGeneration_,
    _DesiredType_, _RegisterPolicy_> _SimdMaskCompare(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept
{
    _VerifyRegisterPolicy(_SimdGeneration_, _RegisterPolicy_);
    return _SimdCompareImplementation<_SimdGeneration_, _RegisterPolicy_>
        ::template _MaskCompare<_DesiredType_, _CompareType_>(_Left, _Right);
}

__SIMD_STL_NUMERIC_NAMESPACE_END
