#pragma once 

#include <src/simd_stl/numeric/IntrinBitcast.h>
#include <src/simd_stl/numeric/SimdConvert.h>

#include <src/simd_stl/type_traits/OperatorWrappers.h>


__SIMD_STL_NUMERIC_NAMESPACE_BEGIN

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_>
class _SimdCompareImplementation;

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_,
    class               _DesiredType_,
    class               _CompareType_,
    class               _VectorType_>
simd_stl_nodiscard simd_stl_always_inline _VectorType_ _SimdCompare(
    _VectorType_ _Left,
    _VectorType_ _Right) noexcept;

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_,
    class               _DesiredType_,
    class               _CompareType_,
    class               _VectorType_>
simd_stl_nodiscard simd_stl_always_inline type_traits::__deduce_simd_mask_type<_SimdGeneration_,
    _DesiredType_, _RegisterPolicy_> _SimdMaskCompare(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept;

#pragma region Sse2-Sse4.2 Simd compare

template <>
class _SimdCompareImplementation<arch::CpuFeature::SSE2, xmm128> {
    static constexpr auto _Generation   = arch::CpuFeature::SSE2;
    using _RegisterPolicy               = xmm128;

    template <class _DesiredType_>
    using _Simd_mask_type = type_traits::__deduce_simd_mask_type<_Generation, _DesiredType_, _RegisterPolicy>;
public:
    template <
        class   _DesiredType_,
        class   _CompareType_,
        class   _VectorType_>
    static simd_stl_always_inline _Simd_mask_type<_DesiredType_> _MaskCompare(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept 
    {
        return _SimdToMask<_Generation, _RegisterPolicy, _DesiredType_>(_Compare<_DesiredType_, _CompareType_>(_Left, _Right));
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

            const auto _RotatedMask = _mm_shuffle_epi32(_EqualMask, 0xB1);
            const auto _CombinedMask = _mm_and_si128(_EqualMask, _RotatedMask);

            const auto _SignMask = _mm_srai_epi32(_CombinedMask, 31);
            return _IntrinBitcast<_VectorType_>(_mm_shuffle_epi32(_SignMask, 0xF5));
        }
        else if constexpr (_Is_epi32_v<_DesiredType_> || _Is_epu32_v<_DesiredType_>) {
            return _IntrinBitcast<_VectorType_>(_mm_cmpeq_epi32(
                _IntrinBitcast<__m128i>(_Left), _IntrinBitcast<__m128i>(_Right)));
        }
        else if constexpr (_Is_epi16_v<_DesiredType_> || _Is_epu16_v<_DesiredType_>) {
            return _IntrinBitcast<_VectorType_>(_mm_cmpeq_epi16(
                _IntrinBitcast<__m128i>(_Left), _IntrinBitcast<__m128i>(_Right)));
        }
        else if constexpr (_Is_epi8_v<_DesiredType_> || _Is_epu8_v<_DesiredType_>) {
            return _IntrinBitcast<_VectorType_>(_mm_cmpeq_epi8(
                _IntrinBitcast<__m128i>(_Left), _IntrinBitcast<__m128i>(_Right)));
        }
        else if constexpr (_Is_ps_v<_DesiredType_> || _Is_ps_v<_DesiredType_>) {
            return _IntrinBitcast<_VectorType_>(_mm_cmpeq_ps(
                _IntrinBitcast<__m128>(_Left), _IntrinBitcast<__m128>(_Right)));
        }
        else if constexpr (_Is_pd_v<_DesiredType_> || _Is_pd_v<_DesiredType_>) {
            return _IntrinBitcast<_VectorType_>(_mm_cmpeq_pd(
                _IntrinBitcast<__m128d>(_Left), _IntrinBitcast<__m128d>(_Right)));
        }
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
        else if constexpr (_Is_epi32_v<_DesiredType_> || _Is_epu32_v<_DesiredType_>) {
            return _IntrinBitcast<_VectorType_>(_mm_cmplt_epi32(
                _IntrinBitcast<__m128i>(_Left), _IntrinBitcast<__m128i>(_Right)));
        }
        else if constexpr (_Is_epi16_v<_DesiredType_> || _Is_epu16_v<_DesiredType_>) {
            return _IntrinBitcast<_VectorType_>(_mm_cmplt_epi16(
                _IntrinBitcast<__m128i>(_Left), _IntrinBitcast<__m128i>(_Right)));
        }
        else if constexpr (_Is_epi8_v<_DesiredType_> || _Is_epu8_v<_DesiredType_>) {
            return _IntrinBitcast<_VectorType_>(_mm_cmplt_epi8(
                _IntrinBitcast<__m128i>(_Left), _IntrinBitcast<__m128i>(_Right)));
        }
        else if constexpr (_Is_ps_v<_DesiredType_>) {
            return _IntrinBitcast<_VectorType_>(_mm_cmplt_ps(
                _IntrinBitcast<__m128>(_Left), _IntrinBitcast<__m128>(_Right)));
        }
        else if constexpr (_Is_pd_v<_DesiredType_>) {
            return _IntrinBitcast<_VectorType_>(_mm_cmplt_pd(
                _IntrinBitcast<__m128d>(_Left), _IntrinBitcast<__m128d>(_Right)));
        }
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
        else if constexpr (_Is_epi32_v<_DesiredType_> || _Is_epu32_v<_DesiredType_>) {
            return _IntrinBitcast<_VectorType_>(_mm_cmpgt_epi32(
                _IntrinBitcast<__m128i>(_Left), _IntrinBitcast<__m128i>(_Right)));
        }
        else if constexpr (_Is_epi16_v<_DesiredType_> || _Is_epu16_v<_DesiredType_>) {
            return _IntrinBitcast<_VectorType_>(_mm_cmpgt_epi16(
                _IntrinBitcast<__m128i>(_Left), _IntrinBitcast<__m128i>(_Right)));
        }
        else if constexpr (_Is_epi8_v<_DesiredType_> || _Is_epu8_v<_DesiredType_>) {
            return _IntrinBitcast<_VectorType_>(_mm_cmpgt_epi8(
                _IntrinBitcast<__m128i>(_Left), _IntrinBitcast<__m128i>(_Right)));
        }
        else if constexpr (_Is_ps_v<_DesiredType_>) {
            return _IntrinBitcast<_VectorType_>(_mm_cmpgt_ps(
                _IntrinBitcast<__m128>(_Left), _IntrinBitcast<__m128>(_Right)));
        }
        else if constexpr (_Is_pd_v<_DesiredType_>) {
            return _IntrinBitcast<_VectorType_>(_mm_cmpgt_pd(
                _IntrinBitcast<__m128d>(_Left), _IntrinBitcast<__m128d>(_Right)));
        }
    }
};

template <>
class _SimdCompareImplementation<arch::CpuFeature::SSE3, xmm128> :
    public _SimdCompareImplementation<arch::CpuFeature::SSE2, xmm128>
{};

template <>
class _SimdCompareImplementation<arch::CpuFeature::SSSE3, xmm128> :
    public _SimdCompareImplementation<arch::CpuFeature::SSE3, xmm128>
{
    static constexpr auto _Generation   = arch::CpuFeature::SSSE3;
    using _RegisterPolicy               = xmm128;

    template <class _DesiredType_>
    using _Simd_mask_type = type_traits::__deduce_simd_mask_type<_Generation, _DesiredType_, _RegisterPolicy>;
public:
    template <
        class   _DesiredType_,
        class   _CompareType_,
        class   _VectorType_>
    static simd_stl_always_inline _Simd_mask_type<_DesiredType_> _MaskCompare(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept 
    {
        return _SimdToMask<_Generation, _RegisterPolicy, _DesiredType_>(_Compare<_DesiredType_, _CompareType_>(_Left, _Right));
    }
};

template <>
class _SimdCompareImplementation<arch::CpuFeature::SSE41, xmm128> :
    public _SimdCompareImplementation<arch::CpuFeature::SSSE3, xmm128>
{
    static constexpr auto _Generation   = arch::CpuFeature::SSE41;
    using _RegisterPolicy               = xmm128;

    template <class _DesiredType_>
    using _Simd_mask_type = type_traits::__deduce_simd_mask_type<_Generation, _DesiredType_, _RegisterPolicy>;
public:
    template <
        class   _DesiredType_,
        class   _CompareType_,
        class   _VectorType_>
    static simd_stl_always_inline _Simd_mask_type<_DesiredType_> _MaskCompare(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept 
    {
        return _SimdToMask<_Generation, _RegisterPolicy, _DesiredType_>(_Compare<_DesiredType_, _CompareType_>(_Left, _Right));
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
        if constexpr (_Is_epi64_v<_DesiredType_> || _Is_epu64_v<_DesiredType_>)
            return _IntrinBitcast<_VectorType_>(
                _mm_cmpeq_epi64(_IntrinBitcast<__m128i>(_Left), _IntrinBitcast<__m128i>(_Right)));
        else
            return _SimdCompare<arch::CpuFeature::SSE2, _RegisterPolicy, type_traits::equal_to<>>(_Left, _Right);
        
    }
};

template <>
class _SimdCompareImplementation<arch::CpuFeature::SSE42, xmm128>:
    public _SimdCompareImplementation<arch::CpuFeature::SSE41, xmm128>
{
    static constexpr auto _Generation   = arch::CpuFeature::SSE42;
    using _RegisterPolicy               = xmm128;

    template <class _DesiredType_>
    using _Simd_mask_type = type_traits::__deduce_simd_mask_type<_Generation, _DesiredType_, _RegisterPolicy>;
public:
    template <
        class   _DesiredType_,
        class   _CompareType_,
        class   _VectorType_>
    static simd_stl_always_inline _Simd_mask_type<_DesiredType_> _MaskCompare(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept 
    {
        return _SimdToMask<_Generation, _RegisterPolicy, _DesiredType_>(_Compare<_DesiredType_, _CompareType_>(_Left, _Right));
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
    static simd_stl_always_inline _VectorType_ _CompareGreater(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept
    {
        if constexpr (_Is_epi64_v<_DesiredType_> || _Is_epu64_v<_DesiredType_>)
            return _IntrinBitcast<_VectorType_>(
                _mm_cmpgt_epi64(_IntrinBitcast<__m128i>(_Left), _IntrinBitcast<__m128i>(_Right)));
        else
            return _SimdCompare<arch::CpuFeature::SSE2, _RegisterPolicy, type_traits::greater<>>(_Left, _Right);
    }
};

#pragma endregion

#pragma region Avx Simd compare

template <>
class _SimdCompareImplementation<arch::CpuFeature::AVX, ymm256> {
    static constexpr auto _Generation   = arch::CpuFeature::AVX;
    using _RegisterPolicy               = ymm256;

    template <class _DesiredType_>
    using _Simd_mask_type = type_traits::__deduce_simd_mask_type<_Generation, _DesiredType_, _RegisterPolicy>;
public:
    template <
        class   _DesiredType_,
        class   _CompareType_,
        class   _VectorType_>
    static simd_stl_always_inline _Simd_mask_type<_DesiredType_> _MaskCompare(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept 
    {
        return _SimdToMask<_Generation, _RegisterPolicy, _DesiredType_>(_Compare<_DesiredType_, _CompareType_>(_Left, _Right));
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
        if constexpr (_Is_pd_v<_DesiredType_>) {
            return _IntrinBitcast<_VectorType_>(_mm256_cmp_pd(
                _IntrinBitcast<__m256d>(_Left), _IntrinBitcast<__m256d>(_Right), _CMP_EQ_OQ));
        }
        else if constexpr (_Is_ps_v<_DesiredType_>) {
            return _IntrinBitcast<_VectorType_>(_mm256_cmp_ps(
                _IntrinBitcast<__m256>(_Left), _IntrinBitcast<__m256>(_Right), _CMP_EQ_OQ));
        }
        else {
            const auto _Low = _SimdCompare<arch::CpuFeature::SSE42, xmm128, _DesiredType_, type_traits::equal_to<>>(
                _IntrinBitcast<__m128i>(_Left), _IntrinBitcast<__m128i>(_Right));
            
            const auto _High = _SimdCompare<arch::CpuFeature::SSE42, xmm128, _DesiredType_, type_traits::equal_to<>>(
                _mm256_extractf128_si256(_IntrinBitcast<__m256i>(_Left), 1),
                _mm256_extractf128_si256(_IntrinBitcast<__m256i>(_Right), 1));

            return _IntrinBitcast<_VectorType_>(_mm256_insertf128_si256(_IntrinBitcast<__m256i>(_Low), _High, 1));
        }
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _CompareLess(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept
    {
        if constexpr (_Is_pd_v<_DesiredType_>) {
            return _IntrinBitcast<_VectorType_>(_mm256_cmp_pd(
                _IntrinBitcast<__m256d>(_Left), _IntrinBitcast<__m256d>(_Right), _MM_CMPINT_LT));
        }
        else if constexpr (_Is_ps_v<_DesiredType_>) {
            return _IntrinBitcast<_VectorType_>(_mm256_cmp_ps(
                _IntrinBitcast<__m256>(_Left), _IntrinBitcast<__m256>(_Right), _MM_CMPINT_LT));
        }
        else {
            const auto _Low = _SimdCompare<arch::CpuFeature::SSE42, xmm128, _DesiredType_, type_traits::less<>>(
                _IntrinBitcast<__m128i>(_Left), _IntrinBitcast<__m128i>(_Right));
            
            const auto _High = _SimdCompare<arch::CpuFeature::SSE42, xmm128, _DesiredType_, type_traits::less<>>(
                _mm256_extractf128_si256(_IntrinBitcast<__m256i>(_Left), 1),
                _mm256_extractf128_si256(_IntrinBitcast<__m256i>(_Right), 1));

            return _IntrinBitcast<_VectorType_>(_mm256_insertf128_si256(_IntrinBitcast<__m256i>(_Low), _High, 1));
        }
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _CompareGreater(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept
    {
        if constexpr (_Is_pd_v<_DesiredType_>) {
            return _IntrinBitcast<_VectorType_>(_mm256_cmp_pd(
                _IntrinBitcast<__m256d>(_Left), _IntrinBitcast<__m256d>(_Right), _MM_CMPINT_GT));
        }
        else if constexpr (_Is_ps_v<_DesiredType_>) {
            return _IntrinBitcast<_VectorType_>(_mm256_cmp_ps(
                _IntrinBitcast<__m256>(_Left), _IntrinBitcast<__m256>(_Right), _MM_CMPINT_GT));
        }
        else {
            const auto _Low = _SimdCompare<arch::CpuFeature::SSE42, xmm128, _DesiredType_, type_traits::greater<>>(
                _IntrinBitcast<__m128i>(_Left), _IntrinBitcast<__m128i>(_Right));
            
            const auto _High = _SimdCompare<arch::CpuFeature::SSE42, xmm128, _DesiredType_, type_traits::greater<>>(
                _mm256_extractf128_si256(_IntrinBitcast<__m256i>(_Left), 1),
                _mm256_extractf128_si256(_IntrinBitcast<__m256i>(_Right), 1));

            return _IntrinBitcast<_VectorType_>(_mm256_insertf128_si256(_IntrinBitcast<__m256i>(_Low), _High, 1));
        }
    }
};

template <>
class _SimdCompareImplementation<arch::CpuFeature::AVX2, ymm256>:
    public _SimdCompareImplementation<arch::CpuFeature::AVX, ymm256>
{
    static constexpr auto _Generation   = arch::CpuFeature::AVX2;
    using _RegisterPolicy               = ymm256;

    template <class _DesiredType_>
    using _Simd_mask_type = type_traits::__deduce_simd_mask_type<_Generation, _DesiredType_, _RegisterPolicy>;
public:
    template <
        class   _DesiredType_,
        class   _CompareType_,
        class   _VectorType_>
    static simd_stl_always_inline _Simd_mask_type<_DesiredType_> _MaskCompare(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept
    {
        return _SimdToMask<_Generation, _RegisterPolicy, _DesiredType_>(_Compare<_DesiredType_, _CompareType_>(_Left, _Right));
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
        if constexpr (_Is_pd_v<_DesiredType_>)
            return _IntrinBitcast<_VectorType_>(_mm256_cmp_pd(
                _IntrinBitcast<__m256d>(_Left), _IntrinBitcast<__m256d>(_Right), _CMP_EQ_OQ));
        
        else if constexpr (_Is_ps_v<_DesiredType_>)
            return _IntrinBitcast<_VectorType_>(_mm256_cmp_ps(
                _IntrinBitcast<__m256>(_Left), _IntrinBitcast<__m256>(_Right), _CMP_EQ_OQ));

        else if constexpr (_Is_epi64_v<_DesiredType_> || _Is_epu64_v<_DesiredType_>)
            return _IntrinBitcast<_VectorType_>(_mm256_cmpeq_epi64(
                _IntrinBitcast<__m256i>(_Left), _IntrinBitcast<__m256i>(_Right)));

        else if constexpr (_Is_epi32_v<_DesiredType_> || _Is_epu32_v<_DesiredType_>)
            return _IntrinBitcast<_VectorType_>(_mm256_cmpeq_epi32(
                _IntrinBitcast<__m256i>(_Left), _IntrinBitcast<__m256i>(_Right)));

        else if constexpr (_Is_epi16_v<_DesiredType_> || _Is_epu16_v<_DesiredType_>)
            return _IntrinBitcast<_VectorType_>(_mm256_cmpeq_epi16(
                _IntrinBitcast<__m256i>(_Left), _IntrinBitcast<__m256i>(_Right)));

        else if constexpr (_Is_epi8_v<_DesiredType_> || _Is_epu8_v<_DesiredType_>)
            return _IntrinBitcast<_VectorType_>(_mm256_cmpeq_epi8(
                _IntrinBitcast<__m256i>(_Left), _IntrinBitcast<__m256i>(_Right)));
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _CompareLess(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept
    {
        return _CompareGreater<_DesiredType_>(_Right, _Left);
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _CompareGreater(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept
    {
        if constexpr (_Is_pd_v<_DesiredType_>)
            return _IntrinBitcast<_VectorType_>(_mm256_cmp_pd(
                _IntrinBitcast<__m256d>(_Left), _IntrinBitcast<__m256d>(_Right), _MM_CMPINT_GT));
        
        else if constexpr (_Is_ps_v<_DesiredType_>)
            return _IntrinBitcast<_VectorType_>(_mm256_cmp_ps(
                _IntrinBitcast<__m256>(_Left), _IntrinBitcast<__m256>(_Right), _MM_CMPINT_GT));
        
        else if constexpr (_Is_epi64_v<_DesiredType_> || _Is_epu64_v<_DesiredType_>)
            return _IntrinBitcast<_VectorType_>(_mm256_cmpgt_epi64(
                _IntrinBitcast<__m256i>(_Left), _IntrinBitcast<__m256i>(_Right)));

        else if constexpr (_Is_epi32_v<_DesiredType_> || _Is_epu32_v<_DesiredType_>)
            return _IntrinBitcast<_VectorType_>(_mm256_cmpgt_epi32(
                _IntrinBitcast<__m256i>(_Left), _IntrinBitcast<__m256i>(_Right)));

        else if constexpr (_Is_epi16_v<_DesiredType_> || _Is_epu16_v<_DesiredType_>)
            return _IntrinBitcast<_VectorType_>(_mm256_cmpgt_epi16(
                _IntrinBitcast<__m256i>(_Left), _IntrinBitcast<__m256i>(_Right)));

        else if constexpr (_Is_epi8_v<_DesiredType_> || _Is_epu8_v<_DesiredType_>)
            return _IntrinBitcast<_VectorType_>(_mm256_cmpgt_epi8(
                _IntrinBitcast<__m256i>(_Left), _IntrinBitcast<__m256i>(_Right)));
    }
};


#pragma endregion

#pragma region Avx512 Simd compare

template <>
class _SimdCompareImplementation<arch::CpuFeature::AVX512F, zmm512> {
    static constexpr auto _Generation = arch::CpuFeature::AVX512F;
    using _RegisterPolicy = zmm512;

    template <class _DesiredType_>
    using _Simd_mask_type = type_traits::__deduce_simd_mask_type<_Generation, _DesiredType_, _RegisterPolicy>;
public:
    template <
        class   _DesiredType_,
        class   _CompareType_,
        class   _VectorType_>
    static simd_stl_always_inline _Simd_mask_type<_DesiredType_> _MaskCompare(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept 
    {
        if constexpr (std::is_same_v<_CompareType_, type_traits::equal_to<>>)
            return _MaskCompareEqual<_DesiredType_>(_Left, _Right);

        else if constexpr (std::is_same_v<_CompareType_, type_traits::not_equal_to<>>)
            return ~_MaskCompareEqual<_DesiredType_>(_Left, _Right);

        else if constexpr (std::is_same_v<_CompareType_, type_traits::less<>>)
            return _MaskCompareLess<_DesiredType_>(_Left, _Right);

        else if constexpr (std::is_same_v<_CompareType_, type_traits::less_equal<>>)
            return ~_MaskCompareGreater<_DesiredType_>(_Left, _Right);

        else if constexpr (std::is_same_v<_CompareType_, type_traits::greater<>>)
            return _MaskCompareGreater<_DesiredType_>(_Right, _Left);

        else if constexpr (std::is_same_v<_CompareType_, type_traits::greater_equal<>>)
            return ~_MaskCompareLess<_DesiredType_>(_Right, _Left);
    }

    template <
        typename _DesiredType_,
        class    _CompareType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _Compare(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept
    {
        return _SimdToVector<_Generation, _RegisterPolicy, _VectorType_>(_MaskCompare<_DesiredType_, _CompareType_>(_Left, _Right));
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _Simd_mask_type<_DesiredType_> _MaskCompareEqual(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept
    {
        if constexpr (_Is_pd_v<_DesiredType_>) {
            return _mm512_cmpeq_pd_mask(_IntrinBitcast<__m512d>(_Left), _IntrinBitcast<__m512d>(_Right));
        }
        else if constexpr (_Is_ps_v<_DesiredType_>) {
            return _mm512_cmpeq_ps_mask(_IntrinBitcast<__m512>(_Left), _IntrinBitcast<__m512>(_Right));
        }
        else if constexpr (_Is_epi64_v<_DesiredType_> || _Is_epu64_v<_DesiredType_>) {
            return _mm512_cmpeq_epi64_mask(_IntrinBitcast<__m512i>(_Left), _IntrinBitcast<__m512i>(_Right));
        }
        else if constexpr (_Is_epi32_v<_DesiredType_> || _Is_epu32_v<_DesiredType_>) {
            return _mm512_cmpeq_epi32_mask(_IntrinBitcast<__m512i>(_Left), _IntrinBitcast<__m512i>(_Right));
        }
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _Simd_mask_type<_DesiredType_> _MaskCompareLess(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept
    {
        if constexpr (_Is_epi64_v<_DesiredType_>) {
            return _mm512_cmplt_epi64_mask(_IntrinBitcast<__m512i>(_Left), _IntrinBitcast<__m512i>(_Right));
        }
        else if constexpr (_Is_epu64_v<_DesiredType_>) {
            return _mm512_cmplt_epu64_mask(_IntrinBitcast<__m512i>(_Left), _IntrinBitcast<__m512i>(_Right));
        }
        else if constexpr (_Is_epi32_v<_DesiredType_>) {
            return _mm512_cmplt_epi32_mask(_IntrinBitcast<__m512i>(_Left), _IntrinBitcast<__m512i>(_Right));
        }
        else if constexpr (_Is_epu32_v<_DesiredType_>) {
            return _mm512_cmplt_epu32_mask(_IntrinBitcast<__m512i>(_Left), _IntrinBitcast<__m512i>(_Right));
        }
        else if constexpr (_Is_ps_v<_DesiredType_>) {
            return _mm512_cmplt_ps_mask(_IntrinBitcast<__m512>(_Left), _IntrinBitcast<__m512>(_Right));
        }
        else if constexpr (_Is_pd_v<_DesiredType_>) {
            return _mm512_cmplt_pd_mask(_IntrinBitcast<__m512d>(_Left), _IntrinBitcast<__m512d>(_Right));
        }
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _Simd_mask_type<_DesiredType_> _MaskCompareGreater(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept
    {
        if constexpr (_Is_epi64_v<_DesiredType_>) {
            return _mm512_cmpgt_epi64_mask(_IntrinBitcast<__m512i>(_Left), _IntrinBitcast<__m512i>(_Right));
        }
        else if constexpr (_Is_epu64_v<_DesiredType_>) {
            return _mm512_cmpgt_epu64_mask(_IntrinBitcast<__m512i>(_Left), _IntrinBitcast<__m512i>(_Right));
        }
        else if constexpr (_Is_epi32_v<_DesiredType_>) {
            return _mm512_cmpgt_epi32_mask(_IntrinBitcast<__m512i>(_Left), _IntrinBitcast<__m512i>(_Right));
        }
        else if constexpr (_Is_epu32_v<_DesiredType_>) {
            return _mm512_cmpgt_epu32_mask(_IntrinBitcast<__m512i>(_Left), _IntrinBitcast<__m512i>(_Right));
        }
        else if constexpr (_Is_ps_v<_DesiredType_>) {
            return _mm512_cmplt_ps_mask(_IntrinBitcast<__m512>(_Right), _IntrinBitcast<__m512>(_Left));
        }
        else if constexpr (_Is_pd_v<_DesiredType_>) {
            return _mm512_cmplt_pd_mask(_IntrinBitcast<__m512d>(_Right), _IntrinBitcast<__m512d>(_Left));
        }
    }
};

#pragma endregion 

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
    return _SimdCompareImplementation<_SimdGeneration_, _RegisterPolicy_>::template
        _MaskCompare<_DesiredType_, _CompareType_>(_Left, _Right);
}

__SIMD_STL_NUMERIC_NAMESPACE_END
