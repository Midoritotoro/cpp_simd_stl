#pragma once 


#include <src/simd_stl/numeric/SimdElementAccess.h>
#include <src/simd_stl/numeric/SimdDivisors.h>


__SIMD_STL_NUMERIC_NAMESPACE_BEGIN

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_>
class _SimdArithmetic;

#pragma region Sse2-Sse4.2 Simd arithmetic

template <>
class _SimdArithmetic<arch::CpuFeature::SSE2, xmm128> {
    static constexpr auto _Generation   = arch::CpuFeature::SSE2;
    using _RegisterPolicy               = numeric::xmm128;
public:
    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _ShiftRightVector(
        _VectorType_    _Vector,
        uint32          _ByteShift) noexcept
    {
        return _IntrinBitcast<_VectorType_>(_mm_srli_si128(_IntrinBitcast<__m128i>(_Vector), _ByteShift));
    }

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _ShiftLeftVector(
        _VectorType_    _Vector,
        uint32          _ByteShift) noexcept
    {
        return _IntrinBitcast<_VectorType_>(_mm_slli_si128(_IntrinBitcast<__m128i>(_Vector), _ByteShift));
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _ShiftRightElements(
        _VectorType_    _Vector,
        uint32          _BitShift) noexcept
    {
        if      constexpr (_Is_epi64_v<_DesiredType_> || _Is_epu64_v<_DesiredType_>)
            return _IntrinBitcast<_VectorType_>(_mm_srli_epi64(_IntrinBitcast<__m128i>(_Vector), _BitShift));

        else if constexpr (_Is_epi32_v<_DesiredType_> || _Is_epu32_v<_DesiredType_>)
            return _IntrinBitcast<_VectorType_>(_mm_srli_epi32(_IntrinBitcast<__m128i>(_Vector), _BitShift));

        else if constexpr (_Is_epi16_v<_DesiredType_> || _Is_epu16_v<_DesiredType_>)
            return _IntrinBitcast<_VectorType_>(_mm_srli_epi16(_IntrinBitcast<__m128i>(_Vector), _BitShift));

        else if constexpr (_Is_epi8_v<_DesiredType_> || _Is_epu8_v<_DesiredType_>) {
            const auto _EvenVector = _mm_sra_epi16(_mm_slli_epi16(_IntrinBitcast<__m128i>(_Vector), 8), _mm_cvtsi32_si128(_BitShift + 8));
            const auto _OddVector = _mm_sra_epi16(_IntrinBitcast<__m128i>(_Vector), _mm_cvtsi32_si128(_BitShift));

            const auto _Mask = _mm_set1_epi32(0x00FF00FF);
            return _IntrinBitcast<_VectorType_>(_mm_or_si128(_mm_and_si128(_Mask, _EvenVector), _mm_andnot_si128(_Mask, _OddVector)));
        }
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _ShiftLeftElements(
        _VectorType_    _Vector,
        uint32          _BitShift) noexcept
    {
        if constexpr (_Is_epi64_v<_DesiredType_> || _Is_epu64_v<_DesiredType_>)
            return _IntrinBitcast<_VectorType_>(_mm_slli_epi64(_IntrinBitcast<__m128i>(_Vector), _BitShift));

        else if constexpr (_Is_epi32_v<_DesiredType_> || _Is_epu32_v<_DesiredType_>)
            return _IntrinBitcast<_VectorType_>(_mm_slli_epi32(_IntrinBitcast<__m128i>(_Vector), _BitShift));

        else if constexpr (_Is_epi16_v<_DesiredType_> || _Is_epu16_v<_DesiredType_>)
            return _IntrinBitcast<_VectorType_>(_mm_slli_epi16(_IntrinBitcast<__m128i>(_Vector), _BitShift));

        else if constexpr (_Is_epi8_v<_DesiredType_> || _Is_epu8_v<_DesiredType_>) {
            const auto _AndMask  = _mm_and_si128(_IntrinBitcast<__m128i>(_Vector),
                _mm_set1_epi8(static_cast<int8>(0xFFu >> _BitShift)));

            return _IntrinBitcast<_VectorType_>(_mm_sll_epi16(_AndMask, _mm_cvtsi32_si128(_BitShift)));
        }
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _Negate(_VectorType_ _Vector) noexcept {
        if      constexpr (_Is_ps_v<_DesiredType_>)
            return _IntrinBitcast<_VectorType_>(_mm_xor_ps(_Vector, _mm_set1_ps(-0.0f)));

        else if constexpr (_Is_pd_v<_DesiredType_>)
            return _IntrinBitcast<_VectorType_>(_mm_xor_pd(_Vector,
                _IntrinBitcast<__m128d>(_mm_setr_epi32(0, 0x80000000, 0, 0x80000000))));

        else
            return _Substract<_DesiredType_>(_SimdBroadcastZeros<_Generation, _RegisterPolicy, _VectorType_>(), _Vector);
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _Add(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept
    {
        if constexpr (_Is_epi64_v<_DesiredType_> || _Is_epu64_v<_DesiredType_>)
            return _IntrinBitcast<_VectorType_>(_mm_add_epi64(_IntrinBitcast<__m128i>(_Left), _IntrinBitcast<__m128i>(_Right)));

        else if constexpr (_Is_epi32_v<_DesiredType_> || _Is_epu32_v<_DesiredType_>)
            return _IntrinBitcast<_VectorType_>(_mm_add_epi32(_IntrinBitcast<__m128i>(_Left), _IntrinBitcast<__m128i>(_Right)));

        else if constexpr (_Is_epi16_v<_DesiredType_> || _Is_epu16_v<_DesiredType_>)
            return _IntrinBitcast<_VectorType_>(_mm_add_epi16(_IntrinBitcast<__m128i>(_Left), _IntrinBitcast<__m128i>(_Right)));

        else if constexpr (_Is_epi8_v<_DesiredType_> || _Is_epu8_v<_DesiredType_>)
            return _IntrinBitcast<_VectorType_>(_mm_add_epi8(_IntrinBitcast<__m128i>(_Left), _IntrinBitcast<__m128i>(_Right)));

        else if constexpr (_Is_ps_v<_DesiredType_>)
            return _IntrinBitcast<_VectorType_>(_mm_add_ps(_IntrinBitcast<__m128>(_Left), _IntrinBitcast<__m128>(_Right)));

        else if constexpr (_Is_pd_v<_DesiredType_>)
            return _IntrinBitcast<_VectorType_>(_mm_add_pd(_IntrinBitcast<__m128d>(_Left), _IntrinBitcast<__m128d>(_Right)));
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _Substract(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept
    {
        if constexpr (_Is_epi64_v<_DesiredType_> || _Is_epu64_v<_DesiredType_>)
            return _IntrinBitcast<_VectorType_>(_mm_sub_epi64(_IntrinBitcast<__m128i>(_Left), _IntrinBitcast<__m128i>(_Right)));

        else if constexpr (_Is_epi32_v<_DesiredType_> || _Is_epu32_v<_DesiredType_>)
            return _IntrinBitcast<_VectorType_>(_mm_sub_epi32(_IntrinBitcast<__m128i>(_Left), _IntrinBitcast<__m128i>(_Right)));

        else if constexpr (_Is_epi16_v<_DesiredType_> || _Is_epu16_v<_DesiredType_>)
            return _IntrinBitcast<_VectorType_>(_mm_sub_epi16(_IntrinBitcast<__m128i>(_Left), _IntrinBitcast<__m128i>(_Right)));

        else if constexpr (_Is_epi8_v<_DesiredType_> || _Is_epu8_v<_DesiredType_>)
            return _IntrinBitcast<_VectorType_>(_mm_sub_epi8(_IntrinBitcast<__m128i>(_Left), _IntrinBitcast<__m128i>(_Right)));

        else if constexpr (_Is_ps_v<_DesiredType_>)
            return _IntrinBitcast<_VectorType_>(_mm_sub_ps(_IntrinBitcast<__m128>(_Left), _IntrinBitcast<__m128>(_Right)));

        else if constexpr (_Is_pd_v<_DesiredType_>)
            return _IntrinBitcast<_VectorType_>(_mm_sub_pd(_IntrinBitcast<__m128d>(_Left), _IntrinBitcast<__m128d>(_Right)));
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _Multiply(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept
    {
       if constexpr (_Is_epi64_v<_DesiredType_>)
            return _IntrinBitcast<_VectorType_>(_mm_mul_epi64(_IntrinBitcast<__m128i>(_Left), _IntrinBitcast<__m128i>(_Right)));

        else if constexpr (_Is_epu64_v<_DesiredType_>)
            return _IntrinBitcast<_VectorType_>(_mm_mul_epu64(_IntrinBitcast<__m128i>(_Left), _IntrinBitcast<__m128i>(_Right)));

        else if constexpr (_Is_epi32_v<_DesiredType_>) {
           const auto _ShuffledLeft         = _mm_shuffle_epi32(_IntrinBitcast<__m128i>(_Left), 0xF5);
           const auto _ShuffledRight        = _mm_shuffle_epi32(_IntrinBitcast<__m128i>(_Right), 0xF5);

           const auto _ProductEvenIndices   = _mm_mul_epu32(_Left, _Right);
           const auto _ProductOddIndices    = _mm_mul_epu32(_ShuffledLeft, _ShuffledRight);

           const auto _ProductLowPair       = _mm_unpacklo_epi32(_ProductEvenIndices, _ProductOddIndices);
           const auto _ProductHighPair      = _mm_unpackhi_epi32(_ProductEvenIndices, _ProductOddIndices);

           return _IntrinBitcast<_VectorType_>(_mm_unpacklo_epi64(_ProductLowPair, _ProductHighPair));
        }

        else if constexpr (_Is_epu32_v<_DesiredType_>)
            return _IntrinBitcast<_VectorType_>(_mm_mul_epu32(_IntrinBitcast<__m128i>(_Left), _IntrinBitcast<__m128i>(_Right)));

        else if constexpr (_Is_epi16_v<_DesiredType_>)
            return _IntrinBitcast<_VectorType_>(_mm_mul_epi16(_IntrinBitcast<__m128i>(_Left), _IntrinBitcast<__m128i>(_Right)));

        else if constexpr (_Is_epu16_v<_DesiredType_>)
            return _IntrinBitcast<_VectorType_>(_mm_mul_epu16(_IntrinBitcast<__m128i>(_Left), _IntrinBitcast<__m128i>(_Right)));

        else if constexpr (_Is_epi8_v<_DesiredType_>)
            return _IntrinBitcast<_VectorType_>(_mm_mul_epi8(_IntrinBitcast<__m128i>(_Left), _IntrinBitcast<__m128i>(_Right)));

        else if constexpr (_Is_epu8_v<_DesiredType_>)
            return _IntrinBitcast<_VectorType_>(_mm_mul_epi8(_IntrinBitcast<__m128i>(_Left), _IntrinBitcast<__m128i>(_Right)));

        else if constexpr (_Is_ps_v<_DesiredType_>)
            return _IntrinBitcast<_VectorType_>(_mm_mul_ps(_IntrinBitcast<__m128>(_Left), _IntrinBitcast<__m128>(_Right)));

        else if constexpr (_Is_pd_v<_DesiredType_>)
            return _IntrinBitcast<_VectorType_>(_mm_mul_pd(_IntrinBitcast<__m128d>(_Left), _IntrinBitcast<__m128d>(_Right)));
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _Divide(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept
    {
        if constexpr (_Is_pd_v<_DesiredType_>)
            return _IntrinBitcast<_VectorType_>(_mm_div_pd(_IntrinBitcast<__m128d>(_Left), _IntrinBitcast<__m128d>(_Right)));

        else if constexpr (_Is_ps_v<_DesiredType_>)
            return _IntrinBitcast<_VectorType_>(_mm_div_ps(_IntrinBitcast<__m128>(_Left), _IntrinBitcast<__m128>(_Right)));
    }

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _BitNot(_VectorType_ _Vector) noexcept {
        if      constexpr (std::is_same_v<_VectorType_, __m128d>)
            return _mm_xor_pd(_Vector, _mm_cmpeq_pd(_Vector, _Vector));

        else if constexpr (std::is_same_v<_VectorType_, __m128i>)
            return _mm_xor_si128(_Vector, _mm_cmpeq_epi32(_Vector, _Vector));

        else if constexpr (std::is_same_v<_VectorType_, __m128>)
            return _mm_xor_ps(_Vector, _mm_cmpeq_ps(_Vector, _Vector));
    }

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _BitXor(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept
    {
        if      constexpr (std::is_same_v<_VectorType_, __m128d>)
            return _mm_xor_pd(_Left, _Right);

        else if constexpr (std::is_same_v<_VectorType_, __m128i>)
            return _mm_xor_si128(_Left, _Right);

        else if constexpr (std::is_same_v<_VectorType_, __m128>)
            return _mm_xor_ps(_Left, _Right);
    }

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _BitAnd(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept
    {
        if      constexpr (std::is_same_v<_VectorType_, __m128d>)
            return _mm_and_pd(_Left, _Right);

        else if constexpr (std::is_same_v<_VectorType_, __m128i>)
            return _mm_and_si128(_Left, _Right);

        else if constexpr (std::is_same_v<_VectorType_, __m128>)
            return _mm_and_ps(_Left, _Right);
    }

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _BitOr(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept
    {
        if      constexpr (std::is_same_v<_VectorType_, __m128d>)
            return _mm_or_pd(_Left, _Right);

        else if constexpr (std::is_same_v<_VectorType_, __m128i>)
            return _mm_or_si128(_Left, _Right);

        else if constexpr (std::is_same_v<_VectorType_, __m128>)
            return _mm_or_ps(_Left, _Right);
    }
};

template <>
class _SimdArithmetic<arch::CpuFeature::SSE3, xmm128>:
    public _SimdArithmetic<arch::CpuFeature::SSE2, xmm128>
{};

template <>
class _SimdArithmetic<arch::CpuFeature::SSSE3, xmm128>:
    public _SimdArithmetic<arch::CpuFeature::SSE3, xmm128>
{};

template <>
class _SimdArithmetic<arch::CpuFeature::SSE41, xmm128>:
    public _SimdArithmetic<arch::CpuFeature::SSSE3, xmm128>
{
public:
    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _Multiply(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept
    {
       if constexpr (_Is_epi64_v<_DesiredType_>)
            return _IntrinBitcast<_VectorType_>(_mm_mul_epi64(_IntrinBitcast<__m128i>(_Left), _IntrinBitcast<__m128i>(_Right)));

        else if constexpr (_Is_epu64_v<_DesiredType_>)
            return _IntrinBitcast<_VectorType_>(_mm_mul_epu64(_IntrinBitcast<__m128i>(_Left), _IntrinBitcast<__m128i>(_Right)));

        else if constexpr (_Is_epi32_v<_DesiredType_>)
           return _IntrinBitcast<_VectorType_>(_mm_mul_epi32(_IntrinBitcast<__m128i>(_Left), _IntrinBitcast<__m128i>(_Right)));

        else if constexpr (_Is_epu32_v<_DesiredType_>)
            return _IntrinBitcast<_VectorType_>(_mm_mul_epu32(_IntrinBitcast<__m128i>(_Left), _IntrinBitcast<__m128i>(_Right)));

        else if constexpr (_Is_epi16_v<_DesiredType_>)
            return _IntrinBitcast<_VectorType_>(_mm_mul_epi16(_IntrinBitcast<__m128i>(_Left), _IntrinBitcast<__m128i>(_Right)));

        else if constexpr (_Is_epu16_v<_DesiredType_>)
            return _IntrinBitcast<_VectorType_>(_mm_mul_epu16(_IntrinBitcast<__m128i>(_Left), _IntrinBitcast<__m128i>(_Right)));

        else if constexpr (_Is_epi8_v<_DesiredType_>)
            return _IntrinBitcast<_VectorType_>(_mm_mul_epi8(_IntrinBitcast<__m128i>(_Left), _IntrinBitcast<__m128i>(_Right)));

        else if constexpr (_Is_epu8_v<_DesiredType_>)
            return _IntrinBitcast<_VectorType_>(_mm_mul_epi8(_IntrinBitcast<__m128i>(_Left), _IntrinBitcast<__m128i>(_Right)));

        else if constexpr (_Is_ps_v<_DesiredType_>)
            return _IntrinBitcast<_VectorType_>(_mm_mul_ps(_IntrinBitcast<__m128>(_Left), _IntrinBitcast<__m128>(_Right)));

        else if constexpr (_Is_pd_v<_DesiredType_>)
            return _IntrinBitcast<_VectorType_>(_mm_mul_pd(_IntrinBitcast<__m128d>(_Left), _IntrinBitcast<__m128d>(_Right)));
    }
};

template <>
class _SimdArithmetic<arch::CpuFeature::SSE42, xmm128>:
    public _SimdArithmetic<arch::CpuFeature::SSE41, xmm128>
{};

#pragma endregion 

#pragma region Avx Simd arithmetic

template <>
class _SimdArithmetic<arch::CpuFeature::AVX, ymm256> {
    static constexpr auto _Generation = arch::CpuFeature::AVX;
    using _RegisterPolicy = numeric::ymm256;
public:
    /*template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _ShiftRightVector(
        _VectorType_    _Vector,
        uint32          _ByteShift) noexcept
    {
        auto _Low   = _IntrinBitcast<__m128i>(_Vector);
        auto _High  = _mm256_extractf128_si256(_IntrinBitcast<__m256i>(_Vector), 1);

        if (_ByteShift >= 16) {
            _High = _mm_srli_si128(_Low, _ByteShift - 16);
            _Low = _mm_setzero_si128();
        }
        else {
            const auto _LowXmmShift = _mm_srli_si128(_Low, _ByteShift);
            const auto _Shifted = _mm_alignr_epi8(_High, _Low, 16 - _ByteShift);

            _High = _Shifted;
            _Low = _LowXmmShift;
        }

        return _IntrinBitcast<_VectorType_>(_mm256_insertf128_si256(_IntrinBitcast<__m256i>(_Low), _High, 1));
    }*/
};

template <>
class _SimdArithmetic<arch::CpuFeature::AVX2, ymm256>:
    public _SimdArithmetic<arch::CpuFeature::AVX, ymm256> 
{
    static constexpr auto _Generation   = arch::CpuFeature::AVX2;
    using _RegisterPolicy               = numeric::ymm256;
public:
    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _ShiftRightVector(
        _VectorType_    _Vector,
        uint32          _ByteShift) noexcept
    {
        return _IntrinBitcast<_VectorType_>(_mm256_srli_si256(_IntrinBitcast<__m256i>(_Vector), _ByteShift));
    }

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _ShiftLeftVector(
        _VectorType_    _Vector,
        uint32          _ByteShift) noexcept
    {
        return _IntrinBitcast<_VectorType_>(_mm256_slli_si256(_IntrinBitcast<__m256i>(_Vector), _ByteShift));
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _ShiftRightElements(
        _VectorType_    _Vector,
        uint32          _BitShift) noexcept
    {
        if      constexpr (_Is_epi64_v<_DesiredType_> || _Is_epu64_v<_DesiredType_>)
            return _IntrinBitcast<_VectorType_>(_mm256_srli_epi64(_IntrinBitcast<__m256i>(_Vector), _BitShift));

        else if constexpr (_Is_epi32_v<_DesiredType_> || _Is_epu32_v<_DesiredType_>)
            return _IntrinBitcast<_VectorType_>(_mm256_srli_epi32(_IntrinBitcast<__m256i>(_Vector), _BitShift));

        else if constexpr (_Is_epi16_v<_DesiredType_> || _Is_epu16_v<_DesiredType_>)
            return _IntrinBitcast<_VectorType_>(_mm256_srli_epi16(_IntrinBitcast<__m256i>(_Vector), _BitShift));

        else if constexpr (_Is_epi8_v<_DesiredType_> || _Is_epu8_v<_DesiredType_>) {
            const auto _EvenVector = _mm256_sra_epi16(_mm256_slli_epi16(
                _IntrinBitcast<__m256i>(_Vector), 8), _mm_cvtsi32_si128(_BitShift + 8));

            const auto _OddVector = _mm256_sra_epi16(_IntrinBitcast<__m256i>(_Vector), _mm_cvtsi32_si128(_BitShift));

            const auto _Mask = _mm256_set1_epi32(0x00FF00FF);
            return _SimdBlend<_Generation, _RegisterPolicy, _DesiredType_>(_EvenVector, _OddVector, _Mask);
        }
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _ShiftLeftElements(
        _VectorType_    _Vector,
        uint32          _BitShift) noexcept
    {
        if constexpr (_Is_epi64_v<_DesiredType_> || _Is_epu64_v<_DesiredType_>)
            return _IntrinBitcast<_VectorType_>(_mm256_slli_epi64(_IntrinBitcast<__m256i>(_Vector), _BitShift));

        else if constexpr (_Is_epi32_v<_DesiredType_> || _Is_epu32_v<_DesiredType_>)
            return _IntrinBitcast<_VectorType_>(_mm256_slli_epi32(_IntrinBitcast<__m256i>(_Vector), _BitShift));

        else if constexpr (_Is_epi16_v<_DesiredType_> || _Is_epu16_v<_DesiredType_>)
            return _IntrinBitcast<_VectorType_>(_mm256_slli_epi16(_IntrinBitcast<__m256i>(_Vector), _BitShift));

        else if constexpr (_Is_epi8_v<_DesiredType_> || _Is_epu8_v<_DesiredType_>) {
            const auto _AndMask = _mm256_and_si256(_IntrinBitcast<__m256i>(_Vector),
                _mm256_set1_epi8(static_cast<int8>(0xFFu >> _BitShift)));

            return _IntrinBitcast<_VectorType_>(_mm256_sll_epi16(_AndMask, _mm_cvtsi32_si128(_BitShift)));
        }
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _Negate(_VectorType_ _Vector) noexcept {
        if      constexpr (_Is_ps_v<_DesiredType_>)
            return _IntrinBitcast<_VectorType_>(_mm256_xor_ps(_IntrinBitcast<__m256>(_Vector), _mm256_set1_ps(-0.0f)));

        else if constexpr (_Is_pd_v<_DesiredType_>)
            return _IntrinBitcast<_VectorType_>(_mm256_xor_pd(_IntrinBitcast<__m256d>(_Vector),
                _IntrinBitcast<__m256d>(_mm256_setr_epi32(0, 0x80000000, 0, 0x80000000, 0, 0x80000000, 0, 0x80000000))));

        else
            return _Substract<_DesiredType_>(_SimdBroadcastZeros<_Generation, _RegisterPolicy, _VectorType_>(), _Vector);
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _Add(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept
    {
        if constexpr (_Is_epi64_v<_DesiredType_> || _Is_epu64_v<_DesiredType_>)
            return _IntrinBitcast<_VectorType_>(_mm256_add_epi64(_IntrinBitcast<__m256i>(_Left), _IntrinBitcast<__m256i>(_Right)));

        else if constexpr (_Is_epi32_v<_DesiredType_> || _Is_epu32_v<_DesiredType_>)
            return _IntrinBitcast<_VectorType_>(_mm256_add_epi32(_IntrinBitcast<__m256i>(_Left), _IntrinBitcast<__m256i>(_Right)));

        else if constexpr (_Is_epi16_v<_DesiredType_> || _Is_epu16_v<_DesiredType_>)
            return _IntrinBitcast<_VectorType_>(_mm256_add_epi16(_IntrinBitcast<__m256i>(_Left), _IntrinBitcast<__m256i>(_Right)));

        else if constexpr (_Is_epi8_v<_DesiredType_> || _Is_epu8_v<_DesiredType_>)
            return _IntrinBitcast<_VectorType_>(_mm256_add_epi8(_IntrinBitcast<__m256i>(_Left), _IntrinBitcast<__m256i>(_Right)));

        else if constexpr (_Is_ps_v<_DesiredType_>)
            return _IntrinBitcast<_VectorType_>(_mm256_add_ps(_IntrinBitcast<__m256>(_Left), _IntrinBitcast<__m256>(_Right)));

        else if constexpr (_Is_pd_v<_DesiredType_>)
            return _IntrinBitcast<_VectorType_>(_mm256_add_pd(_IntrinBitcast<__m256d>(_Left), _IntrinBitcast<__m256d>(_Right)));
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _Substract(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept
    {
        if constexpr (_Is_epi64_v<_DesiredType_> || _Is_epu64_v<_DesiredType_>)
            return _IntrinBitcast<_VectorType_>(_mm256_sub_epi64(_IntrinBitcast<__m256i>(_Left), _IntrinBitcast<__m256i>(_Right)));

        else if constexpr (_Is_epi32_v<_DesiredType_> || _Is_epu32_v<_DesiredType_>)
            return _IntrinBitcast<_VectorType_>(_mm256_sub_epi32(_IntrinBitcast<__m256i>(_Left), _IntrinBitcast<__m256i>(_Right)));

        else if constexpr (_Is_epi16_v<_DesiredType_> || _Is_epu16_v<_DesiredType_>)
            return _IntrinBitcast<_VectorType_>(_mm256_sub_epi16(_IntrinBitcast<__m256i>(_Left), _IntrinBitcast<__m256i>(_Right)));

        else if constexpr (_Is_epi8_v<_DesiredType_> || _Is_epu8_v<_DesiredType_>)
            return _IntrinBitcast<_VectorType_>(_mm256_sub_epi8(_IntrinBitcast<__m256i>(_Left), _IntrinBitcast<__m256i>(_Right)));

        else if constexpr (_Is_ps_v<_DesiredType_>)
            return _IntrinBitcast<_VectorType_>(_mm256_sub_ps(_IntrinBitcast<__m256>(_Left), _IntrinBitcast<__m256>(_Right)));

        else if constexpr (_Is_pd_v<_DesiredType_>)
            return _IntrinBitcast<_VectorType_>(_mm256_sub_pd(_IntrinBitcast<__m256d>(_Left), _IntrinBitcast<__m256d>(_Right)));
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _Multiply(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept
    {
        if constexpr (_Is_epi32_v<_DesiredType_>)
           return _IntrinBitcast<_VectorType_>(_mm256_mul_epi32(_IntrinBitcast<__m256i>(_Left), _IntrinBitcast<__m256i>(_Right)));

        else if constexpr (_Is_epu32_v<_DesiredType_>)
            return _IntrinBitcast<_VectorType_>(_mm256_mul_epu32(_IntrinBitcast<__m256i>(_Left), _IntrinBitcast<__m256i>(_Right)));

        else if constexpr (_Is_ps_v<_DesiredType_>)
            return _IntrinBitcast<_VectorType_>(_mm256_mul_ps(_IntrinBitcast<__m256>(_Left), _IntrinBitcast<__m256>(_Right)));

        else if constexpr (_Is_pd_v<_DesiredType_>)
            return _IntrinBitcast<_VectorType_>(_mm256_mul_pd(_IntrinBitcast<__m256d>(_Left), _IntrinBitcast<__m256d>(_Right)));

        else {

        }
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _Divide(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept
    {
        if constexpr (_Is_pd_v<_DesiredType_>)
            return _IntrinBitcast<_VectorType_>(_mm256_div_pd(_IntrinBitcast<__m256d>(_Left), _IntrinBitcast<__m256d>(_Right)));

        else if constexpr (_Is_ps_v<_DesiredType_>)
            return _IntrinBitcast<_VectorType_>(_mm256_div_ps(_IntrinBitcast<__m256>(_Left), _IntrinBitcast<__m256>(_Right)));
    }

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _BitNot(_VectorType_ _Vector) noexcept {
        if      constexpr (std::is_same_v<_VectorType_, __m256d>)
            return _mm256_xor_pd(_Vector, _mm256_cmp_pd(_Vector, _Vector, _CMP_EQ_OQ));

        else if constexpr (std::is_same_v<_VectorType_, __m256i>)
            return _mm256_xor_si256(_Vector, _mm256_cmpeq_epi32(_Vector, _Vector));

        else if constexpr (std::is_same_v<_VectorType_, __m256>)
            return _mm256_xor_ps(_Vector, _mm256_cmp_ps(_Vector, _Vector, _CMP_EQ_OQ));
    }

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _BitXor(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept
    {
        if      constexpr (std::is_same_v<_VectorType_, __m256d>)
            return _mm256_xor_pd(_Left, _Right);

        else if constexpr (std::is_same_v<_VectorType_, __m256i>)
            return _mm256_xor_si256(_Left, _Right);

        else if constexpr (std::is_same_v<_VectorType_, __m256>)
            return _mm256_xor_ps(_Left, _Right);
    }

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _BitAnd(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept
    {
        if      constexpr (std::is_same_v<_VectorType_, __m256d>)
            return _mm256_and_pd(_Left, _Right);

        else if constexpr (std::is_same_v<_VectorType_, __m256i>)
            return _mm256_and_si256(_Left, _Right);

        else if constexpr (std::is_same_v<_VectorType_, __m256>)
            return _mm256_and_ps(_Left, _Right);
    }

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _BitOr(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept
    {
        if      constexpr (std::is_same_v<_VectorType_, __m256d>)
            return _mm256_or_pd(_Left, _Right);

        else if constexpr (std::is_same_v<_VectorType_, __m256i>)
            return _mm256_or_si256(_Left, _Right);

        else if constexpr (std::is_same_v<_VectorType_, __m256>)
            return _mm256_or_ps(_Left, _Right);
    }
};

#pragma endregion

#pragma region Avx512 Simd arithmetic

template <>
class _SimdArithmetic<arch::CpuFeature::AVX512F, zmm512> {
    static constexpr auto _Generation = arch::CpuFeature::AVX512F;
    using _RegisterPolicy = zmm512;
public:
    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _ShiftRightVector(
        _VectorType_    _Vector,
        uint32          _ByteShift) noexcept
    {
        auto _Low = _IntrinBitcast<__m256i>(_Vector);
        auto _High = _mm512_extracti64x4_epi64(_IntrinBitcast<__m512i>(_Vector), 1);

        if (_ByteShift >= 32) {
            _Low = _mm256_srli_si256(_High, _ByteShift - 32);
            _High = _mm256_setzero_si256();
        }
        else {
            const auto _LowYmmShift = _mm256_srli_si256(_Low, _ByteShift);
            const auto _Shifted = _mm256_alignr_epi8(_High, _Low, 32 - _ByteShift);

            _High = _Shifted;
            _Low = _LowYmmShift;
        }

        return _IntrinBitcast<_VectorType_>(_mm512_inserti64x4(_IntrinBitcast<__m512i>(_Low), _High, 1));
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _ShiftLeftVector(
        _VectorType_    _Vector,
        uint32          _ByteShift) noexcept
    {
        auto _Low = _IntrinBitcast<__m256i>(_Vector);
        auto _High = _mm512_extracti64x4_epi64(_IntrinBitcast<__m512i>(_Vector), 1);

        if (_ByteShift >= 32) {
            
        }
        else {
            const auto _LowYmmShift = _mm256_srli_si256(_Low, _ByteShift);
            const auto _Shifted = _mm256_alignr_epi8(_High, _Low, 32 - _ByteShift);

            _High = _Shifted;
            _Low = _LowYmmShift;
        }

        return _IntrinBitcast<_VectorType_>(_mm512_inserti64x4(_IntrinBitcast<__m512i>(_Low), _High, 1));
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _ShiftRightElements(
        _VectorType_    _Vector,
        uint32          _BitShift) noexcept
    {
        if constexpr (sizeof(_DesiredType_) == 8) {
            return _IntrinBitcast<_VectorType_>(_mm512_srli_epi64(_IntrinBitcast<__m512i>(_Vector), _BitShift));
        }
        else if constexpr (sizeof(_DesiredType_) == 4) {
            return _IntrinBitcast<_VectorType_>(_mm512_srli_epi32(_IntrinBitcast<__m512i>(_Vector), _BitShift));
        }
        else {
            const auto _Low = _SimdShiftRightElements<arch::CpuFeature::AVX2, ymm256, _DesiredType_>(
                _IntrinBitcast<__m256i>(_Vector), _BitShift);

            const auto _High = _SimdShiftRightElements<arch::CpuFeature::AVX2, ymm256, _DesiredType_>(
                _mm512_extracti64x4_epi64(_IntrinBitcast<__m512i>(_Vector), 1), _BitShift);

            return _IntrinBitcast<_VectorType_>(_mm512_inserti64x4(_IntrinBitcast<__m512i>(_Low), _High, 1));
        }
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _ShiftLeftElements(
        _VectorType_    _Vector,
        uint32          _BitShift) noexcept
    {
        if constexpr (sizeof(_DesiredType_) == 8) {
            return _IntrinBitcast<_VectorType_>(_mm512_slli_epi64(_IntrinBitcast<__m512i>(_Vector), _BitShift));
        }
        else if constexpr (sizeof(_DesiredType_) == 4) {
            return _IntrinBitcast<_VectorType_>(_mm512_slli_epi32(_IntrinBitcast<__m512i>(_Vector), _BitShift));
        }
        else {
            const auto _Low = _SimdShiftLeftElements<arch::CpuFeature::AVX2, ymm256, _DesiredType_>(
                _IntrinBitcast<__m256i>(_Vector), _BitShift);

            const auto _High = _SimdShiftLeftElements<arch::CpuFeature::AVX2, ymm256, _DesiredType_>(
                _mm512_extracti64x4_epi64(_IntrinBitcast<__m512i>(_Vector), 1), _BitShift);

            return _IntrinBitcast<_VectorType_>(_mm512_inserti64x4(_IntrinBitcast<__m512i>(_Low), _High, 1));
        }
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _Negate(_VectorType_ _Vector) noexcept {
        if      constexpr (_Is_ps_v<_DesiredType_>)
            return _IntrinBitcast<_VectorType_>(_mm512_xor_ps(_Vector, _mm512_set1_ps(-0.0f)));

        else if constexpr (_Is_pd_v<_DesiredType_>)
            return _IntrinBitcast<_VectorType_>(_mm512_xor_pd(_Vector,
                _IntrinBitcast<__m512d>(_mm512_setr_epi32(0, 0x80000000, 0, 0x80000000, 0, 0x80000000,
                    0, 0x80000000, 0, 0x80000000, 0, 0x80000000, 0, 0x80000000, 0, 0x80000000))));

        else
            return _Substract<_DesiredType_>(_SimdBroadcastZeros<_Generation, _RegisterPolicy, _VectorType_>(), _Vector);
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _Add(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept
    {
        if constexpr (_Is_pd_v<_DesiredType_>) { 
            return _IntrinBitcast<_VectorType_>(_mm512_add_pd(_IntrinBitcast<__m512d>(_Left), _IntrinBitcast<__m512d>(_Right)));
        }
        else if constexpr (_Is_ps_v<_DesiredType_>) {
            return _IntrinBitcast<_VectorType_>(_mm512_add_ps(_IntrinBitcast<__m512>(_Left), _IntrinBitcast<__m512>(_Right)));
        }
        else if constexpr (_Is_epi32_v<_DesiredType_> || _Is_epu32_v<_DesiredType_>) {
            return _IntrinBitcast<_VectorType_>(_mm512_add_epi32(_IntrinBitcast<__m512i>(_Left), _IntrinBitcast<__m512i>(_Right)));
        }
        else if constexpr (_Is_epi64_v<_DesiredType_> || _Is_epu64_v<_DesiredType_>) {
            return _IntrinBitcast<_VectorType_>(_mm512_add_epi64(_IntrinBitcast<__m512i>(_Left), _IntrinBitcast<__m512i>(_Right)));
        }
        else {
            const auto _Low = _SimdAdd<arch::CpuFeature::AVX2, ymm256, _DesiredType_>(
                _IntrinBitcast<__m256i>(_Left), _IntrinBitcast<__m256i>(_Right));
            
            const auto _High = _SimdAdd<arch::CpuFeature::AVX2, ymm256, _DesiredType_>(
                _mm512_extracti64x4_epi64(_IntrinBitcast<__m512i>(_Left), 1),
                _mm512_extracti64x4_epi64(_IntrinBitcast<__m512i>(_Right), 1));

            return _IntrinBitcast<_VectorType_>(_mm512_inserti64x4(_IntrinBitcast<__m512i>(_Low), _High, 1));
        }
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _Substract(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept
    {
        if constexpr (_Is_pd_v<_DesiredType_>) {
            return _IntrinBitcast<_VectorType_>(_mm512_sub_pd(_IntrinBitcast<__m512d>(_Left), _IntrinBitcast<__m512d>(_Right)));
        }
        else if constexpr (_Is_ps_v<_DesiredType_>) {
            return _IntrinBitcast<_VectorType_>(_mm512_sub_ps(_IntrinBitcast<__m512>(_Left), _IntrinBitcast<__m512>(_Right)));
        }
        else if constexpr (_Is_epi32_v<_DesiredType_> || _Is_epu32_v<_DesiredType_>) {
            return _IntrinBitcast<_VectorType_>(_mm512_sub_epi32(_IntrinBitcast<__m512i>(_Left), _IntrinBitcast<__m512i>(_Right)));
        }
        else if constexpr (_Is_epi64_v<_DesiredType_> || _Is_epu64_v<_DesiredType_>) {
            return _IntrinBitcast<_VectorType_>(_mm512_sub_epi64(_IntrinBitcast<__m512i>(_Left), _IntrinBitcast<__m512i>(_Right)));
        }
        else {
            const auto _Low = _SimdSubstract<arch::CpuFeature::AVX2, ymm256, _DesiredType_>(
                _IntrinBitcast<__m256i>(_Left), _IntrinBitcast<__m256i>(_Right));

            const auto _High = _SimdSubstract<arch::CpuFeature::AVX2, ymm256, _DesiredType_>(
                _mm512_extracti64x4_epi64(_IntrinBitcast<__m512i>(_Left), 1),
                _mm512_extracti64x4_epi64(_IntrinBitcast<__m512i>(_Right), 1));

            return _IntrinBitcast<_VectorType_>(_mm512_inserti64x4(_IntrinBitcast<__m512i>(_Low), _High, 1));
        }
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _Multiply(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept
    {
        return _Left;
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _Divide(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept
    {
        return _Left;
    }

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _BitNot(_VectorType_ _Vector) noexcept {
        if      constexpr (std::is_same_v<_VectorType_, __m512d>)
            return _mm512_xor_pd(_Vector, _mm512_set1_pd(-1));

        else if constexpr (std::is_same_v<_VectorType_, __m512i>)
            return _mm512_xor_si512(_Vector, _mm512_set1_epi32(-1));

        else if constexpr (std::is_same_v<_VectorType_, __m512>)
            return _mm512_xor_ps(_Vector, _mm512_set1_ps(-1));
    }

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _BitXor(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept
    {
        if      constexpr (std::is_same_v<_VectorType_, __m512d>)
            return _mm512_xor_pd(_Left, _Right);

        else if constexpr (std::is_same_v<_VectorType_, __m512i>)
            return _mm512_xor_si512(_Left, _Right);

        else if constexpr (std::is_same_v<_VectorType_, __m512>)
            return _mm512_xor_ps(_Left, _Right);
    }

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _BitAnd(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept
    {
        if      constexpr (std::is_same_v<_VectorType_, __m512d>)
            return _mm512_and_pd(_Left, _Right);

        else if constexpr (std::is_same_v<_VectorType_, __m512i>)
            return _mm512_and_si512(_Left, _Right);

        else if constexpr (std::is_same_v<_VectorType_, __m512>)
            return _mm512_and_ps(_Left, _Right);
    }

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _BitOr(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept
    {
        if      constexpr (std::is_same_v<_VectorType_, __m512d>)
            return _mm512_or_pd(_Left, _Right);

        else if constexpr (std::is_same_v<_VectorType_, __m512i>)
            return _mm512_or_si512(_Left, _Right);

        else if constexpr (std::is_same_v<_VectorType_, __m512>)
            return _mm512_or_ps(_Left, _Right);
    }
};

template <>
class _SimdArithmetic<arch::CpuFeature::AVX512BW, zmm512> :
    public _SimdArithmetic<arch::CpuFeature::AVX512F, zmm512>
{};

template <>
class _SimdArithmetic<arch::CpuFeature::AVX512DQ, zmm512> :
    public _SimdArithmetic<arch::CpuFeature::AVX512BW, zmm512>
{};

template <>
class _SimdArithmetic<arch::CpuFeature::AVX512VL, zmm512> :
    public _SimdArithmetic<arch::CpuFeature::AVX512DQ, zmm512>
{};

#pragma endregion

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_,
    typename            _DesiredType_,
    typename            _VectorType_>
static simd_stl_always_inline _VectorType_ _SimdShiftRightElements(
    _VectorType_    _Vector,
    uint32          _BitShift) noexcept
{
    _VerifyRegisterPolicy(_SimdGeneration_, _RegisterPolicy_);
    return _SimdArithmetic<_SimdGeneration_, _RegisterPolicy_>::template _ShiftRightElements<_DesiredType_>(_Vector, _BitShift);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_,
    typename            _DesiredType_,
    typename            _VectorType_>
static simd_stl_always_inline _VectorType_ _SimdShiftLeftElements(
    _VectorType_    _Vector,
    uint32          _BitShift) noexcept
{
    _VerifyRegisterPolicy(_SimdGeneration_, _RegisterPolicy_);
    return _SimdArithmetic<_SimdGeneration_, _RegisterPolicy_>::template _ShiftLeftElements<_DesiredType_>(_Vector, _BitShift);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_,
    typename            _VectorType_>
simd_stl_always_inline _VectorType_ _SimdShiftRightVector(
    _VectorType_    _Vector,
    uint32          _ByteShift) noexcept
{
    _VerifyRegisterPolicy(_SimdGeneration_, _RegisterPolicy_);
    return _SimdArithmetic<_SimdGeneration_, _RegisterPolicy_>::template _ShiftRightVector(_Vector, _ByteShift);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_,
    typename            _VectorType_>
simd_stl_always_inline _VectorType_ _SimdShiftLeftVector(
    _VectorType_    _Vector,
    uint32          _ByteShift) noexcept
{
    _VerifyRegisterPolicy(_SimdGeneration_, _RegisterPolicy_);
    return _SimdArithmetic<_SimdGeneration_, _RegisterPolicy_>::template _ShiftLeftVector(_Vector, _ByteShift);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_,
    typename            _DesiredType_,
    typename            _VectorType_>
simd_stl_always_inline _VectorType_ _SimdNegate(_VectorType_ _Vector) noexcept {
    _VerifyRegisterPolicy(_SimdGeneration_, _RegisterPolicy_);
    return _SimdArithmetic<_SimdGeneration_, _RegisterPolicy_>::template _Negate<_DesiredType_>(_Vector);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_,
    typename            _DesiredType_,
    typename            _VectorType_>
simd_stl_always_inline _VectorType_ _SimdAdd(
    _VectorType_ _Left,
    _VectorType_ _Right) noexcept
{
    _VerifyRegisterPolicy(_SimdGeneration_, _RegisterPolicy_);
    return _SimdArithmetic<_SimdGeneration_, _RegisterPolicy_>::template _Add<_DesiredType_>(_Left, _Right);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_,
    typename            _DesiredType_,
    typename            _VectorType_>
simd_stl_always_inline _VectorType_ _SimdSubstract(
    _VectorType_ _Left,
    _VectorType_ _Right) noexcept
{
    _VerifyRegisterPolicy(_SimdGeneration_, _RegisterPolicy_);
    return _SimdArithmetic<_SimdGeneration_, _RegisterPolicy_>::template _Substract<_DesiredType_>(_Left, _Right);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_,
    typename            _DesiredType_,
    typename            _VectorType_>
simd_stl_always_inline _VectorType_ _SimdMultiply(
    _VectorType_ _Left,
    _VectorType_ _Right) noexcept
{
    _VerifyRegisterPolicy(_SimdGeneration_, _RegisterPolicy_);
    return _SimdArithmetic<_SimdGeneration_, _RegisterPolicy_>::template _Multiply<_DesiredType_>(_Left, _Right);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_,
    typename            _DesiredType_,
    typename            _VectorType_>
simd_stl_always_inline _VectorType_ _SimdDivide(
    _VectorType_ _Left,
    _VectorType_ _Right) noexcept
{
    _VerifyRegisterPolicy(_SimdGeneration_, _RegisterPolicy_);
    return _SimdArithmetic<_SimdGeneration_, _RegisterPolicy_>::template _Divide<_DesiredType_>(_Left, _Right);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_,
    typename            _VectorType_>
simd_stl_always_inline _VectorType_ _SimdBitNot(_VectorType_ _Vector) noexcept {
    _VerifyRegisterPolicy(_SimdGeneration_, _RegisterPolicy_);
    return _SimdArithmetic<_SimdGeneration_, _RegisterPolicy_>::template _BitNot(_Vector);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_,
    typename            _VectorType_>
simd_stl_always_inline _VectorType_ _SimdBitXor(
    _VectorType_ _Left,
    _VectorType_ _Right) noexcept
{
    _VerifyRegisterPolicy(_SimdGeneration_, _RegisterPolicy_);
    return _SimdArithmetic<_SimdGeneration_, _RegisterPolicy_>::template _BitXor(_Left, _Right);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_,
    typename            _VectorType_>
simd_stl_always_inline _VectorType_ _SimdBitAnd(
    _VectorType_ _Left,
    _VectorType_ _Right) noexcept
{
    _VerifyRegisterPolicy(_SimdGeneration_, _RegisterPolicy_);
    return _SimdArithmetic<_SimdGeneration_, _RegisterPolicy_>::template _BitAnd(_Left, _Right);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_,
    typename            _VectorType_>
simd_stl_always_inline _VectorType_ _SimdBitOr(
    _VectorType_ _Left,
    _VectorType_ _Right) noexcept
{
    _VerifyRegisterPolicy(_SimdGeneration_, _RegisterPolicy_);
    return _SimdArithmetic<_SimdGeneration_, _RegisterPolicy_>::template _BitOr(_Left, _Right);
}

__SIMD_STL_NUMERIC_NAMESPACE_END
