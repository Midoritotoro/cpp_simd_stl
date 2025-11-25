#pragma once 


#include <src/simd_stl/numeric/SimdElementAccess.h>
#include <src/simd_stl/numeric/SimdDivisors.h>


__SIMD_STL_NUMERIC_NAMESPACE_BEGIN

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_>
class _SimdArithmetic;

template <>
class _SimdArithmetic<arch::CpuFeature::SSE2, xmm128> {
    static constexpr auto _Generation   = arch::CpuFeature::SSE2;
    using _RegisterPolicy               = numeric::xmm128;
public:
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
            const auto _EvenVector = _mm_sra_epi16(_mm_slli_epi16(
                _IntrinBitcast<__m128i>(_Vector), 8), _mm_cvtsi32_si128(_BitShift + 8));
            const auto _OddVector = _mm_sra_epi16(_IntrinBitcast<__m128i>(_Vector), _mm_cvtsi32_si128(_BitShift));

            const auto _Mask = _mm_set1_epi32(0x00FF00FF);

            return _IntrinBitcast<_VectorType_>(_mm_or_si128(
                _mm_and_si128(_Mask, _EvenVector), _mm_andnot_si128(_Mask, _OddVector)));
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
    static simd_stl_always_inline _VectorType_ _UnaryMinus(_VectorType_ _Vector) noexcept {
        if      constexpr (_Is_ps_v<_DesiredType_>)
            return _IntrinBitcast<_VectorType_>(_mm_xor_ps(_Vector, _mm_set1_ps(0x80000000)));

        else if constexpr (_Is_pd_v<_DesiredType_>)
            return _IntrinBitcast<_VectorType_>(_mm_xor_pd(_Vector,
                _IntrinBitcast<__m128d>(_mm_setr_epi32(0, 0x80000000, 0, 0x80000000))));

        else
            return _Substract<_DesiredType_>(_SimdBroadcastZeros<_Generation, _RegisterPolicy, _VectorType_>(), _Vector);
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _Decrement(_VectorType_ _Vector) noexcept {
        return _Substract<_DesiredType_>(_Vector, _SimdBroadcast<_Generation, _RegisterPolicy, _VectorType_, _DesiredType_>(1));
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _Increment(_VectorType_ _Vector) noexcept {
        return _Add<_DesiredType_>(_Vector, _SimdBroadcast<_Generation, _RegisterPolicy, _VectorType_, _DesiredType_>(1));
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _Add(
        _VectorType_ left,
        _VectorType_ right) noexcept
    {
        if constexpr (is_epi64_v<_DesiredType_> || is_epu64_v<_DesiredType_>)
            return _Cast_::template cast<__m128i, _VectorType_>(_mm_add_epi64(
                _Cast_::template cast<_VectorType_, __m128i>(left),
                _Cast_::template cast<_VectorType_, __m128i>(right)));
        else if constexpr (is_epi32_v<_DesiredType_> || is_epu32_v<_DesiredType_>)
            return _Cast_::template cast<__m128i, _VectorType_>(_mm_add_epi32(
                _Cast_::template cast<_VectorType_, __m128i>(left),
                _Cast_::template cast<_VectorType_, __m128i>(right)));
        else if constexpr (is_epi16_v<_DesiredType_> || is_epu16_v<_DesiredType_>)
            return _Cast_::template cast<__m128i, _VectorType_>(_mm_add_epi16(
                _Cast_::template cast<_VectorType_, __m128i>(left),
                _Cast_::template cast<_VectorType_, __m128i>(right)));
        else if constexpr (is_epi8_v<_DesiredType_> || is_epu8_v<_DesiredType_>)
            return _Cast_::template cast<__m128i, _VectorType_>(_mm_add_epi8(
                _Cast_::template cast<_VectorType_, __m128i>(left),
                _Cast_::template cast<_VectorType_, __m128i>(right)));
        else if constexpr (is_ps_v<_DesiredType_>)
            return _Cast_::template cast<__m128, _VectorType_>(_mm_add_ps(
                _Cast_::template cast<_VectorType_, __m128>(left),
                _Cast_::template cast<_VectorType_, __m128>(right)));
        else if constexpr (is_pd_v<_DesiredType_>)
            return _Cast_::template cast<__m128d, _VectorType_>(_mm_add_pd(
                _Cast_::template cast<_VectorType_, __m128d>(left),
                _Cast_::template cast<_VectorType_, __m128d>(right)));
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _Substract(
        _VectorType_ left,
        _VectorType_ right) noexcept
    {
        if constexpr (is_epi64_v<_DesiredType_> || is_epu64_v<_DesiredType_>)
            return _Cast_::template cast<__m128i, _VectorType_>(_mm_sub_epi64(
                _Cast_::template cast<_VectorType_, __m128i>(left),
                _Cast_::template cast<_VectorType_, __m128i>(right)));
        else if constexpr (is_epi32_v<_DesiredType_> || is_epu32_v<_DesiredType_>)
            return _Cast_::template cast<__m128i, _VectorType_>(_mm_sub_epi32(
                _Cast_::template cast<_VectorType_, __m128i>(left),
                _Cast_::template cast<_VectorType_, __m128i>(right)));
        else if constexpr (is_epi16_v<_DesiredType_> || is_epu16_v<_DesiredType_>)
            return _Cast_::template cast<__m128i, _VectorType_>(_mm_sub_epi16(
                _Cast_::template cast<_VectorType_, __m128i>(left),
                _Cast_::template cast<_VectorType_, __m128i>(right)));
        else if constexpr (is_epi8_v<_DesiredType_> || is_epu8_v<_DesiredType_>)
            return _Cast_::template cast<__m128i, _VectorType_>(_mm_sub_epi8(
                _Cast_::template cast<_VectorType_, __m128i>(left),
                _Cast_::template cast<_VectorType_, __m128i>(right)));
        else if constexpr (is_ps_v<_DesiredType_>)
            return _Cast_::template cast<__m128, _VectorType_>(_mm_sub_ps(
                _Cast_::template cast<_VectorType_, __m128>(left),
                _Cast_::template cast<_VectorType_, __m128>(right)));
        else if constexpr (is_pd_v<_DesiredType_>)
            return _Cast_::template cast<__m128d, _VectorType_>(_mm_sub_pd(
                _Cast_::template cast<_VectorType_, __m128d>(left),
                _Cast_::template cast<_VectorType_, __m128d>(right)));
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ mul(
        _VectorType_ left,
        _VectorType_ right) noexcept
    {
        // ? 
       if      constexpr (is_epi64_v<_DesiredType_>)
            return _Cast_::template cast<__m128i, _VectorType_>(_mm_mul_epi64(
                _Cast_::template cast<_VectorType_, __m128i>(left),
                _Cast_::template cast<_VectorType_, __m128i>(right)));
        else if constexpr (is_epu64_v<_DesiredType_>)
            return _Cast_::template cast<__m128i, _VectorType_>(_mm_mul_epu64(
                _Cast_::template cast<_VectorType_, __m128i>(left),
                _Cast_::template cast<_VectorType_, __m128i>(right)));
        else if constexpr (is_epi32_v<_DesiredType_>)
            return _Cast_::template cast<__m128i, _VectorType_>(_mm_mul_epi32(
                _Cast_::template cast<_VectorType_, __m128i>(left),
                _Cast_::template cast<_VectorType_, __m128i>(right)));
        else if constexpr (is_epu32_v<_DesiredType_>)
            return _Cast_::template cast<__m128i, _VectorType_>(_mm_mul_epu32(
                _Cast_::template cast<_VectorType_, __m128i>(left),
                _Cast_::template cast<_VectorType_, __m128i>(right)));
        else if constexpr (is_epi16_v<_DesiredType_>)
            return _Cast_::template cast<__m128i, _VectorType_>(_mm_mul_epi16(
                _Cast_::template cast<_VectorType_, __m128i>(left),
                _Cast_::template cast<_VectorType_, __m128i>(right)));
        else if constexpr (is_epu16_v<_DesiredType_>)
            return _Cast_::template cast<__m128i, _VectorType_>(_mm_mul_epu16(
                _Cast_::template cast<_VectorType_, __m128i>(left),
                _Cast_::template cast<_VectorType_, __m128i>(right)));
        else if constexpr (is_epi8_v<_DesiredType_>)
            return _Cast_::template cast<__m128i, _VectorType_>(_mm_mul_epi8(
                _Cast_::template cast<_VectorType_, __m128i>(left),
                _Cast_::template cast<_VectorType_, __m128i>(right)));
        else if constexpr (is_epu8_v<_DesiredType_>)
            return _Cast_::template cast<__m128i, _VectorType_>(_mm_mul_epi8(
                _Cast_::template cast<_VectorType_, __m128i>(left),
                _Cast_::template cast<_VectorType_, __m128i>(right)));
        else if constexpr (is_ps_v<_DesiredType_>)
            return _mm_mul_ps(left, right);
        /*else if constexpr (is_pd_v<_DesiredType_>)
            return _mm_mul_pd(left, right);*/
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ div(
        _VectorType_ left,
        _VectorType_ right) noexcept
    {
        if constexpr (is_pd_v<_DesiredType_>)
            return _Cast_::template cast<__m128d, _VectorType_>(_mm_div_pd(
                _Cast_::template cast<_VectorType_, __m128d>(left),
                _Cast_::template cast<_VectorType_, __m128d>(right)));
        else if constexpr (is_ps_v<_DesiredType_>)
            return _Cast_::template cast<__m128, _VectorType_>(_mm_div_ps(
                _Cast_::template cast<_VectorType_, __m128>(left),
                _Cast_::template cast<_VectorType_, __m128>(right)));
        else if constexpr (is_epi32_v<_DesiredType_>) {

        }
    }

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ bitwiseNot(_VectorType_ vector) noexcept {
        if      constexpr (std::is_same_v<_VectorType_, __m128d>)
            return _mm_xor_pd(vector, _mm_cmpeq_pd(vector, vector));
        else if constexpr (std::is_same_v<_VectorType_, __m128i>)
            return _mm_xor_si128(vector, _mm_cmpeq_epi32(vector, vector));
        else if constexpr (std::is_same_v<_VectorType_, __m128>)
            return _mm_xor_ps(vector, _mm_cmpeq_ps(vector, vector));
    }

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ bitwiseXor(
        _VectorType_ left,
        _VectorType_ right) noexcept
    {
        if      constexpr (std::is_same_v<_VectorType_, __m128d>)
            return _mm_xor_pd(left, right);
        else if constexpr (std::is_same_v<_VectorType_, __m128i>)
            return _mm_xor_si128(left, right);
        else if constexpr (std::is_same_v<_VectorType_, __m128>)
            return _mm_xor_ps(left, right);
    }

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ bitwiseAnd(
        _VectorType_ left,
        _VectorType_ right) noexcept
    {
        if      constexpr (std::is_same_v<_VectorType_, __m128d>)
            return _mm_and_pd(left, right);
        else if constexpr (std::is_same_v<_VectorType_, __m128i>)
            return _mm_and_si128(left, right);
        else if constexpr (std::is_same_v<_VectorType_, __m128>)
            return _mm_and_ps(left, right);
    }

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ bitwiseOr(
        _VectorType_ left,
        _VectorType_ right) noexcept
    {
        if      constexpr (std::is_same_v<_VectorType_, __m128d>)
            return _mm_or_pd(left, right);
        else if constexpr (std::is_same_v<_VectorType_, __m128i>)
            return _mm_or_si128(left, right);
        else if constexpr (std::is_same_v<_VectorType_, __m128>)
            return _mm_or_ps(left, right);
    }
};

template <class _RegisterPolicy_>
class _SimdArithmetic<arch::CpuFeature::SSE3, _RegisterPolicy_> :
    public _SimdArithmetic<arch::CpuFeature::SSE2, _RegisterPolicy_>
{};

template <class _RegisterPolicy_>
class _SimdArithmetic<arch::CpuFeature::SSSE3, _RegisterPolicy_> :
    public _SimdArithmetic<arch::CpuFeature::SSE3, _RegisterPolicy_>
{};

template <class _RegisterPolicy_>
class _SimdArithmetic<arch::CpuFeature::SSE41, _RegisterPolicy_> :
    public _SimdArithmetic<arch::CpuFeature::SSSE3, _RegisterPolicy_>
{};

template <class _RegisterPolicy_>
class _SimdArithmetic<arch::CpuFeature::SSE42, _RegisterPolicy_> :
    public _SimdArithmetic<arch::CpuFeature::SSE41, _RegisterPolicy_>
{
};

__SIMD_STL_NUMERIC_NAMESPACE_END
