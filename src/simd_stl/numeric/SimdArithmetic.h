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
    static simd_stl_always_inline _VectorType_ _ShiftRightVector(
        _VectorType_    _Vector,
        uint32          _ByteShift) noexcept
    {
        return _IntrinBitcast<_VectorType_>(_mm_srli_si128(_IntrinBitcast<__m128i>(_Vector), _ByteShift));
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
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

        else if constexpr (is_epi32_v<_DesiredType_>)
            return _IntrinBitcast<_VectorType_>(_mm_mul_epi32(_IntrinBitcast<__m128i>(_Left), _IntrinBitcast<__m128i>(_Right)));

        else if constexpr (is_epu32_v<_DesiredType_>)
            return _IntrinBitcast<_VectorType_>(_mm_mul_epu32(_IntrinBitcast<__m128i>(_Left), _IntrinBitcast<__m128i>(_Right)));

        else if constexpr (is_epi16_v<_DesiredType_>)
            return _IntrinBitcast<_VectorType_>(_mm_mul_epi16(_IntrinBitcast<__m128i>(_Left), _IntrinBitcast<__m128i>(_Right)));

        else if constexpr (is_epu16_v<_DesiredType_>)
            return _IntrinBitcast<_VectorType_>(_mm_mul_epu16(_IntrinBitcast<__m128i>(_Left), _IntrinBitcast<__m128i>(_Right)));

        else if constexpr (is_epi8_v<_DesiredType_>)
            return _IntrinBitcast<_VectorType_>(_mm_mul_epi8(_IntrinBitcast<__m128i>(_Left), _IntrinBitcast<__m128i>(_Right)));

        else if constexpr (is_epu8_v<_DesiredType_>)
            return _IntrinBitcast<_VectorType_>(_mm_mul_epi8(_IntrinBitcast<__m128i>(_Left), _IntrinBitcast<__m128i>(_Right)));

        else if constexpr (is_ps_v<_DesiredType_>)
            return _IntrinBitcast<_VectorType_>(_mm_mul_ps(_IntrinBitcast<__m128>(_Left), _IntrinBitcast<__m128>(_Right)));

        /*else if constexpr (is_pd_v<_DesiredType_>)
            return _mm_mul_pd(left, right);*/
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
{};

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
    return _SimdArithmetic<_SimdGeneration_, _RegisterPolicy_>::template _ShiftRightElements(_Vector, _BitShift);
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
    return _SimdArithmetic<_SimdGeneration_, _RegisterPolicy_>::template _ShiftLeftElements(_Vector, _BitShift);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_,
    typename            _DesiredType_,
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
    typename            _DesiredType_,
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
