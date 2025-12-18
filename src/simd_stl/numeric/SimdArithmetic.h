#pragma once 


#include <src/simd_stl/numeric/SimdElementAccess.h>
#include <src/simd_stl/numeric/SimdDivisors.h>

#include <src/simd_stl/numeric/SimdCompare.h>


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
    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline auto _Reduce(_VectorType_ _Vector) noexcept;

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _ShiftRightVector(
        _VectorType_    _Vector,
        uint32          _ByteShift) noexcept;

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _ShiftLeftVector(
        _VectorType_    _Vector,
        uint32          _ByteShift) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _ShiftRightElements(
        _VectorType_    _Vector,
        uint32          _BitShift) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _ShiftLeftElements(
        _VectorType_    _Vector,
        uint32          _BitShift) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _Negate(_VectorType_ _Vector) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _Add(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _Substract(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _Multiply(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _Divide(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept;

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _BitNot(_VectorType_ _Vector) noexcept;

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _BitXor(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept;

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _BitAnd(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept;

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _BitOr(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _Min(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _DesiredType_ _Min(_VectorType_ _Vector) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _Max(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _DesiredType_ _Max(_VectorType_ _Vector) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _Abs(_VectorType_ _Vector) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _AdjustToUnsigned(_VectorType_ _Vector) noexcept;
};

template <>
class _SimdArithmetic<arch::CpuFeature::SSE3, xmm128>:
    public _SimdArithmetic<arch::CpuFeature::SSE2, xmm128>
{};

template <>
class _SimdArithmetic<arch::CpuFeature::SSSE3, xmm128>:
    public _SimdArithmetic<arch::CpuFeature::SSE3, xmm128>
{
    static constexpr auto _Generation   = arch::CpuFeature::SSE2;
    using _RegisterPolicy               = numeric::xmm128;
public:
    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline auto _Reduce(_VectorType_ _Vector) noexcept;
};

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
        _VectorType_ _Right) noexcept;
};

template <>
class _SimdArithmetic<arch::CpuFeature::SSE42, xmm128>:
    public _SimdArithmetic<arch::CpuFeature::SSE41, xmm128>
{};

#pragma endregion 

#pragma region Avx Simd arithmetic

template <>
class _SimdArithmetic<arch::CpuFeature::AVX, ymm256> {};

template <>
class _SimdArithmetic<arch::CpuFeature::AVX2, ymm256>:
    public _SimdArithmetic<arch::CpuFeature::AVX, ymm256> 
{
    static constexpr auto _Generation   = arch::CpuFeature::AVX2;
    using _RegisterPolicy               = numeric::ymm256;
public:
    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline auto _Reduce(_VectorType_ _Vector) noexcept;

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _ShiftRightVector(
        _VectorType_    _Vector,
        uint32          _ByteShift) noexcept;

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _ShiftLeftVector(
        _VectorType_    _Vector,
        uint32          _ByteShift) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _ShiftRightElements(
        _VectorType_    _Vector,
        uint32          _BitShift) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _ShiftLeftElements(
        _VectorType_    _Vector,
        uint32          _BitShift) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _Negate(_VectorType_ _Vector) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _Add(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _Substract(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _Multiply(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _Divide(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept;

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _BitNot(_VectorType_ _Vector) noexcept;

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _BitXor(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept;

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _BitAnd(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept;

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _BitOr(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _Min(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _DesiredType_ _Min(_VectorType_ _Vector) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _Max(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _DesiredType_ _Max(_VectorType_ _Vector) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _Abs(_VectorType_ _Vector) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _AdjustToUnsigned(_VectorType_ _Vector) noexcept;
};

#pragma endregion

#pragma region Avx512 Simd arithmetic

template <>
class _SimdArithmetic<arch::CpuFeature::AVX512F, zmm512> {
    static constexpr auto _Generation   = arch::CpuFeature::AVX512F;
    using _RegisterPolicy               = zmm512;
public:
    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _Min(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _DesiredType_ _Min(_VectorType_ _Vector) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _Max(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _DesiredType_ _Max(_VectorType_ _Vector) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _Abs(_VectorType_ _Vector) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _AdjustToUnsigned(_VectorType_ _Vector) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline auto _Reduce(_VectorType_ _Vector) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _ShiftRightVector(
        _VectorType_    _Vector,
        uint32          _ByteShift) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _ShiftLeftVector(
        _VectorType_    _Vector,
        uint32          _ByteShift) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _ShiftRightElements(
        _VectorType_    _Vector,
        uint32          _BitShift) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _ShiftLeftElements(
        _VectorType_    _Vector,
        uint32          _BitShift) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _Add(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _Substract(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _Multiply(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _Divide(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _Negate(_VectorType_ _Vector) noexcept;

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _BitNot(_VectorType_ _Vector) noexcept;

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _BitXor(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept;

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _BitAnd(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept;

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _BitOr(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept;
};

template <>
class _SimdArithmetic<arch::CpuFeature::AVX512BW, zmm512> :
    public _SimdArithmetic<arch::CpuFeature::AVX512F, zmm512>
{
public:
    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline auto _Reduce(_VectorType_ _Vector) noexcept;
};

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

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_,
    typename            _DesiredType_,
    typename            _VectorType_>
simd_stl_always_inline auto _SimdReduce(_VectorType_ _Vector) noexcept {
    _VerifyRegisterPolicy(_SimdGeneration_, _RegisterPolicy_);
    return _SimdArithmetic<_SimdGeneration_, _RegisterPolicy_>::template _Reduce<_DesiredType_>(_Vector);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_,
    typename            _DesiredType_,
    typename            _VectorType_>
simd_stl_always_inline _VectorType_ _SimdMin(
    _VectorType_ _Left,
    _VectorType_ _Right) noexcept
{
    _VerifyRegisterPolicy(_SimdGeneration_, _RegisterPolicy_);
    return _SimdArithmetic<_SimdGeneration_, _RegisterPolicy_>::template _Min<_DesiredType_>(_Left, _Right);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_,
    typename            _DesiredType_,
    typename            _VectorType_>
simd_stl_always_inline _DesiredType_ _SimdMax(_VectorType_ _Vector) noexcept {
    _VerifyRegisterPolicy(_SimdGeneration_, _RegisterPolicy_);
    return _SimdArithmetic<_SimdGeneration_, _RegisterPolicy_>::template _Max<_DesiredType_>(_Vector);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_,
    typename            _DesiredType_,
    typename            _VectorType_>
simd_stl_always_inline _DesiredType_ _SimdMin(_VectorType_ _Vector) noexcept {
    _VerifyRegisterPolicy(_SimdGeneration_, _RegisterPolicy_);
    return _SimdArithmetic<_SimdGeneration_, _RegisterPolicy_>::template _Min<_DesiredType_>(_Vector);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_,
    typename            _DesiredType_,
    typename            _VectorType_>
simd_stl_always_inline _VectorType_ _SimdMax(
    _VectorType_ _Left, 
    _VectorType_ _Right) noexcept
{
    _VerifyRegisterPolicy(_SimdGeneration_, _RegisterPolicy_);
    return _SimdArithmetic<_SimdGeneration_, _RegisterPolicy_>::template _Max<_DesiredType_>(_Left, _Right);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_,
    typename            _DesiredType_,
    typename            _VectorType_>
simd_stl_always_inline _VectorType_ _SimdAbs(_VectorType_ _Vector) noexcept {
    _VerifyRegisterPolicy(_SimdGeneration_, _RegisterPolicy_);
    return _SimdArithmetic<_SimdGeneration_, _RegisterPolicy_>::template _Abs<_DesiredType_>(_Vector);
}

__SIMD_STL_NUMERIC_NAMESPACE_END

#include <src/simd_stl/numeric/SimdArithmetic.inl>