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
    static constexpr auto __generation  = arch::CpuFeature::SSE2;
    using __register_policy             = numeric::xmm128;
public:
    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline auto __reduce(_VectorType_ __vector) noexcept;

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ __shift_right_vector(
        _VectorType_    __vector,
        uint32          __byte_shift) noexcept;

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ __shift_left_vector(
        _VectorType_    __vector,
        uint32          __byte_shift) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ __shift_right_elements(
        _VectorType_    __vector,
        uint32          __bit_shift) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ __shift_left_elements(
        _VectorType_    __vector,
        uint32          __bit_shift) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ __negate(_VectorType_ __vector) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ __add(
        _VectorType_ __left,
        _VectorType_ __right) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ __substract(
        _VectorType_ __left,
        _VectorType_ __right) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ __multiply(
        _VectorType_ __left,
        _VectorType_ __right) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ __divide(
        _VectorType_ __left,
        _VectorType_ __right) noexcept;

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ __bit_not(_VectorType_ __vector) noexcept;

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ __bit_xor(
        _VectorType_ __left,
        _VectorType_ __right) noexcept;

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ __bit_and(
        _VectorType_ __left,
        _VectorType_ __right) noexcept;

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ __bit_or(
        _VectorType_ __left,
        _VectorType_ __right) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ __vertical_min(
        _VectorType_ __left,
        _VectorType_ __right) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_,
        typename _ReduceBinaryFunction_>
    static simd_stl_always_inline _DesiredType_ __horizontal_fold(
        _VectorType_            __vector,
        _ReduceBinaryFunction_  __reduce) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _DesiredType_ __horizontal_min(_VectorType_ __vector) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ __vertical_max(
        _VectorType_ __left,
        _VectorType_ __right) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _DesiredType_ __horizontal_max(_VectorType_ __vector) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ __abs(_VectorType_ __vector) noexcept;
};

template <>
class _SimdArithmetic<arch::CpuFeature::SSE3, xmm128>:
    public _SimdArithmetic<arch::CpuFeature::SSE2, xmm128>
{};

template <>
class _SimdArithmetic<arch::CpuFeature::SSSE3, xmm128>:
    public _SimdArithmetic<arch::CpuFeature::SSE3, xmm128>
{
    static constexpr auto _Generation   = arch::CpuFeature::SSSE3;
    using _RegisterPolicy               = numeric::xmm128;
public:
    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline auto __reduce(_VectorType_ __vector) noexcept;

template <
        typename _DesiredType_,
        typename _VectorType_,
        typename _ReduceBinaryFunction_>
    static simd_stl_always_inline _DesiredType_ __horizontal_fold(
        _VectorType_            __vector,
        _ReduceBinaryFunction_  __reduce) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _DesiredType_ __horizontal_min(_VectorType_ __vector) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _DesiredType_ __horizontal_max(_VectorType_ __vector) noexcept;
};

template <>
class _SimdArithmetic<arch::CpuFeature::SSE41, xmm128>:
    public _SimdArithmetic<arch::CpuFeature::SSSE3, xmm128>
{
    static constexpr auto _Generation   = arch::CpuFeature::SSE41;
    using _RegisterPolicy               = numeric::xmm128;
public:
    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ __multiply(
        _VectorType_ __left,
        _VectorType_ __right) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ __vertical_min(
        _VectorType_ __left,
        _VectorType_ __right) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _DesiredType_ __horizontal_min(_VectorType_ __vector) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ __vertical_max(
        _VectorType_ __left,
        _VectorType_ __right) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _DesiredType_ __horizontal_max(_VectorType_ __vector) noexcept;
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
    static constexpr auto __generation  = arch::CpuFeature::AVX2;
    using __register_policy             = numeric::ymm256;
public:
    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline auto __reduce(_VectorType_ __vector) noexcept;

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ __shift_right_vector(
        _VectorType_    __vector,
        uint32          __byte_shift) noexcept;

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ __shift_left_vector(
        _VectorType_    __vector,
        uint32          __byte_shift) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ __shift_right_elements(
        _VectorType_    __vector,
        uint32          __bit_shift) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ __shift_left_elements(
        _VectorType_    __vector,
        uint32          __bit_shift) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ __negate(_VectorType_ __vector) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ __add(
        _VectorType_ __left,
        _VectorType_ __right) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ __substract(
        _VectorType_ __left,
        _VectorType_ __right) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ __multiply(
        _VectorType_ __left,
        _VectorType_ __right) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ __divide(
        _VectorType_ __left,
        _VectorType_ __right) noexcept;

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ __bit_not(_VectorType_ __vector) noexcept;

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ __bit_xor(
        _VectorType_ __left,
        _VectorType_ __right) noexcept;

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ __bit_and(
        _VectorType_ __left,
        _VectorType_ __right) noexcept;

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ __bit_or(
        _VectorType_ __left,
        _VectorType_ __right) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ __vertical_min(
        _VectorType_ __left,
        _VectorType_ __right) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_,
        typename _ReduceBinaryFunction_>
    static simd_stl_always_inline _DesiredType_ __horizontal_fold(
        _VectorType_            __vector,
        _ReduceBinaryFunction_  __reduce) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _DesiredType_ __horizontal_min(_VectorType_ __vector) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ __vertical_max(
        _VectorType_ __left,
        _VectorType_ __right) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _DesiredType_ __horizontal_max(_VectorType_ __vector) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ __abs(_VectorType_ __vector) noexcept;
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
    static simd_stl_always_inline auto __reduce(_VectorType_ __vector) noexcept;

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ __shift_right_vector(
        _VectorType_    __vector,
        uint32          __byte_shift) noexcept;

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ __shift_left_vector(
        _VectorType_    __vector,
        uint32          __byte_shift) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ __shift_right_elements(
        _VectorType_    __vector,
        uint32          __bit_shift) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ __shift_left_elements(
        _VectorType_    __vector,
        uint32          __bit_shift) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ __negate(_VectorType_ __vector) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ __add(
        _VectorType_ __left,
        _VectorType_ __right) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ __substract(
        _VectorType_ __left,
        _VectorType_ __right) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ __multiply(
        _VectorType_ __left,
        _VectorType_ __right) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ __divide(
        _VectorType_ __left,
        _VectorType_ __right) noexcept;

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ __bit_not(_VectorType_ __vector) noexcept;

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ __bit_xor(
        _VectorType_ __left,
        _VectorType_ __right) noexcept;

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ __bit_and(
        _VectorType_ __left,
        _VectorType_ __right) noexcept;

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ __bit_or(
        _VectorType_ __left,
        _VectorType_ __right) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ __vertical_min(
        _VectorType_ __left,
        _VectorType_ __right) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_,
        typename _ReduceBinaryFunction_>
    static simd_stl_always_inline _DesiredType_ __horizontal_fold(
        _VectorType_            __vector,
        _ReduceBinaryFunction_  __reduce) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _DesiredType_ __horizontal_min(_VectorType_ __vector) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ __vertical_max(
        _VectorType_ __left,
        _VectorType_ __right) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _DesiredType_ __horizontal_max(_VectorType_ __vector) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ __abs(_VectorType_ __vector) noexcept;
};

template <>
class _SimdArithmetic<arch::CpuFeature::AVX512BW, zmm512> :
    public _SimdArithmetic<arch::CpuFeature::AVX512F, zmm512>
{
    static constexpr auto _Generation   = arch::CpuFeature::AVX512BW;
    using _RegisterPolicy               = zmm512;
public:
    template <
        typename _DesiredType_,
        typename _VectorType_,
        typename _ReduceBinaryFunction_>
    static simd_stl_always_inline _DesiredType_ _HorizontalFold(
        _VectorType_            _Vector,
        _ReduceBinaryFunction_  _Reduce) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _VerticalMin(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _DesiredType_ _HorizontalMin(_VectorType_ _Vector) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _VerticalMax(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _DesiredType_ _HorizontalMax(_VectorType_ _Vector) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline auto _Reduce(_VectorType_ _Vector) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _Abs(_VectorType_ _Vector) noexcept;
};

template <>
class _SimdArithmetic<arch::CpuFeature::AVX512DQ, zmm512> :
    public _SimdArithmetic<arch::CpuFeature::AVX512F, zmm512>
{};

template <>
class _SimdArithmetic<arch::CpuFeature::AVX512VLF, ymm256>:
    public _SimdArithmetic<arch::CpuFeature::AVX2, ymm256>
{
    static constexpr auto _Generation   = arch::CpuFeature::AVX512VLF;
    using _RegisterPolicy               = ymm256;
public:
    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _VerticalMin(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _VerticalMax(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _Abs(_VectorType_ _Vector) noexcept;
};

template <>
class _SimdArithmetic<arch::CpuFeature::AVX512VLBW, ymm256> :
    public _SimdArithmetic<arch::CpuFeature::AVX512VLF, ymm256>
{};

template <>
class _SimdArithmetic<arch::CpuFeature::AVX512VLDQ, ymm256> :
    public _SimdArithmetic<arch::CpuFeature::AVX512VLF, ymm256>
{};

template <>
class _SimdArithmetic<arch::CpuFeature::AVX512VLBWDQ, ymm256>:
    public _SimdArithmetic<arch::CpuFeature::AVX512VLBW, ymm256>
{};

template <>
class _SimdArithmetic<arch::CpuFeature::AVX512VLF, xmm128> :
    public _SimdArithmetic<arch::CpuFeature::SSE42, xmm128>
{
    static constexpr auto _Generation = arch::CpuFeature::AVX512VLF;
    using _RegisterPolicy = xmm128;
public:
    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _VerticalMin(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _VerticalMax(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _Abs(_VectorType_ _Vector) noexcept;
};

template <>
class _SimdArithmetic<arch::CpuFeature::AVX512VLBW, xmm128> :
    public _SimdArithmetic<arch::CpuFeature::AVX512VLF, xmm128>
{
};

template <>
class _SimdArithmetic<arch::CpuFeature::AVX512VLDQ, xmm128> :
    public _SimdArithmetic<arch::CpuFeature::AVX512VLF, xmm128>
{
};


template <>
class _SimdArithmetic<arch::CpuFeature::AVX512VLBWDQ, xmm128>:
    public _SimdArithmetic<arch::CpuFeature::AVX512VLBW, xmm128>
{
};


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
simd_stl_always_inline _VectorType_ _SimdVerticalMin(
    _VectorType_ _Left,
    _VectorType_ _Right) noexcept
{
    _VerifyRegisterPolicy(_SimdGeneration_, _RegisterPolicy_);
    return _SimdArithmetic<_SimdGeneration_, _RegisterPolicy_>::template _VerticalMin<_DesiredType_>(_Left, _Right);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_,
    typename            _DesiredType_,
    typename            _VectorType_>
simd_stl_always_inline _DesiredType_ _SimdHorizontalMax(_VectorType_ _Vector) noexcept {
    _VerifyRegisterPolicy(_SimdGeneration_, _RegisterPolicy_);
    return _SimdArithmetic<_SimdGeneration_, _RegisterPolicy_>::template _HorizontalMax<_DesiredType_>(_Vector);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_,
    typename            _DesiredType_,
    typename            _VectorType_>
simd_stl_always_inline _DesiredType_ _SimdHorizontalMin(_VectorType_ _Vector) noexcept {
    _VerifyRegisterPolicy(_SimdGeneration_, _RegisterPolicy_);
    return _SimdArithmetic<_SimdGeneration_, _RegisterPolicy_>::template _HorizontalMin<_DesiredType_>(_Vector);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_,
    typename            _DesiredType_,
    typename            _VectorType_>
simd_stl_always_inline _VectorType_ _SimdVerticalMax(
    _VectorType_ _Left, 
    _VectorType_ _Right) noexcept
{
    _VerifyRegisterPolicy(_SimdGeneration_, _RegisterPolicy_);
    return _SimdArithmetic<_SimdGeneration_, _RegisterPolicy_>::template _VerticalMax<_DesiredType_>(_Left, _Right);
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

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_,
    typename            _DesiredType_,
    typename            _VectorType_,
    typename            _ReduceBinaryFunction_>
simd_stl_always_inline _DesiredType_ _SimdHorizontalFold(
    _VectorType_            _Vector, 
    _ReduceBinaryFunction_  _Reduce) noexcept
{
    _VerifyRegisterPolicy(_SimdGeneration_, _RegisterPolicy_);
    return _SimdArithmetic<_SimdGeneration_, _RegisterPolicy_>::template _HorizontalFold<_DesiredType_>(_Vector, type_traits::passFunction(_Reduce));
}

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_, 
    typename            _DesiredType_>
struct _VerticalMinWrapper {
    template <typename _VectorType_>
    simd_stl_always_inline _VectorType_ operator()(
        _VectorType_ _Left,
        _VectorType_ _Right) const noexcept
    {
        return _SimdVerticalMin<_SimdGeneration_, _RegisterPolicy_, _DesiredType_>(_Left, _Right);
    }
};

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_, 
    typename            _DesiredType_>
struct _VerticalMaxWrapper {
    template <typename _VectorType_>
    simd_stl_always_inline _VectorType_ operator()(
        _VectorType_ _Left,
        _VectorType_ _Right) const noexcept
    {
        return _SimdVerticalMax<_SimdGeneration_, _RegisterPolicy_, _DesiredType_>(_Left, _Right);
    }
};

__SIMD_STL_NUMERIC_NAMESPACE_END

#include <src/simd_stl/numeric/SimdArithmetic.inl>