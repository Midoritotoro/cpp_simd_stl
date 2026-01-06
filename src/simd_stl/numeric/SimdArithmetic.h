#pragma once 

#include <src/simd_stl/numeric/SimdElementAccess.h>
#include <src/simd_stl/numeric/SimdDivisors.h>

#include <src/simd_stl/numeric/SimdCompare.h>


__SIMD_STL_NUMERIC_NAMESPACE_BEGIN

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_>
class __simd_arithmetic;

#pragma region Sse2-Sse4.2 Simd arithmetic

template <>
class __simd_arithmetic<arch::CpuFeature::SSE2, xmm128> {
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
class __simd_arithmetic<arch::CpuFeature::SSE3, xmm128>:
    public __simd_arithmetic<arch::CpuFeature::SSE2, xmm128>
{};

template <>
class __simd_arithmetic<arch::CpuFeature::SSSE3, xmm128>:
    public __simd_arithmetic<arch::CpuFeature::SSE3, xmm128>
{
    static constexpr auto __generation   = arch::CpuFeature::SSSE3;
    using __register_policy               = numeric::xmm128;
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
class __simd_arithmetic<arch::CpuFeature::SSE41, xmm128>:
    public __simd_arithmetic<arch::CpuFeature::SSSE3, xmm128>
{
    static constexpr auto __generation  = arch::CpuFeature::SSE41;
    using __register_policy             = numeric::xmm128;
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
class __simd_arithmetic<arch::CpuFeature::SSE42, xmm128>:
    public __simd_arithmetic<arch::CpuFeature::SSE41, xmm128>
{};

#pragma endregion 

#pragma region Avx Simd arithmetic

template <>
class __simd_arithmetic<arch::CpuFeature::AVX, ymm256> {};

template <>
class __simd_arithmetic<arch::CpuFeature::AVX2, ymm256>:
    public __simd_arithmetic<arch::CpuFeature::AVX, ymm256>
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
class __simd_arithmetic<arch::CpuFeature::AVX512F, zmm512> {
    static constexpr auto __generation  = arch::CpuFeature::AVX512F;
    using __register_policy             = zmm512;
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
class __simd_arithmetic<arch::CpuFeature::AVX512BW, zmm512> :
    public __simd_arithmetic<arch::CpuFeature::AVX512F, zmm512>
{
    static constexpr auto __generation  = arch::CpuFeature::AVX512BW;
    using __register_policy             = zmm512;
public:
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

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline auto __reduce(_VectorType_ __vector) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ __abs(_VectorType_ __vector) noexcept;
};

template <>
class __simd_arithmetic<arch::CpuFeature::AVX512DQ, zmm512> :
    public __simd_arithmetic<arch::CpuFeature::AVX512F, zmm512>
{};

template <>
class __simd_arithmetic<arch::CpuFeature::AVX512VLF, ymm256>:
    public __simd_arithmetic<arch::CpuFeature::AVX2, ymm256>
{
    static constexpr auto __generation  = arch::CpuFeature::AVX512VLF;
    using __register_policy             = ymm256;
public:
    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ __vertical_min(
        _VectorType_ __left,
        _VectorType_ __right) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ __vertical_max(
        _VectorType_ __left,
        _VectorType_ __right) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ __abs(_VectorType_ __vector) noexcept;
};

template <>
class __simd_arithmetic<arch::CpuFeature::AVX512VLBW, ymm256> :
    public __simd_arithmetic<arch::CpuFeature::AVX512VLF, ymm256>
{};

template <>
class __simd_arithmetic<arch::CpuFeature::AVX512VLDQ, ymm256> :
    public __simd_arithmetic<arch::CpuFeature::AVX512VLF, ymm256>
{};

template <>
class __simd_arithmetic<arch::CpuFeature::AVX512VLBWDQ, ymm256>:
    public __simd_arithmetic<arch::CpuFeature::AVX512VLBW, ymm256>
{};

template <>
class __simd_arithmetic<arch::CpuFeature::AVX512VLF, xmm128> :
    public __simd_arithmetic<arch::CpuFeature::SSE42, xmm128>
{
    static constexpr auto __generation  = arch::CpuFeature::AVX512VLF;
    using __register_policy             = xmm128;
public:
    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ __vertical_min(
        _VectorType_ __left,
        _VectorType_ __right) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ __vertical_max(
        _VectorType_ __left,
        _VectorType_ __right) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ __abs(_VectorType_ __vector) noexcept;
};

template <>
class __simd_arithmetic<arch::CpuFeature::AVX512VLBW, xmm128> :
    public __simd_arithmetic<arch::CpuFeature::AVX512VLF, xmm128>
{
};

template <>
class __simd_arithmetic<arch::CpuFeature::AVX512VLDQ, xmm128> :
    public __simd_arithmetic<arch::CpuFeature::AVX512VLF, xmm128>
{
};


template <>
class __simd_arithmetic<arch::CpuFeature::AVX512VLBWDQ, xmm128>:
    public __simd_arithmetic<arch::CpuFeature::AVX512VLBW, xmm128>
{
};


#pragma endregion

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_,
    typename            _DesiredType_,
    typename            _VectorType_>
static simd_stl_always_inline _VectorType_ __simd_shift_right_elements(
    _VectorType_    __vector,
    uint32          __bit_shift) noexcept
{
    __verify_register_policy(_SimdGeneration_, _RegisterPolicy_);
    return __simd_arithmetic<_SimdGeneration_, _RegisterPolicy_>::template __shift_right_elements<_DesiredType_>(__vector, __bit_shift);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_,
    typename            _DesiredType_,
    typename            _VectorType_>
static simd_stl_always_inline _VectorType_ __simd_shift_left_elements(
    _VectorType_    __vector,
    uint32          __bit_shift) noexcept
{
    __verify_register_policy(_SimdGeneration_, _RegisterPolicy_);
    return __simd_arithmetic<_SimdGeneration_, _RegisterPolicy_>::template __shift_left_elements<_DesiredType_>(__vector, __bit_shift);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_,
    typename            _VectorType_>
simd_stl_always_inline _VectorType_ __simd_shift_right_vector(
    _VectorType_    __vector,
    uint32          __byte_shift) noexcept
{
    __verify_register_policy(_SimdGeneration_, _RegisterPolicy_);
    return __simd_arithmetic<_SimdGeneration_, _RegisterPolicy_>::template __shift_right_vector(__vector, __byte_shift);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_,
    typename            _VectorType_>
simd_stl_always_inline _VectorType_ __simd_shift_left_vector(
    _VectorType_    __vector,
    uint32          __byte_shift) noexcept
{
    __verify_register_policy(_SimdGeneration_, _RegisterPolicy_);
    return __simd_arithmetic<_SimdGeneration_, _RegisterPolicy_>::template __shift_left_vector(__vector, __byte_shift);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_,
    typename            _DesiredType_,
    typename            _VectorType_>
simd_stl_always_inline _VectorType_ __simd_negate(_VectorType_ __vector) noexcept {
    __verify_register_policy(_SimdGeneration_, _RegisterPolicy_);
    return __simd_arithmetic<_SimdGeneration_, _RegisterPolicy_>::template __negate<_DesiredType_>(__vector);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_,
    typename            _DesiredType_,
    typename            _VectorType_>
simd_stl_always_inline _VectorType_ __simd_add(
    _VectorType_ __left,
    _VectorType_ __right) noexcept
{
    __verify_register_policy(_SimdGeneration_, _RegisterPolicy_);
    return __simd_arithmetic<_SimdGeneration_, _RegisterPolicy_>::template _Add<_DesiredType_>(__left, __right);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_,
    typename            _DesiredType_,
    typename            _VectorType_>
simd_stl_always_inline _VectorType_ __simd_substract(
    _VectorType_ __left,
    _VectorType_ __right) noexcept
{
    __verify_register_policy(_SimdGeneration_, _RegisterPolicy_);
    return __simd_arithmetic<_SimdGeneration_, _RegisterPolicy_>::template _Substract<_DesiredType_>(__left, __right);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_,
    typename            _DesiredType_,
    typename            _VectorType_>
simd_stl_always_inline _VectorType_ __simd_multiply(
    _VectorType_ __left,
    _VectorType_ __right) noexcept
{
    __verify_register_policy(_SimdGeneration_, _RegisterPolicy_);
    return __simd_arithmetic<_SimdGeneration_, _RegisterPolicy_>::template _Multiply<_DesiredType_>(__left, __right);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_,
    typename            _DesiredType_,
    typename            _VectorType_>
simd_stl_always_inline _VectorType_ __simd_divide(
    _VectorType_ __left,
    _VectorType_ __right) noexcept
{
    __verify_register_policy(_SimdGeneration_, _RegisterPolicy_);
    return __simd_arithmetic<_SimdGeneration_, _RegisterPolicy_>::template _Divide<_DesiredType_>(__left, __right);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_,
    typename            _VectorType_>
simd_stl_always_inline _VectorType_ __simd_bit_not(_VectorType_ __vector) noexcept {
    __verify_register_policy(_SimdGeneration_, _RegisterPolicy_);
    return __simd_arithmetic<_SimdGeneration_, _RegisterPolicy_>::template __bit_not(__vector);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_,
    typename            _VectorType_>
simd_stl_always_inline _VectorType_ __simd_bit_xor(
    _VectorType_ __left,
    _VectorType_ __right) noexcept
{
    __verify_register_policy(_SimdGeneration_, _RegisterPolicy_);
    return __simd_arithmetic<_SimdGeneration_, _RegisterPolicy_>::template __bit_xor(__left, __right);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_,
    typename            _VectorType_>
simd_stl_always_inline _VectorType_ __simd_bit_and(
    _VectorType_ __left,
    _VectorType_ __right) noexcept
{
    __verify_register_policy(_SimdGeneration_, _RegisterPolicy_);
    return __simd_arithmetic<_SimdGeneration_, _RegisterPolicy_>::template __bit_and(__left, __right);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_,
    typename            _VectorType_>
simd_stl_always_inline _VectorType_ __simd_bit_or(
    _VectorType_ __left,
    _VectorType_ __right) noexcept
{
    __verify_register_policy(_SimdGeneration_, _RegisterPolicy_);
    return __simd_arithmetic<_SimdGeneration_, _RegisterPolicy_>::template __bit_or(__left, __right);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_,
    typename            _DesiredType_,
    typename            _VectorType_>
simd_stl_always_inline auto __simd_reduce(_VectorType_ __vector) noexcept {
    __verify_register_policy(_SimdGeneration_, _RegisterPolicy_);
    return __simd_arithmetic<_SimdGeneration_, _RegisterPolicy_>::template __reduce<_DesiredType_>(__vector);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_,
    typename            _DesiredType_,
    typename            _VectorType_>
simd_stl_always_inline _VectorType_ __simd_vertical_min(
    _VectorType_ __left,
    _VectorType_ __right) noexcept
{
    __verify_register_policy(_SimdGeneration_, _RegisterPolicy_);
    return __simd_arithmetic<_SimdGeneration_, _RegisterPolicy_>::template __vertical_min<_DesiredType_>(__left, __right);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_,
    typename            _DesiredType_,
    typename            _VectorType_>
simd_stl_always_inline _DesiredType_ __simd_horizontal_max(_VectorType_ __vector) noexcept {
    __verify_register_policy(_SimdGeneration_, _RegisterPolicy_);
    return __simd_arithmetic<_SimdGeneration_, _RegisterPolicy_>::template __horizontal_max<_DesiredType_>(__vector);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_,
    typename            _DesiredType_,
    typename            _VectorType_>
simd_stl_always_inline _DesiredType_ __simd_horizontal_min(_VectorType_ __vector) noexcept {
    __verify_register_policy(_SimdGeneration_, _RegisterPolicy_);
    return __simd_arithmetic<_SimdGeneration_, _RegisterPolicy_>::template __horizontal_min<_DesiredType_>(__vector);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_,
    typename            _DesiredType_,
    typename            _VectorType_>
simd_stl_always_inline _VectorType_ __simd_vertical_max(
    _VectorType_ __left, 
    _VectorType_ __right) noexcept
{
    __verify_register_policy(_SimdGeneration_, _RegisterPolicy_);
    return __simd_arithmetic<_SimdGeneration_, _RegisterPolicy_>::template __vertical_max<_DesiredType_>(__left, __right);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_,
    typename            _DesiredType_,
    typename            _VectorType_>
simd_stl_always_inline _VectorType_ __simd_abs(_VectorType_ __vector) noexcept {
    __verify_register_policy(_SimdGeneration_, _RegisterPolicy_);
    return __simd_arithmetic<_SimdGeneration_, _RegisterPolicy_>::template __abs<_DesiredType_>(__vector);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_,
    typename            _DesiredType_,
    typename            _VectorType_,
    typename            _ReduceBinaryFunction_>
simd_stl_always_inline _DesiredType_ __simd_horizontal_fold(
    _VectorType_            __vector, 
    _ReduceBinaryFunction_  __reduce) noexcept
{
    __verify_register_policy(_SimdGeneration_, _RegisterPolicy_);
    return __simd_arithmetic<_SimdGeneration_, _RegisterPolicy_>::template __horizontal_fold<_DesiredType_>(__vector, type_traits::__pass_function(__reduce));
}

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_, 
    typename            _DesiredType_>
struct __vertical_min_wrapper {
    template <typename _VectorType_>
    simd_stl_always_inline _VectorType_ operator()(
        _VectorType_ __left,
        _VectorType_ __right) const noexcept
    {
        return __simd_vertical_min<_SimdGeneration_, _RegisterPolicy_, _DesiredType_>(__left, __right);
    }
};

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_, 
    typename            _DesiredType_>
struct __vertical_max_wrapper {
    template <typename _VectorType_>
    simd_stl_always_inline _VectorType_ operator()(
        _VectorType_ __left,
        _VectorType_ __right) const noexcept
    {
        return __simd_vertical_max<_SimdGeneration_, _RegisterPolicy_, _DesiredType_>(__left, __right);
    }
};

__SIMD_STL_NUMERIC_NAMESPACE_END

#include <src/simd_stl/numeric/SimdArithmetic.inl>