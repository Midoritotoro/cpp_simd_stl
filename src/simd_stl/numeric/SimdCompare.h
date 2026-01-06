#pragma once 

#include <src/simd_stl/numeric/IntrinBitcast.h>
#include <src/simd_stl/numeric/SimdConvert.h>

#include <src/simd_stl/type_traits/OperatorWrappers.h>
#include <src/simd_stl/numeric/SimdComparison.h>


__SIMD_STL_NUMERIC_NAMESPACE_BEGIN

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_,
    typename            _VectorType_>
simd_stl_always_inline _VectorType_ __simd_bit_not(_VectorType_ __vector) noexcept;

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_>
class __simd_compare_implementation;

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_,
    class               _DesiredType_,
    __simd_comparison   _CompareType_,
    class               _VectorType_>
simd_stl_nodiscard simd_stl_always_inline _VectorType_ __simd_compare(
    _VectorType_ __left,
    _VectorType_ __right) noexcept;

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_,
    class               _DesiredType_,
    __simd_comparison   _CompareType_,
    class               _VectorType_>
simd_stl_nodiscard simd_stl_always_inline type_traits::__deduce_simd_mask_type<_SimdGeneration_,
    _DesiredType_, _RegisterPolicy_> __simd_mask_compare(
        _VectorType_ __left,
        _VectorType_ __right) noexcept;

#pragma region Sse2-Sse4.2 Simd compare

template <>
class __simd_compare_implementation<arch::CpuFeature::SSE2, xmm128> {
    static constexpr auto __generation  = arch::CpuFeature::SSE2;
    using __register_policy             = xmm128;

    template <class _DesiredType_>
    using __simd_mask_type = type_traits::__deduce_simd_mask_type<__generation, _DesiredType_, __register_policy>;
public:
    template <
        class               _DesiredType_,
        __simd_comparison   _CompareType_,
        class               _VectorType_>
    static simd_stl_always_inline __simd_mask_type<_DesiredType_> __mask_compare(
        _VectorType_ __left,
        _VectorType_ __right) noexcept;

    template <
        typename            _DesiredType_,
        __simd_comparison   _CompareType_,
        typename            _VectorType_>
    static simd_stl_always_inline _VectorType_ __compare(
        _VectorType_ __left,
        _VectorType_ __right) noexcept;

    template <
        typename            _DesiredType_,
        __simd_comparison   _CompareType_,
        typename            _VectorType_>
    static simd_stl_always_inline auto __native_compare(
        _VectorType_ __left,
        _VectorType_ __right) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ __compare_equal(
        _VectorType_ __left,
        _VectorType_ __right) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ __compare_less(
        _VectorType_ __left,
        _VectorType_ __right) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ __compare_greater(
        _VectorType_ __left,
        _VectorType_ __right) noexcept;
};

template <>
class __simd_compare_implementation<arch::CpuFeature::SSE3, xmm128> :
    public __simd_compare_implementation<arch::CpuFeature::SSE2, xmm128>
{};

template <>
class __simd_compare_implementation<arch::CpuFeature::SSSE3, xmm128> :
    public __simd_compare_implementation<arch::CpuFeature::SSE3, xmm128>
{
    static constexpr auto __generation   = arch::CpuFeature::SSSE3;
    using __register_policy               = xmm128;

    template <class _DesiredType_>
    using __simd_mask_type = type_traits::__deduce_simd_mask_type<__generation, _DesiredType_, __register_policy>;
public:
    template <
        class               _DesiredType_,
        __simd_comparison   _CompareType_,
        class               _VectorType_>
    static simd_stl_always_inline __simd_mask_type<_DesiredType_> __mask_compare(
        _VectorType_ __left,
        _VectorType_ __right) noexcept;
};

template <>
class __simd_compare_implementation<arch::CpuFeature::SSE41, xmm128> :
    public __simd_compare_implementation<arch::CpuFeature::SSSE3, xmm128>
{
    static constexpr auto __generation   = arch::CpuFeature::SSE41;
    using __register_policy               = xmm128;

    template <class _DesiredType_>
    using __simd_mask_type = type_traits::__deduce_simd_mask_type<__generation, _DesiredType_, __register_policy>;
public:
    template <
        class               _DesiredType_,
        __simd_comparison   _CompareType_,
        class               _VectorType_>
    static simd_stl_always_inline __simd_mask_type<_DesiredType_> __mask_compare(
        _VectorType_ __left,
        _VectorType_ __right) noexcept;

    template <
        typename            _DesiredType_,
        __simd_comparison   _CompareType_,
        typename            _VectorType_>
    static simd_stl_always_inline _VectorType_ __compare(
        _VectorType_ __left,
        _VectorType_ __right) noexcept;

    template <
        typename            _DesiredType_,
        __simd_comparison   _CompareType_,
        typename            _VectorType_>
    static simd_stl_always_inline auto __native_compare(
        _VectorType_ __left,
        _VectorType_ __right) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ __compare_equal(
        _VectorType_ __left,
        _VectorType_ __right) noexcept;
};

template <>
class __simd_compare_implementation<arch::CpuFeature::SSE42, xmm128>:
    public __simd_compare_implementation<arch::CpuFeature::SSE41, xmm128>
{
    static constexpr auto __generation   = arch::CpuFeature::SSE42;
    using __register_policy               = xmm128;

    template <class _DesiredType_>
    using __simd_mask_type = type_traits::__deduce_simd_mask_type<__generation, _DesiredType_, __register_policy>;
public:
    template <
        class               _DesiredType_,
        __simd_comparison   _CompareType_,
        class               _VectorType_>
    static simd_stl_always_inline __simd_mask_type<_DesiredType_> __mask_compare(
        _VectorType_ __left,
        _VectorType_ __right) noexcept;

    template <
        typename            _DesiredType_,
        __simd_comparison   _CompareType_,
        typename            _VectorType_>
    static simd_stl_always_inline _VectorType_ __compare(
        _VectorType_ __left,
        _VectorType_ __right) noexcept;

    template <
        typename            _DesiredType_,
        __simd_comparison   _CompareType_,
        typename            _VectorType_>
    static simd_stl_always_inline auto __native_compare(
        _VectorType_ __left,
        _VectorType_ __right) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ __compare_greater(
        _VectorType_ __left,
        _VectorType_ __right) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ __compare_less(
        _VectorType_ __left,
        _VectorType_ __right) noexcept;
};

#pragma endregion

#pragma region Avx Simd compare

template <>
class __simd_compare_implementation<arch::CpuFeature::AVX, ymm256>
{};

template <>
class __simd_compare_implementation<arch::CpuFeature::AVX2, ymm256>:
    public __simd_compare_implementation<arch::CpuFeature::AVX, ymm256>
{
    static constexpr auto __generation   = arch::CpuFeature::AVX2;
    using __register_policy               = ymm256;

    template <class _DesiredType_>
    using __simd_mask_type = type_traits::__deduce_simd_mask_type<__generation, _DesiredType_, __register_policy>;
public:
    template <
        class               _DesiredType_,
        __simd_comparison   _CompareType_,
        class               _VectorType_>
    static simd_stl_always_inline __simd_mask_type<_DesiredType_> __mask_compare(
        _VectorType_ __left,
        _VectorType_ __right) noexcept;

    template <
        typename            _DesiredType_,
        __simd_comparison   _CompareType_,
        typename            _VectorType_>
    static simd_stl_always_inline _VectorType_ __compare(
        _VectorType_ __left,
        _VectorType_ __right) noexcept;

    template <
        typename            _DesiredType_,
        __simd_comparison   _CompareType_,
        typename            _VectorType_>
    static simd_stl_always_inline auto __native_compare(
        _VectorType_ __left,
        _VectorType_ __right) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ __compare_equal(
        _VectorType_ __left,
        _VectorType_ __right) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ __compare_less(
        _VectorType_ __left,
        _VectorType_ __right) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ __compare_greater(
        _VectorType_ __left,
        _VectorType_ __right) noexcept;
};

#pragma endregion

#pragma region Avx512 Simd compare

template <>
class __simd_compare_implementation<arch::CpuFeature::AVX512F, zmm512> {
    static constexpr auto __generation = arch::CpuFeature::AVX512F;
    using __register_policy = zmm512;

    template <class _DesiredType_>
    using __simd_mask_type = type_traits::__deduce_simd_mask_type<__generation, _DesiredType_, __register_policy>;

    template <
        class               _DesiredType_,
        __simd_comparison   _CompareType_,
        class               _VectorType_>
    static simd_stl_always_inline _VectorType_ __blockwise_compare(
        _VectorType_ __left,
        _VectorType_ __right) noexcept;
public:
    template <
        class               _DesiredType_,
        __simd_comparison   _CompareType_,
        class               _VectorType_>
    static simd_stl_always_inline __simd_mask_type<_DesiredType_> __mask_compare(
        _VectorType_ __left,
        _VectorType_ __right) noexcept;

    template <
        typename            _DesiredType_,
        __simd_comparison   _CompareType_,
        typename            _VectorType_>
    static simd_stl_always_inline auto __native_compare(
        _VectorType_ __left,
        _VectorType_ __right) noexcept;

    template <
        typename            _DesiredType_,
        __simd_comparison   _CompareType_,
        typename            _VectorType_>
    static simd_stl_always_inline _VectorType_ __compare(
        _VectorType_ __left,
        _VectorType_ __right) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline __simd_mask_type<_DesiredType_> __mask_compare_equal(
        _VectorType_ __left,
        _VectorType_ __right) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline __simd_mask_type<_DesiredType_> __mask_compare_less(
        _VectorType_ __left,
        _VectorType_ __right) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline __simd_mask_type<_DesiredType_> __mask_compare_greater(
        _VectorType_ __left,
        _VectorType_ __right) noexcept;
};

template <>
class __simd_compare_implementation<arch::CpuFeature::AVX512BW, zmm512>:
    public __simd_compare_implementation<arch::CpuFeature::AVX512F, zmm512> 
{
    static constexpr auto __generation   = arch::CpuFeature::AVX512BW;
    using __register_policy               = zmm512;

    template <class _DesiredType_>
    using __simd_mask_type = type_traits::__deduce_simd_mask_type<__generation, _DesiredType_, __register_policy>;
public:
    template <
        class               _DesiredType_,
        __simd_comparison   _CompareType_,
        class               _VectorType_>
    static simd_stl_always_inline __simd_mask_type<_DesiredType_> __mask_compare(
        _VectorType_ __left,
        _VectorType_ __right) noexcept;

    template <
        typename            _DesiredType_,
        __simd_comparison   _CompareType_,
        typename            _VectorType_>
    static simd_stl_always_inline _VectorType_ __compare(
        _VectorType_ __left,
        _VectorType_ __right) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline __simd_mask_type<_DesiredType_> __mask_compare_equal(
        _VectorType_ __left,
        _VectorType_ __right) noexcept;

    template <
        typename            _DesiredType_,
        __simd_comparison   _CompareType_,
        typename            _VectorType_>
    static simd_stl_always_inline auto __native_compare(
        _VectorType_ __left,
        _VectorType_ __right) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline __simd_mask_type<_DesiredType_> __mask_compare_less(
        _VectorType_ __left,
        _VectorType_ __right) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline __simd_mask_type<_DesiredType_> __mask_compare_greater(
        _VectorType_ __left,
        _VectorType_ __right) noexcept;
};

template <>
class __simd_compare_implementation<arch::CpuFeature::AVX512DQ, zmm512> :
    public __simd_compare_implementation<arch::CpuFeature::AVX512BW, zmm512>
{};

template <>
class __simd_compare_implementation<arch::CpuFeature::AVX512VLF, ymm256> :
	public __simd_compare_implementation<arch::CpuFeature::AVX2, ymm256>
{
    static constexpr auto __generation = arch::CpuFeature::AVX512VLF;
    using __register_policy = ymm256;

    template <class _DesiredType_>
    using __simd_mask_type = type_traits::__deduce_simd_mask_type<__generation, _DesiredType_, __register_policy>;
public:
    template <
        class               _DesiredType_,
        __simd_comparison   _CompareType_,
        class               _VectorType_>
    static simd_stl_always_inline __simd_mask_type<_DesiredType_> __mask_compare(
        _VectorType_ __left,
        _VectorType_ __right) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline __simd_mask_type<_DesiredType_> __mask_compare_equal(
        _VectorType_ __left,
        _VectorType_ __right) noexcept;

    template <
        typename            _DesiredType_,
        __simd_comparison   _CompareType_,
        typename            _VectorType_>
    static simd_stl_always_inline auto __native_compare(
        _VectorType_ __left,
        _VectorType_ __right) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline __simd_mask_type<_DesiredType_> __mask_compare_less(
        _VectorType_ __left,
        _VectorType_ __right) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline __simd_mask_type<_DesiredType_> __mask_compare_greater(
        _VectorType_ __left,
        _VectorType_ __right) noexcept;
};

template <>
class __simd_compare_implementation<arch::CpuFeature::AVX512VLBW, ymm256> :
	public __simd_compare_implementation<arch::CpuFeature::AVX512VLF, ymm256>
{
    static constexpr auto __generation = arch::CpuFeature::AVX512VLBW;
    using __register_policy = ymm256;

    template <class _DesiredType_>
    using __simd_mask_type = type_traits::__deduce_simd_mask_type<__generation, _DesiredType_, __register_policy>;
public:
    template <
        class               _DesiredType_,
        __simd_comparison   _CompareType_,
        class               _VectorType_>
    static simd_stl_always_inline __simd_mask_type<_DesiredType_> __mask_compare(
        _VectorType_ __left,
        _VectorType_ __right) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline __simd_mask_type<_DesiredType_> __mask_compare_equal(
        _VectorType_ __left,
        _VectorType_ __right) noexcept;

    template <
        typename            _DesiredType_,
        __simd_comparison   _CompareType_,
        typename            _VectorType_>
    static simd_stl_always_inline auto __native_compare(
        _VectorType_ __left,
        _VectorType_ __right) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline __simd_mask_type<_DesiredType_> __mask_compare_less(
        _VectorType_ __left,
        _VectorType_ __right) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline __simd_mask_type<_DesiredType_> __mask_compare_greater(
        _VectorType_ __left,
        _VectorType_ __right) noexcept;
};

template <>
class __simd_compare_implementation<arch::CpuFeature::AVX512VLDQ, ymm256> :
	public __simd_compare_implementation<arch::CpuFeature::AVX512VLF, ymm256>
{};

template <>
class __simd_compare_implementation<arch::CpuFeature::AVX512VLBWDQ, ymm256> :
	public __simd_compare_implementation<arch::CpuFeature::AVX512VLBW, ymm256>
{};

template <>
class __simd_compare_implementation<arch::CpuFeature::AVX512VLF, xmm128> :
	public __simd_compare_implementation<arch::CpuFeature::SSE42, xmm128>
{
    static constexpr auto __generation = arch::CpuFeature::AVX512VLF;
    using __register_policy = xmm128;

    template <class _DesiredType_>
    using __simd_mask_type = type_traits::__deduce_simd_mask_type<__generation, _DesiredType_, __register_policy>;
public:
    template <
        class               _DesiredType_,
        __simd_comparison   _CompareType_,
        class               _VectorType_>
    static simd_stl_always_inline __simd_mask_type<_DesiredType_> __mask_compare(
        _VectorType_ __left,
        _VectorType_ __right) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline __simd_mask_type<_DesiredType_> __mask_compare_equal(
        _VectorType_ __left,
        _VectorType_ __right) noexcept;

    template <
        typename            _DesiredType_,
        __simd_comparison   _CompareType_,
        typename            _VectorType_>
    static simd_stl_always_inline auto __native_compare(
        _VectorType_ __left,
        _VectorType_ __right) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline __simd_mask_type<_DesiredType_> __mask_compare_less(
        _VectorType_ __left,
        _VectorType_ __right) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline __simd_mask_type<_DesiredType_> __mask_compare_greater(
        _VectorType_ __left,
        _VectorType_ __right) noexcept;
};

template <>
class __simd_compare_implementation<arch::CpuFeature::AVX512VLBW, xmm128> :
	public __simd_compare_implementation<arch::CpuFeature::AVX512VLF, xmm128>
{
    static constexpr auto __generation = arch::CpuFeature::AVX512VLBW;
    using __register_policy = xmm128;

    template <class _DesiredType_>
    using __simd_mask_type = type_traits::__deduce_simd_mask_type<__generation, _DesiredType_, __register_policy>;
public:
    template <
        class               _DesiredType_,
        __simd_comparison   _CompareType_,
        class               _VectorType_>
    static simd_stl_always_inline __simd_mask_type<_DesiredType_> __mask_compare(
        _VectorType_ __left,
        _VectorType_ __right) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline __simd_mask_type<_DesiredType_> __mask_compare_equal(
        _VectorType_ __left,
        _VectorType_ __right) noexcept;

    template <
        typename            _DesiredType_,
        __simd_comparison   _CompareType_,
        typename            _VectorType_>
    static simd_stl_always_inline auto __native_compare(
        _VectorType_ __left,
        _VectorType_ __right) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline __simd_mask_type<_DesiredType_> __mask_compare_less(
        _VectorType_ __left,
        _VectorType_ __right) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline __simd_mask_type<_DesiredType_> __mask_compare_greater(
        _VectorType_ __left,
        _VectorType_ __right) noexcept;
};

template <>
class __simd_compare_implementation<arch::CpuFeature::AVX512VLDQ, xmm128> :
	public __simd_compare_implementation<arch::CpuFeature::AVX512VLF, xmm128>
{};

template <>
class __simd_compare_implementation<arch::CpuFeature::AVX512VLBWDQ, xmm128> :
	public __simd_compare_implementation<arch::CpuFeature::AVX512VLBW, xmm128>
{};

#pragma endregion 

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_,
    class               _DesiredType_,
    __simd_comparison   _CompareType_,
    class               _VectorType_>
simd_stl_nodiscard simd_stl_always_inline _VectorType_ __simd_compare(
    _VectorType_ __left,
    _VectorType_ __right) noexcept
{
    __verify_register_policy(_SimdGeneration_, _RegisterPolicy_);
    return __simd_compare_implementation<_SimdGeneration_, _RegisterPolicy_>
        ::template __compare<_DesiredType_, _CompareType_>(__left, __right);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_,
    class               _DesiredType_,
    __simd_comparison   _CompareType_,
    class               _VectorType_>
simd_stl_nodiscard simd_stl_always_inline type_traits::__deduce_simd_mask_type<_SimdGeneration_,
    _DesiredType_, _RegisterPolicy_> __simd_mask_compare(
        _VectorType_ __left,
        _VectorType_ __right) noexcept
{
    __verify_register_policy(_SimdGeneration_, _RegisterPolicy_);
    return __simd_compare_implementation<_SimdGeneration_, _RegisterPolicy_>::template
        __mask_compare<_DesiredType_, _CompareType_>(__left, __right);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_,
    class               _DesiredType_,
    __simd_comparison   _CompareType_,
    class               _VectorType_>
simd_stl_nodiscard simd_stl_always_inline auto __simd_native_compare(
    _VectorType_ __left,
    _VectorType_ __right) noexcept
{
    __verify_register_policy(_SimdGeneration_, _RegisterPolicy_);
    return __simd_compare_implementation<_SimdGeneration_, _RegisterPolicy_>
        ::template __native_compare<_DesiredType_, _CompareType_>(__left, __right);
}

template <
    class               _BasicSimd_, 
    typename            _DesiredType_,
    __simd_comparison   _CompareType_>
using __native_compare_return_type = __native_compare_return_type_helper<_BasicSimd_,
    type_traits::invoke_result_type<decltype(__simd_native_compare<_BasicSimd_::__generation,
        typename _BasicSimd_::policy_type, _DesiredType_, _CompareType_, typename _BasicSimd_::vector_type>),
    typename _BasicSimd_::vector_type, typename _BasicSimd_::vector_type>, _DesiredType_>;

__SIMD_STL_NUMERIC_NAMESPACE_END

#include <src/simd_stl/numeric/SimdCompare.inl>
