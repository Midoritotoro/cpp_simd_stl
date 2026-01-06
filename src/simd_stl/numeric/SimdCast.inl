#pragma once

__SIMD_STL_NUMERIC_NAMESPACE_BEGIN

template <
    class _RebindType_,
    class _VectorType_,
    bool _IsBasicSimd_,
    bool _IsIntrin_>
struct __rebind_vector_element_t {
    using type = void;
};

template <
    class _RebindType_,
    class _VectorType_>
struct __rebind_vector_element_t<_RebindType_, _VectorType_, false, true> {
    using type = std::conditional_t<__is_intrin_type_v<_RebindType_> || __is_valid_basic_simd_v<_RebindType_>, _RebindType_,
        std::conditional_t<sizeof(_VectorType_) == __zmm_width,
        type_traits::__deduce_simd_vector_type<arch::CpuFeature::AVX512F, _RebindType_>,
        std::conditional_t<sizeof(_VectorType_) == __ymm_width,
        type_traits::__deduce_simd_vector_type<arch::CpuFeature::AVX, _RebindType_>,
        std::conditional_t<sizeof(_VectorType_) == __xmm_width,
        type_traits::__deduce_simd_vector_type<arch::CpuFeature::SSE2, _RebindType_>, void>>>>;
};

template <
    class _RebindType_,
    class _VectorType_>
struct __rebind_vector_element_t<_RebindType_, _VectorType_, true, false> {
    using type = std::conditional_t<__is_intrin_type_v<_RebindType_> || __is_valid_basic_simd_v<_RebindType_>,
        _RebindType_, simd<_VectorType_::__generation, _RebindType_, typename _VectorType_::policy_type>>;
};

template <
    arch::CpuFeature	_ToSimdGeneration_,
    class               _RebindType_,
    class               _VectorType_,
    bool                _IsBasicSimd_,
    bool                _IsIntrin_>
struct __rebind_vector_generation_t {
    using type = void;
};

template <
    arch::CpuFeature	_ToSimdGeneration_,
    class               _RebindType_,
    class               _VectorType_>
struct __rebind_vector_generation_t<_ToSimdGeneration_, _RebindType_, _VectorType_, false, true> {
    using type = type_traits::__deduce_simd_vector_type<_ToSimdGeneration_, _RebindType_>;
};

template <
    arch::CpuFeature	_ToSimdGeneration_,
    class               _RebindType_,
    class               _VectorType_>
struct __rebind_vector_generation_t<_ToSimdGeneration_, _RebindType_, _VectorType_, true, false> {
    using type = simd<_ToSimdGeneration_, _RebindType_, __default_register_policy<_ToSimdGeneration_>>;
};

template <
    class _RebindType_,
    class _VectorType_>
using __rebind_vector_element_type = typename __rebind_vector_element_t<_RebindType_, _VectorType_>::type;

template <
    arch::CpuFeature	_ToSimdGeneration_,
    class               _RebindType_,
    class               _VectorType_>
using __rebind_vector_generation_type = typename __rebind_vector_generation_t<_ToSimdGeneration_, _RebindType_, _VectorType_>::type;

template <class _VectorType_>
__simd_nodiscard_inline __unwrapped_vector_type<_VectorType_> __simd_unwrap(_VectorType_ __vector) noexcept {
    if constexpr (__is_valid_basic_simd_v<_VectorType_>)
        return __vector.unwrap();
    else
        return __vector;
}

template <class _MaskType_, std::enable_if_t<__is_valid_basic_simd_v<_MaskType_> || __is_intrin_type_v<_MaskType_> || std::is_integral_v<_MaskType_>, int>>
__simd_nodiscard_inline auto __simd_unwrap_mask(_MaskType_ __mask) noexcept {
    if constexpr (std::is_integral_v<_MaskType_>)
        return __mask;
    else
        return __simd_unwrap(__mask);
}

template <
    class _ToType_,
    class _FromType_,
    std::enable_if_t<(__is_valid_basic_simd_v<_ToType_> || __is_intrin_type_v<_ToType_> ||
        type_traits::__is_vector_type_supported_v<_ToType_>) &&
    (__is_valid_basic_simd_v<_FromType_> || __is_intrin_type_v<_FromType_>), int>>
__simd_nodiscard_inline __rebind_vector_element_type<_ToType_, _FromType_> simd_cast(_FromType_ __from) noexcept {
    return __intrin_bitcast<__unwrapped_vector_type<__rebind_vector_element_type<_ToType_, _FromType_>>>(__simd_unwrap(__from));
}

template <
    arch::CpuFeature	_ToSimdGeneration_,
    class               _FromVector_,
    std::enable_if_t<__is_valid_basic_simd_v<_FromVector_> || __is_intrin_type_v<_FromVector_>, int>>
__simd_nodiscard_inline __rebind_vector_generation_type<_ToSimdGeneration_,
    __vector_element_type<_FromVector_>, _FromVector_> simd_cast(_FromVector_ __from) noexcept
{
    return __intrin_bitcast<__unwrapped_vector_type<__rebind_vector_generation_type<_ToSimdGeneration_,
        __vector_element_type<_FromVector_>, _FromVector_>>>(__simd_unwrap(__from));
}

template <
    arch::CpuFeature	_ToSimdGeneration_,
    class               _ToElementType_,
    class               _FromVector_,
    std::enable_if_t<__is_valid_basic_simd_v<_FromVector_> || __is_intrin_type_v<_FromVector_>, int>>
__simd_nodiscard_inline __rebind_vector_generation_type<_ToSimdGeneration_,
    _ToElementType_, _FromVector_> simd_cast(_FromVector_ __from) noexcept 
{
    return __intrin_bitcast<__unwrapped_vector_type<__rebind_vector_generation_type<_ToSimdGeneration_,
        _ToElementType_, _FromVector_>>>(__simd_unwrap(__from));
}

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_,
    class               _Type_,
    class               _MaskType_>
__simd_nodiscard_inline __make_tail_mask_return_type<simd<_SimdGeneration_, _Type_,
    _RegisterPolicy_>> __simd_convert_to_mask_for_native_store(_MaskType_ __mask) noexcept
{
    using _ConvertTo = __make_tail_mask_return_type<simd<_SimdGeneration_, _Type_, _RegisterPolicy_>>;

    constexpr auto __from_simd      = __is_valid_basic_simd_v<_MaskType_>;
    constexpr auto __to_simd        = __is_valid_basic_simd_v<_ConvertTo>;

    constexpr auto __from_integral  = std::is_integral_v<_MaskType_>;
    constexpr auto __to_integral    = std::is_integral_v<_ConvertTo>;

    constexpr auto __both_simd      = __from_simd && __to_simd;
    constexpr auto __both_integral  = __from_integral && __to_integral;

    if constexpr (__both_simd)
        return _ConvertTo { __simd_unwrap(__mask) };

    else if constexpr (__both_integral)
        return static_cast<_ConvertTo>(__mask);

    else if constexpr (__from_integral && __to_simd)
        return _ConvertTo { __simd_to_vector<_SimdGeneration_, _RegisterPolicy_,
            typename _ConvertTo::vector_type, _Type_>(__mask)
        };

    else if constexpr (__from_simd && __to_integral)
        return __simd_to_mask<_SimdGeneration_, _RegisterPolicy_, _Type_>(__mask);

    else
        static_assert(false, "Invalid _ConvertTo");
}

__SIMD_STL_NUMERIC_NAMESPACE_END
