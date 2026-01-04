#pragma once

__SIMD_STL_NUMERIC_NAMESPACE_BEGIN

template <
    class _RebindType_,
    class _VectorType_,
    bool _IsBasicSimd_,
    bool _IsIntrin_>
struct _Rebind_vector_element_t {
    using type = void;
};

template <
    class _RebindType_,
    class _VectorType_>
struct _Rebind_vector_element_t<_RebindType_, _VectorType_, false, true> {
    using type = std::conditional_t<_Is_intrin_type_v<_RebindType_> || _Is_valid_basic_simd_v<_RebindType_>, _RebindType_,
        std::conditional_t<sizeof(_VectorType_) == _ZmmWidth,
        type_traits::__deduce_simd_vector_type<arch::CpuFeature::AVX512F, _RebindType_>,
        std::conditional_t<sizeof(_VectorType_) == _YmmWidth,
        type_traits::__deduce_simd_vector_type<arch::CpuFeature::AVX, _RebindType_>,
        std::conditional_t<sizeof(_VectorType_) == _XmmWidth,
        type_traits::__deduce_simd_vector_type<arch::CpuFeature::SSE2, _RebindType_>, void>>>>;
};

template <
    class _RebindType_,
    class _VectorType_>
struct _Rebind_vector_element_t<_RebindType_, _VectorType_, true, false> {
    using type = std::conditional_t<_Is_intrin_type_v<_RebindType_> || _Is_valid_basic_simd_v<_RebindType_>,
        _RebindType_, simd<_VectorType_::_Generation, _RebindType_, typename _VectorType_::policy_type>>;
};

template <
    arch::CpuFeature	_ToSimdGeneration_,
    class               _RebindType_,
    class               _VectorType_,
    bool                _IsBasicSimd_,
    bool                _IsIntrin_>
struct _Rebind_vector_generation_t {
    using type = void;
};

template <
    arch::CpuFeature	_ToSimdGeneration_,
    class               _RebindType_,
    class               _VectorType_>
struct _Rebind_vector_generation_t<_ToSimdGeneration_, _RebindType_, _VectorType_, false, true> {
    using type = type_traits::__deduce_simd_vector_type<_ToSimdGeneration_, _RebindType_>;
};

template <
    arch::CpuFeature	_ToSimdGeneration_,
    class               _RebindType_,
    class               _VectorType_>
struct _Rebind_vector_generation_t<_ToSimdGeneration_, _RebindType_, _VectorType_, true, false> {
    using type = simd<_ToSimdGeneration_, _RebindType_, _DefaultRegisterPolicy<_ToSimdGeneration_>>;
};

template <
    class _RebindType_,
    class _VectorType_>
using _Rebind_vector_element_type = typename _Rebind_vector_element_t<_RebindType_, _VectorType_>::type;

template <
    arch::CpuFeature	_ToSimdGeneration_,
    class               _RebindType_,
    class               _VectorType_>
using _Rebind_vector_generation_type = typename _Rebind_vector_generation_t<_ToSimdGeneration_, _RebindType_, _VectorType_>::type;

template <class _VectorType_, std::enable_if_t<_Is_valid_basic_simd_v<_VectorType_> || _Is_intrin_type_v<_VectorType_>, int>>
simd_stl_nodiscard simd_stl_always_inline _Unwrapped_vector_type<_VectorType_> _SimdUnwrap(_VectorType_ _Vector) noexcept {
    if constexpr (_Is_valid_basic_simd_v<_VectorType_>)
        return _Vector.unwrap();
    else
        return _Vector;
}

template <class _MaskType_, std::enable_if_t<_Is_valid_basic_simd_v<_MaskType_> || _Is_intrin_type_v<_MaskType_> || std::is_integral_v<_MaskType_>, int>>
__simd_nodiscard_inline auto _SimdUnwrapMask(_MaskType_ _Mask) noexcept {
    if constexpr (std::is_integral_v<_MaskType_>)
        return _Mask;
    else
        return _SimdUnwrap(_Mask);
}

template <
    class _ToType_,
    class _FromType_,
    std::enable_if_t<(_Is_valid_basic_simd_v<_ToType_> || _Is_intrin_type_v<_ToType_> ||
        type_traits::__is_vector_type_supported_v<_ToType_>) &&
    (_Is_valid_basic_simd_v<_FromType_> || _Is_intrin_type_v<_FromType_>), int>>
simd_stl_nodiscard simd_stl_always_inline _Rebind_vector_element_type<_ToType_, _FromType_> simd_cast(_FromType_ _From) noexcept {
    return __intrin_bitcast<_Unwrapped_vector_type<_Rebind_vector_element_type<_ToType_, _FromType_>>>(_SimdUnwrap(_From));
}

template <
    arch::CpuFeature	_ToSimdGeneration_,
    class               _FromVector_,
    std::enable_if_t<_Is_valid_basic_simd_v<_FromVector_>
    || _Is_intrin_type_v<_FromVector_>, int>>
simd_stl_nodiscard simd_stl_always_inline _Rebind_vector_generation_type<_ToSimdGeneration_,
    _Vector_element_type<_FromVector_>, _FromVector_> simd_cast(_FromVector_ _From) noexcept
{
    return __intrin_bitcast<_Unwrapped_vector_type<_Rebind_vector_generation_type<_ToSimdGeneration_,
        _Vector_element_type<_FromVector_>, _FromVector_>>>(_SimdUnwrap(_From));
}

template <
    arch::CpuFeature	_ToSimdGeneration_,
    class               _ToElementType_,
    class               _FromVector_,
    std::enable_if_t<_Is_valid_basic_simd_v<_FromVector_>
    || _Is_intrin_type_v<_FromVector_>, int>>
simd_stl_nodiscard simd_stl_always_inline _Rebind_vector_generation_type<_ToSimdGeneration_,
    _ToElementType_, _FromVector_> simd_cast(_FromVector_ _From) noexcept
{
    return __intrin_bitcast<_Unwrapped_vector_type<_Rebind_vector_generation_type<_ToSimdGeneration_,
        _ToElementType_, _FromVector_>>>(_SimdUnwrap(_From));
}

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_,
    class               _Type_,
    class               _MaskType_>
simd_stl_always_inline _Make_tail_mask_return_type<simd<_SimdGeneration_, _Type_, 
    _RegisterPolicy_>> _SimdConvertToMaskForNativeStore(_MaskType_ _Mask) noexcept
{
    using _ConvertTo = _Make_tail_mask_return_type<simd<_SimdGeneration_, _Type_, _RegisterPolicy_>>;

    constexpr auto _FromSimd = _Is_valid_basic_simd_v<_MaskType_>;
    constexpr auto _ToSimd = _Is_valid_basic_simd_v<_ConvertTo>;

    constexpr auto _FromIntegral = std::is_integral_v<_MaskType_>;
    constexpr auto _ToIntegral = std::is_integral_v<_ConvertTo>;

    constexpr auto _BothSimd = _FromSimd && _ToSimd;
    constexpr auto _BothIntegral = _FromIntegral && _ToIntegral;

    if constexpr (_BothSimd)
        return _ConvertTo { _SimdUnwrap(_Mask) };

    else if constexpr (_BothIntegral)
        return static_cast<_ConvertTo>(_Mask);

    else if constexpr (_FromIntegral && _ToSimd)
        return _ConvertTo { _SimdToVector<_SimdGeneration_, _RegisterPolicy_,
            typename _ConvertTo::vector_type, _Type_>(_Mask) 
        };

    else if constexpr (_FromSimd && _ToIntegral)
        return _SimdToMask<_SimdGeneration_, _RegisterPolicy_, _Type_>(_Mask);

    else
        static_assert(false, "Invalid _ConvertTo");
}

__SIMD_STL_NUMERIC_NAMESPACE_END
