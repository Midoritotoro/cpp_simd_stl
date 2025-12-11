#pragma once 

#include <src/simd_stl/numeric/IntrinBitcast.h>
#include <src/simd_stl/type_traits/SimdTypeCheck.h>


__SIMD_STL_NUMERIC_NAMESPACE_BEGIN

template <
    class _RebindType_,
    class _VectorType_,
    bool _IsBasicSimd_  = _Is_valid_basic_simd_v<_VectorType_>,
    bool _IsIntrin_     = _Is_intrin_type_v<_VectorType_>>
struct _Rebind_vector_element_t;

template <
    arch::CpuFeature	_ToSimdGeneration_,
    class               _RebindType_,
    class               _VectorType_,
    bool                _IsBasicSimd_ = _Is_valid_basic_simd_v<_VectorType_>,
    bool                _IsIntrin_ = _Is_intrin_type_v<_VectorType_>>
struct _Rebind_vector_generation_t;

template <
    class _RebindType_,
    class _VectorType_>
using _Rebind_vector_element_type = typename _Rebind_vector_element_t<_RebindType_, _VectorType_>::type;

template <
    arch::CpuFeature	_ToSimdGeneration_,
    class               _RebindType_,
    class               _VectorType_>
using _Rebind_vector_generation_type = typename _Rebind_vector_generation_t<_ToSimdGeneration_, _RebindType_, _VectorType_>::type;

template <class _VectorType_>
_Simd_nodiscard_inline _Unwrapped_vector_type<_VectorType_> _SimdUnwrap(_VectorType_ _Vector) noexcept;

template <
    class _ToType_,
    class _FromType_,
    std::enable_if_t<(_Is_valid_basic_simd_v<_ToType_> || _Is_intrin_type_v<_ToType_> ||
        type_traits::__is_vector_type_supported_v<_ToType_>) &&
        (_Is_valid_basic_simd_v<_FromType_> || _Is_intrin_type_v<_FromType_>), int> = 0>
_Simd_nodiscard_inline _Rebind_vector_element_type<_ToType_, _FromType_> simd_cast(_FromType_ _From) noexcept;

template <
    arch::CpuFeature	_ToSimdGeneration_,
    class               _FromVector_,
    std::enable_if_t<_Is_valid_basic_simd_v<_FromVector_> 
        || _Is_intrin_type_v<_FromVector_>, int> = 0>
_Simd_nodiscard_inline _Rebind_vector_generation_type<_ToSimdGeneration_,
    _Vector_element_type<_FromVector_>, _FromVector_> simd_cast(_FromVector_ _From) noexcept;

template <
    arch::CpuFeature	_ToSimdGeneration_,
    class               _ToElementType_,
    class               _FromVector_,
    std::enable_if_t<_Is_valid_basic_simd_v<_FromVector_> 
        || _Is_intrin_type_v<_FromVector_>, int> = 0>
_Simd_nodiscard_inline _Rebind_vector_generation_type<_ToSimdGeneration_,
    _ToElementType_, _FromVector_> simd_cast(_FromVector_ _From) noexcept;

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_,
    class               _Type_,
    class               _MaskType_>
_Simd_nodiscard_inline _Make_tail_mask_return_type<basic_simd<_SimdGeneration_, _Type_,
    _RegisterPolicy_>> _SimdConvertToMaskForNativeStore(_MaskType_ _Mask) noexcept;

__SIMD_STL_NUMERIC_NAMESPACE_END

#include <src/simd_stl/numeric/SimdCast.inl>