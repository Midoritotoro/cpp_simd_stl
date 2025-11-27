#pragma once 

#include <src/simd_stl/numeric/IntrinBitcast.h>
#include <src/simd_stl/type_traits/SimdTypeCheck.h>


__SIMD_STL_NUMERIC_NAMESPACE_BEGIN

template <
    class _ToElementType_,
    class _VectorType_,
    bool _IsBasicSimd_  = _Is_valid_basic_simd_v<_VectorType_>,
    bool _IsIntrin_     = _Is_intrin_type_v<_VectorType_>>
struct _Rebind_vector_element_t {
    using type = void;
};

template <
    class _ToElementType_,
    class _VectorType_>
struct _Rebind_vector_element_t< _ToElementType_, _VectorType_, false, true> {
    using type =
        std::conditional_t<sizeof(_VectorType_) == _ZmmWidth,
            type_traits::__deduce_simd_vector_type<arch::CpuFeature::AVX512F, _ToElementType_>,
        std::conditional_t<sizeof(_VectorType_) == _YmmWidth,
            type_traits::__deduce_simd_vector_type<arch::CpuFeature::AVX, _ToElementType_>,
        std::conditional_t<sizeof(_VectorType_) == _XmmWidth,
            type_traits::__deduce_simd_vector_type<arch::CpuFeature::SSE2, _ToElementType_>, void>>>;
};

template <
    class _ToElementType_,
    class _VectorType_>
struct _Rebind_vector_element_t<_ToElementType_, _VectorType_, true, false> {
    using type = basic_simd<_VectorType_::_Generation, _ToElementType_, typename _VectorType_::policy>;
};

template <
    arch::CpuFeature	_ToSimdGeneration_,
    class               _ToElementType_,
    class               _VectorType_,
    bool                _IsBasicSimd_  = _Is_valid_basic_simd_v<_VectorType_>,
    bool                _IsIntrin_     = _Is_intrin_type_v<_VectorType_>>
struct _Rebind_vector_generation_t {
    using type = void;
};

template <
    arch::CpuFeature	_ToSimdGeneration_,
    class               _ToElementType_,
    class               _VectorType_>
struct _Rebind_vector_generation_t<_ToSimdGeneration_, _ToElementType_, _VectorType_, false, true> {
    using type = type_traits::__deduce_simd_vector_type<_ToSimdGeneration_, _ToElementType_>;
};

template <
    arch::CpuFeature	_ToSimdGeneration_,
    class               _ToElementType_,
    class               _VectorType_>
struct _Rebind_vector_generation_t<_ToSimdGeneration_, _ToElementType_, _VectorType_, true, false> {
    using type = basic_simd<_ToSimdGeneration_, _ToElementType_, typename _VectorType_::policy>;
};

template <
    class _ToElementType_,
    class _VectorType_>
using _Rebind_vector_element_type = typename _Rebind_vector_element_t<_ToElementType_, _VectorType_>::type;

template <
    arch::CpuFeature	_ToSimdGeneration_,
    class               _ToElementType_,
    class               _VectorType_>
using _Rebind_vector_generation_type = typename _Rebind_vector_generation_t<_ToSimdGeneration_, _ToElementType_, _VectorType_>::type;

template <
    class _ToVector_,
    class _FromVector_,
    std::enable_if_t<(_Is_valid_basic_simd_v<_ToVector_> || _Is_intrin_type_v<_ToVector_>) &&
        (_Is_valid_basic_simd_v<_FromVector_> || _Is_intrin_type_v<_FromVector_>), int> = 0>
simd_stl_nodiscard simd_stl_always_inline _ToVector_ simd_cast(_FromVector_ _From) noexcept {
    if constexpr (_Is_valid_basic_simd_v<_FromVector_> && _Is_valid_basic_simd_v<_ToVector_>)
        return _IntrinBitcast<typename _ToVector_::vector_type>(_From.unwrap());

    else if constexpr (_Is_intrin_type_v<_FromVector_> && _Is_valid_basic_simd_v<_ToVector_>)
        return _IntrinBitcast<typename _ToVector_::vector_type>(_From);

    else if constexpr (_Is_valid_basic_simd_v<_FromVector_> && _Is_intrin_type_v<_ToVector_>)
        return _IntrinBitcast<_ToVector_>(_From.unwrap());

    else if constexpr (_Is_intrin_type_v<_FromVector_> && _Is_intrin_type_v<_ToVector_>)
        return _IntrinBitcast<_ToVector_>(_From);
}

template <
    class _ToElementType_,
    class _FromVector_,
    std::enable_if_t<(_Is_valid_basic_simd_v<_FromVector_> || _Is_intrin_type_v<_FromVector_>) &&
        (type_traits::__is_vector_type_supported_v<_ToElementType_>), int> = 0>
simd_stl_nodiscard simd_stl_always_inline _Rebind_vector_element_type<_ToElementType_, _FromVector_> simd_cast(_FromVector_ _From) noexcept {
    if constexpr (_Is_valid_basic_simd_v<_FromVector_>)
        return _IntrinBitcast<_Rebind_vector_element_type<_ToElementType_, _FromVector_>>(_From.unwrap());

    else if constexpr (_Is_intrin_type_v<_FromVector_>)
        return _IntrinBitcast<_Rebind_vector_element_type<_ToElementType_, _FromVector_>>(_From);
}

template <
    arch::CpuFeature	_ToSimdGeneration_,
    class               _FromVector_,
    std::enable_if_t<_Is_valid_basic_simd_v<_FromVector_> 
        || _Is_intrin_type_v<_FromVector_>, int> = 0>
simd_stl_nodiscard simd_stl_always_inline _Rebind_vector_generation_type<_ToSimdGeneration_, 
    _Vector_element_type<_FromVector_>, _FromVector_> simd_cast(_FromVector_ _From) noexcept
{
    if constexpr (_Is_valid_basic_simd_v<_FromVector_>)
        return _IntrinBitcast<typename _Rebind_vector_generation_type<_ToSimdGeneration_, 
            _Vector_element_type<_FromVector_>, _FromVector_>::vector_type>(_From.unwrap());

    else if constexpr (_Is_intrin_type_v<_FromVector_>)
        return _IntrinBitcast<_Rebind_vector_generation_type<_ToSimdGeneration_, 
            _Vector_element_type<_FromVector_>, _FromVector_>>(_From);
}

template <
    arch::CpuFeature	_ToSimdGeneration_,
    class               _ToElementType_,
    class               _FromVector_,
    std::enable_if_t<_Is_valid_basic_simd_v<_FromVector_> 
        || _Is_intrin_type_v<_FromVector_>, int> = 0>
simd_stl_nodiscard simd_stl_always_inline _Rebind_vector_generation_type<_ToSimdGeneration_, 
    _ToElementType_, _FromVector_> simd_cast(_FromVector_ _From) noexcept
{
    if constexpr (_Is_valid_basic_simd_v<_FromVector_>)
        return _IntrinBitcast<typename _Rebind_vector_generation_type<_ToSimdGeneration_, 
            _ToElementType_, _FromVector_>::vector_type>(_From.unwrap());

    else if constexpr (_Is_intrin_type_v<_FromVector_>)
        return _IntrinBitcast<_Rebind_vector_generation_type<_ToSimdGeneration_, 
            _ToElementType_, _FromVector_>>(_From);
}

__SIMD_STL_NUMERIC_NAMESPACE_END
