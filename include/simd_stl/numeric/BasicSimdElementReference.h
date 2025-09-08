#pragma once 

#include <src/simd_stl/type_traits/SimdTypeCheck.h>
#include <src/simd_stl/type_traits/IsVirtualBaseOf.h>

__SIMD_STL_NUMERIC_NAMESPACE_BEGIN

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_>
class basic_simd;

template <typename _BasicSimd_>
constexpr bool __is_valid_basic_simd_v = std::disjunction_v<
    type_traits::is_virtual_base_of<
        basic_simd<_BasicSimd_::_Generation, typename _BasicSimd_::value_type>, _BasicSimd_>,
    std::is_same<
        basic_simd<_BasicSimd_::_Generation, typename _BasicSimd_::value_type>, _BasicSimd_>
>;

template <typename _BasicSimd_>
class BasicSimdElementReference {
    static_assert(__is_valid_basic_simd_v<_BasicSimd_>);
public: 
    using vector_type   = typename _BasicSimd_::vector_type;
    using value_type    = typename _BasicSimd_::value_type;


};

__SIMD_STL_NUMERIC_NAMESPACE_END
