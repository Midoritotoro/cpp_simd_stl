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

    using __parent_impl = typename _BasicSimd_::__impl;
public: 
    using parent_type   = _BasicSimd_;

    using vector_type   = typename _BasicSimd_::vector_type;
    using value_type    = typename _BasicSimd_::value_type;

    simd_stl_constexpr_cxx20 BasicSimdElementReference(
        parent_type*    parent,
        uint8           index = 0
    ) noexcept:
        _parent(parent),
        _index(index)
    {}

    simd_stl_constexpr_cxx20 ~BasicSimdElementReference() noexcept 
    {}

    simd_stl_constexpr_cxx20 simd_stl_always_inline value_type get() const noexcept {
        return __parent_impl::extract(_parent->_vector, _index);
    }

    simd_stl_constexpr_cxx20 simd_stl_always_inline void set(value_type value) noexcept {
        return __parent_impl::insert(_parent->_vector, _index, value);
    }

    simd_stl_constexpr_cxx20 simd_stl_always_inline BasicSimdElementReference& operator=(const value_type other) noexcept {
        set(other);
        return *this;
    }

private:
    parent_type*    _parent = nullptr;
    uint8           _index  = 0;   
};

__SIMD_STL_NUMERIC_NAMESPACE_END
