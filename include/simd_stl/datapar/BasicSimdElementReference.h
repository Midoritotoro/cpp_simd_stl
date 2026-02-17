#pragma once 

#include <src/simd_stl/type_traits/SimdTypeCheck.h>
#include <src/simd_stl/type_traits/IsVirtualBaseOf.h>

#include <simd_stl/compatibility/Inline.h>
#include <src/simd_stl/utility/Assert.h>


__SIMD_STL_DATAPAR_NAMESPACE_BEGIN

template <
    typename _Simd_, 
    typename _ImposedElementType_ = typename _Simd_::value_type>
class simd_element_reference {
    static_assert(__is_valid_simd_v<_Simd_>);
public: 
    using parent_type   = _Simd_;
    using vector_type   = typename _Simd_::vector_type;
    using value_type    = _ImposedElementType_;

    simd_element_reference(
        parent_type*    __parent,
        uint32          __index = 0) noexcept;

    ~simd_element_reference() noexcept;

    simd_stl_always_inline uint32 index() const noexcept;
    simd_stl_always_inline value_type get() const noexcept;
    simd_stl_always_inline void set(value_type value) noexcept;

    simd_stl_always_inline simd_element_reference& operator=(const value_type __other) noexcept;
    simd_stl_always_inline simd_element_reference& operator++() noexcept;
    simd_stl_always_inline simd_element_reference& operator--() noexcept;
    simd_stl_always_inline simd_element_reference operator++(int) noexcept;
    simd_stl_always_inline simd_element_reference operator--(int) noexcept;

    simd_stl_always_inline value_type operator-() noexcept;
    simd_stl_always_inline value_type operator+() noexcept;
    simd_stl_always_inline operator value_type() const noexcept;

    simd_stl_always_inline bool operator==(const value_type __other) const noexcept;
    simd_stl_always_inline bool operator!=(const value_type __other) const noexcept;
    simd_stl_always_inline bool operator>(const value_type __other) const noexcept;
    simd_stl_always_inline bool operator<(const value_type __other) const noexcept;
    simd_stl_always_inline bool operator<=(const value_type __other) const noexcept;
    simd_stl_always_inline bool operator>=(const value_type __other) const noexcept;

    simd_stl_always_inline simd_element_reference& operator+=(const value_type __other) noexcept;
    simd_stl_always_inline simd_element_reference& operator-=(const value_type __other) noexcept;
    simd_stl_always_inline simd_element_reference& operator*=(const value_type __other) noexcept;
    simd_stl_always_inline simd_element_reference& operator/=(const value_type __other) noexcept;
    simd_stl_always_inline simd_element_reference& operator%=(const value_type __other) noexcept;
    simd_stl_always_inline simd_element_reference& operator&=(const value_type __other) noexcept;
    simd_stl_always_inline simd_element_reference& operator^=(const value_type __other) noexcept;
    simd_stl_always_inline simd_element_reference& operator|=(const value_type __other) noexcept;
private:
    parent_type*    _parent = nullptr;
    uint32          _index  = 0;
};

__SIMD_STL_DATAPAR_NAMESPACE_END

#include <src/simd_stl/datapar/BasicSimdElementReference.inl>