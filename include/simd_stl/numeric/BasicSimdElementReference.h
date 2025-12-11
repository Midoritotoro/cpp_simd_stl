#pragma once 

#include <src/simd_stl/type_traits/SimdTypeCheck.h>
#include <src/simd_stl/type_traits/IsVirtualBaseOf.h>

#include <simd_stl/compatibility/Inline.h>
#include <src/simd_stl/utility/Assert.h>


__SIMD_STL_NUMERIC_NAMESPACE_BEGIN

template <
    typename _BasicSimd_, 
    typename _ImposedElementType_ = typename _BasicSimd_::value_type>
class BasicSimdElementReference {
    static_assert(_Is_valid_basic_simd_v<_BasicSimd_>);
public: 
    using parent_type   = _BasicSimd_;
    using vector_type   = typename _BasicSimd_::vector_type;
    using value_type    = _ImposedElementType_;

    BasicSimdElementReference(
        parent_type*    _Parent,
        uint32          _Index = 0) noexcept;

    ~BasicSimdElementReference() noexcept;

    simd_stl_always_inline uint32 index() const noexcept;
    simd_stl_always_inline value_type get() const noexcept;
    simd_stl_always_inline void set(value_type value) noexcept;

    simd_stl_always_inline BasicSimdElementReference& operator=(const value_type other) noexcept;
    simd_stl_always_inline BasicSimdElementReference& operator++() noexcept;
    simd_stl_always_inline BasicSimdElementReference& operator--() noexcept;
    simd_stl_always_inline BasicSimdElementReference operator++(int) noexcept;
    simd_stl_always_inline BasicSimdElementReference operator--(int) noexcept;

    simd_stl_always_inline value_type operator-() noexcept;
    simd_stl_always_inline value_type operator+() noexcept;
    simd_stl_always_inline operator value_type() const noexcept;

    simd_stl_always_inline bool operator==(const value_type other) const noexcept;
    simd_stl_always_inline bool operator!=(const value_type other) const noexcept;
    simd_stl_always_inline bool operator>(const value_type other) const noexcept;
    simd_stl_always_inline bool operator<(const value_type other) const noexcept;
    simd_stl_always_inline bool operator<=(const value_type other) const noexcept;
    simd_stl_always_inline bool operator>=(const value_type other) const noexcept;

    simd_stl_always_inline BasicSimdElementReference& operator+=(const value_type other) noexcept;
    simd_stl_always_inline BasicSimdElementReference& operator-=(const value_type other) noexcept;
    simd_stl_always_inline BasicSimdElementReference& operator*=(const value_type other) noexcept;
    simd_stl_always_inline BasicSimdElementReference& operator/=(const value_type other) noexcept;
    simd_stl_always_inline BasicSimdElementReference& operator%=(const value_type other) noexcept;
    simd_stl_always_inline BasicSimdElementReference& operator&=(const value_type other) noexcept;
    simd_stl_always_inline BasicSimdElementReference& operator^=(const value_type other) noexcept;
    simd_stl_always_inline BasicSimdElementReference& operator|=(const value_type other) noexcept;
private:
    parent_type*    _parent = nullptr;
    uint32          _index  = 0;
};

__SIMD_STL_NUMERIC_NAMESPACE_END

#include <src/simd_stl/numeric/BasicSimdElementReference.inl>