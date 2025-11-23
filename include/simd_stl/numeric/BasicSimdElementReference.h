#pragma once 

#include <src/simd_stl/type_traits/SimdTypeCheck.h>
#include <src/simd_stl/type_traits/IsVirtualBaseOf.h>

#include <simd_stl/compatibility/Inline.h>
#include <src/simd_stl/utility/Assert.h>


__SIMD_STL_NUMERIC_NAMESPACE_BEGIN
    
template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
class basic_simd;

template <typename _BasicSimd_>
constexpr bool _Is_valid_basic_simd_v = std::disjunction_v<
        type_traits::is_virtual_base_of<
            basic_simd<_BasicSimd_::_Generation, typename _BasicSimd_::value_type, typename _BasicSimd_::policy>, _BasicSimd_>,
        std::is_same<
            basic_simd<_BasicSimd_::_Generation, typename _BasicSimd_::value_type, typename _BasicSimd_::policy>, _BasicSimd_>
    >;


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
        parent_type*    parent,
        uint32          index = 0
    ) noexcept:
        _parent(parent),
        _index(index)
    {
        DebugAssert(parent != nullptr);
        DebugAssert(index >= 0 && index < parent_type::template size<value_type>());
    }

    ~BasicSimdElementReference() noexcept 
    {}

    simd_stl_always_inline uint32 index() const noexcept {
        return _index;
    }

    simd_stl_always_inline value_type get() const noexcept {
        return _parent->extract<value_type>(_index);
    }

    simd_stl_always_inline void set(value_type value) noexcept {
        _parent->insert<value_type>(_index, value);
    }

    simd_stl_always_inline BasicSimdElementReference& operator=(const value_type other) noexcept {
        set(other);
        return *this;
    }

    simd_stl_always_inline BasicSimdElementReference& operator++() noexcept {
        set(get() + 1);
        return *this;
    }

    simd_stl_always_inline BasicSimdElementReference& operator--() noexcept {
        set(get() - 1);
        return *this;
    }

    simd_stl_always_inline BasicSimdElementReference operator++(int) noexcept {
        auto self = *this;
        set(get() + 1);
        return self;
    }

    simd_stl_always_inline BasicSimdElementReference operator--(int) noexcept {
        auto self = *this;
        set(get() - 1);
        return self;
    }

    simd_stl_always_inline value_type operator-() noexcept {
        return -get();
    }

    simd_stl_always_inline value_type operator+() noexcept {
        return get();
    }

    simd_stl_always_inline operator value_type() const noexcept {
        return get();
    }

    simd_stl_always_inline bool operator==(const value_type other) const noexcept {
        return get() == other;
    }

    simd_stl_always_inline bool operator!=(const value_type other) const noexcept {
        return get() != other;
    }

    simd_stl_always_inline bool operator>(const value_type other) const noexcept {
        return get() > other;
    }

    simd_stl_always_inline bool operator<(const value_type other) const noexcept {
        return get() < other;
    }

    simd_stl_always_inline bool operator<=(const value_type other) const noexcept {
        return get() <= other;
    }

    simd_stl_always_inline bool operator>=(const value_type other) const noexcept {
        return get() >= other;
    }

    simd_stl_always_inline BasicSimdElementReference& operator+=(const value_type other) noexcept {
        set(get() + other);
        return *this;
    }

    simd_stl_always_inline BasicSimdElementReference& operator-=(const value_type other) noexcept {
        set(get() - other);
        return *this;
    }

    simd_stl_always_inline BasicSimdElementReference& operator*=(const value_type other) noexcept {
        set(get() * other);
        return *this;
    }

    simd_stl_always_inline BasicSimdElementReference& operator/=(const value_type other) noexcept {
        set(get() / other);
        return *this;
    }

    simd_stl_always_inline BasicSimdElementReference& operator%=(const value_type other) noexcept {
        set(get() % other);
        return *this;
    }

    simd_stl_always_inline BasicSimdElementReference& operator&=(const value_type other) noexcept {
        set(get() & other);
        return *this;
    }

    simd_stl_always_inline BasicSimdElementReference& operator^=(const value_type other) noexcept {
        set(get() ^ other);
        return *this;
    }

    simd_stl_always_inline BasicSimdElementReference& operator|=(const value_type other) noexcept {
        set(get() | other);
        return *this;
    }
private:
    parent_type*    _parent = nullptr;
    uint32          _index  = 0;
};

__SIMD_STL_NUMERIC_NAMESPACE_END
