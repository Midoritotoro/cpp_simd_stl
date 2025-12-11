#pragma once 

__SIMD_STL_NUMERIC_NAMESPACE_BEGIN

template <
    typename _BasicSimd_,
    typename _ImposedElementType_>
BasicSimdElementReference<_BasicSimd_, _ImposedElementType_>::BasicSimdElementReference(
    parent_type*    _Parent,
    uint32          _Index
) noexcept:
    _parent(_Parent),
    _index(_Index)
{
    DebugAssert(_Parent != nullptr);
    DebugAssert(_Index >= 0 && _Index < parent_type::template size<value_type>());
}

template <
    typename _BasicSimd_,
    typename _ImposedElementType_>
BasicSimdElementReference<_BasicSimd_, _ImposedElementType_>::~BasicSimdElementReference() noexcept
{}

template <
    typename _BasicSimd_,
    typename _ImposedElementType_>
simd_stl_always_inline uint32 BasicSimdElementReference<_BasicSimd_, _ImposedElementType_>::index() const noexcept {
    return _index;
}

template <
    typename _BasicSimd_,
    typename _ImposedElementType_>
simd_stl_always_inline BasicSimdElementReference<_BasicSimd_, _ImposedElementType_>::value_type
    BasicSimdElementReference<_BasicSimd_, _ImposedElementType_>::get() const noexcept 
{
    return _parent->extract<value_type>(_index);
}

template <
    typename _BasicSimd_,
    typename _ImposedElementType_>
simd_stl_always_inline void BasicSimdElementReference<_BasicSimd_, _ImposedElementType_>::set(value_type value) noexcept {
    _parent->insert<value_type>(_index, value);
}

template <
    typename _BasicSimd_,
    typename _ImposedElementType_>
simd_stl_always_inline BasicSimdElementReference<_BasicSimd_, _ImposedElementType_>& 
    BasicSimdElementReference<_BasicSimd_, _ImposedElementType_>::operator=(const value_type other) noexcept 
{
    set(other);
    return *this;
}

template <
    typename _BasicSimd_,
    typename _ImposedElementType_>
simd_stl_always_inline BasicSimdElementReference<_BasicSimd_, _ImposedElementType_>& 
    BasicSimdElementReference<_BasicSimd_, _ImposedElementType_>::operator++() noexcept 
{
    set(get() + 1);
    return *this;
}

template <
    typename _BasicSimd_,
    typename _ImposedElementType_>
simd_stl_always_inline BasicSimdElementReference<_BasicSimd_, _ImposedElementType_>& 
    BasicSimdElementReference<_BasicSimd_, _ImposedElementType_>::operator--() noexcept
{
    set(get() - 1);
    return *this;
}

template <
    typename _BasicSimd_,
    typename _ImposedElementType_>
simd_stl_always_inline BasicSimdElementReference<_BasicSimd_, _ImposedElementType_>
    BasicSimdElementReference<_BasicSimd_, _ImposedElementType_>::operator++(int) noexcept 
{
    auto self = *this;
    set(get() + 1);
    return self;
}

template <
    typename _BasicSimd_,
    typename _ImposedElementType_>
simd_stl_always_inline BasicSimdElementReference<_BasicSimd_, _ImposedElementType_>
    BasicSimdElementReference<_BasicSimd_, _ImposedElementType_>::operator--(int) noexcept 
{
    auto self = *this;
    set(get() - 1);
    return self;
}

template <
    typename _BasicSimd_,
    typename _ImposedElementType_>
simd_stl_always_inline BasicSimdElementReference<_BasicSimd_, _ImposedElementType_>::value_type
    BasicSimdElementReference<_BasicSimd_, _ImposedElementType_>::operator-() noexcept
{
    return -get();
}

template <
    typename _BasicSimd_,
    typename _ImposedElementType_>
simd_stl_always_inline BasicSimdElementReference<_BasicSimd_, _ImposedElementType_>::value_type
    BasicSimdElementReference<_BasicSimd_, _ImposedElementType_>::operator+() noexcept 
{
    return get();
}

template <
    typename _BasicSimd_,
    typename _ImposedElementType_>
simd_stl_always_inline BasicSimdElementReference<_BasicSimd_, _ImposedElementType_>::operator value_type() const noexcept {
    return get();
}

template <
    typename _BasicSimd_,
    typename _ImposedElementType_>
simd_stl_always_inline bool BasicSimdElementReference<_BasicSimd_, _ImposedElementType_>
    ::operator==(const value_type other) const noexcept
{
    return get() == other;
}

template <
    typename _BasicSimd_,
    typename _ImposedElementType_>
simd_stl_always_inline bool BasicSimdElementReference<_BasicSimd_, _ImposedElementType_>
    ::operator!=(const value_type other) const noexcept
{
    return get() != other;
}

template <
    typename _BasicSimd_,
    typename _ImposedElementType_>
simd_stl_always_inline bool BasicSimdElementReference<_BasicSimd_, _ImposedElementType_>
    ::operator>(const value_type other) const noexcept
{
    return get() > other;
}

template <
    typename _BasicSimd_,
    typename _ImposedElementType_>
simd_stl_always_inline bool BasicSimdElementReference<_BasicSimd_, _ImposedElementType_>
    ::operator<(const value_type other) const noexcept
{
    return get() < other;
}

template <
    typename _BasicSimd_,
    typename _ImposedElementType_>
simd_stl_always_inline bool BasicSimdElementReference<_BasicSimd_, _ImposedElementType_>
    ::operator<=(const value_type other) const noexcept
{
    return get() <= other;
}

template <
    typename _BasicSimd_,
    typename _ImposedElementType_>
simd_stl_always_inline bool BasicSimdElementReference<_BasicSimd_, _ImposedElementType_>
    ::operator>=(const value_type other) const noexcept
{
    return get() >= other;
}

template <
    typename _BasicSimd_,
    typename _ImposedElementType_>
simd_stl_always_inline BasicSimdElementReference<_BasicSimd_, _ImposedElementType_>& 
    BasicSimdElementReference<_BasicSimd_, _ImposedElementType_>::operator+=(const value_type other) noexcept 
{
    set(get() + other);
    return *this;
}

template <
    typename _BasicSimd_,
    typename _ImposedElementType_>
simd_stl_always_inline BasicSimdElementReference<_BasicSimd_, _ImposedElementType_>& 
    BasicSimdElementReference<_BasicSimd_, _ImposedElementType_>::operator-=(const value_type other) noexcept
{
    set(get() - other);
    return *this;
}

template <
    typename _BasicSimd_,
    typename _ImposedElementType_>
simd_stl_always_inline BasicSimdElementReference<_BasicSimd_, _ImposedElementType_>&
    BasicSimdElementReference<_BasicSimd_, _ImposedElementType_>::operator*=(const value_type other) noexcept
{
    set(get() * other);
    return *this;
}

template <
    typename _BasicSimd_,
    typename _ImposedElementType_>
simd_stl_always_inline BasicSimdElementReference<_BasicSimd_, _ImposedElementType_>& 
    BasicSimdElementReference<_BasicSimd_, _ImposedElementType_>::operator/=(const value_type other) noexcept
{
    set(get() / other);
    return *this;
}

template <
    typename _BasicSimd_,
    typename _ImposedElementType_>
simd_stl_always_inline BasicSimdElementReference<_BasicSimd_, _ImposedElementType_>& 
    BasicSimdElementReference<_BasicSimd_, _ImposedElementType_>::operator%=(const value_type other) noexcept 
{
    set(get() % other);
    return *this;
}

template <
    typename _BasicSimd_,
    typename _ImposedElementType_>
simd_stl_always_inline BasicSimdElementReference<_BasicSimd_, _ImposedElementType_>& 
    BasicSimdElementReference<_BasicSimd_, _ImposedElementType_>::operator&=(const value_type other) noexcept
{
    set(get() & other);
    return *this;
}

template <
    typename _BasicSimd_,
    typename _ImposedElementType_>
simd_stl_always_inline BasicSimdElementReference<_BasicSimd_, _ImposedElementType_>&
    BasicSimdElementReference<_BasicSimd_, _ImposedElementType_>::operator^=(const value_type other) noexcept
{
    set(get() ^ other);
    return *this;
}

template <
    typename _BasicSimd_,
    typename _ImposedElementType_>
simd_stl_always_inline BasicSimdElementReference<_BasicSimd_, _ImposedElementType_>&
    BasicSimdElementReference<_BasicSimd_, _ImposedElementType_>::operator|=(const value_type other) noexcept
{
    set(get() | other);
    return *this;
}

__SIMD_STL_NUMERIC_NAMESPACE_END
