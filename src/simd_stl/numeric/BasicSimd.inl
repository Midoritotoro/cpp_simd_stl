#pragma once 

__SIMD_STL_NUMERIC_NAMESPACE_BEGIN

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
template <bool _ZeroMemset_>
basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::basic_simd() noexcept {
    if constexpr (_ZeroMemset_)
        _vector = _SimdBroadcastZeros<_SimdGeneration_, _RegisterPolicy_, vector_type>();
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
template <
    typename _IntrinType_,
    std::enable_if_t<_Is_intrin_type_v<_IntrinType_>, int>>
basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::basic_simd(_IntrinType_ other) noexcept {
    _vector = simd_cast<vector_type>(other);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::basic_simd(const value_type value) noexcept {
    fill<value_type>(value);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
template <typename _OtherType_>
basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::basic_simd(const basic_simd<_SimdGeneration_, _OtherType_, _RegisterPolicy_>& other) noexcept {
    _vector = simd_cast<vector_type>(other);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
template <
    arch::CpuFeature    _OtherFeature_,
    typename            _OtherType_>
basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::basic_simd(const basic_simd<_OtherFeature_, _OtherType_, _RegisterPolicy_>& other) noexcept {
    _vector = simd_cast<vector_type>(other);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::~basic_simd() noexcept
{
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
template <class _BasicSimdTo_>
simd_stl_always_inline _BasicSimdTo_ basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::convert() const noexcept {
    return {};
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
simd_stl_always_inline bool basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::isSupported() noexcept {
    return arch::ProcessorFeatures::isSupported<_SimdGeneration_>();
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
simd_stl_always_inline void basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::broadcastZeros() noexcept {
    _vector = _SimdBroadcastZeros<_SimdGeneration_, _RegisterPolicy_, vector_type>();
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
template <typename _DesiredType_>
simd_stl_always_inline void basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>
    ::fill(const typename std::type_identity<_DesiredType_>::type value) noexcept
{
    _vector = _SimdBroadcast<_SimdGeneration_, _RegisterPolicy_, vector_type>(value);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>&
basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::operator+=(const basic_simd& other) noexcept {
    return *this = (*this + other);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>&
basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::operator-=(const basic_simd& other) noexcept {
    return *this = (*this - other);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>&
basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::operator*=(const basic_simd& other) noexcept {
    return *this = (*this * other);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>&
basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::operator/=(const basic_simd& other) noexcept {
    return *this = (*this / other);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>&
basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::operator=(const basic_simd& left) noexcept {
    _vector = left._vector;
    return *this;
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
simd_stl_always_inline _Element_
basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::operator[](const size_type index) const noexcept {
    return _SimdExtract<_SimdGeneration_, _RegisterPolicy_, _Element_>(_vector, index);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
simd_stl_always_inline BasicSimdElementReference<basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>>
basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::operator[](const size_type index) noexcept {
    return BasicSimdElementReference<basic_simd>(this, index);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>
basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::operator++(int) noexcept {
    basic_simd self = *this;
    _vector = _SimdAdd<_SimdGeneration_, _RegisterPolicy_, _Element_>(
        _vector, _SimdBroadcast<_SimdGeneration_, _RegisterPolicy_, vector_type, _Element_>(1));

    return self;
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>&
basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::operator++() noexcept {
    _vector = _SimdAdd<_SimdGeneration_, _RegisterPolicy_, _Element_>(
        _vector, _SimdBroadcast<_SimdGeneration_, _RegisterPolicy_, vector_type, _Element_>(1));

    return *this;
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>
basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::operator--(int) noexcept
{
    basic_simd self = *this;
    _vector = _SimdSubstract<_SimdGeneration_, _RegisterPolicy_, _Element_>(
        _vector, _SimdBroadcast<_SimdGeneration_, _RegisterPolicy_, vector_type, _Element_>(1));

    return self;
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>&
basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::operator--() noexcept
{
    _vector = _SimdSubstract<_SimdGeneration_, _RegisterPolicy_, _Element_>(
        _vector, _SimdBroadcast<_SimdGeneration_, _RegisterPolicy_, vector_type, _Element_>(1));

    return *this;
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::mask_type
basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::operator!() const noexcept {
    return !toMask();
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>
basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::operator~() const noexcept {
    return _SimdBitNot<_SimdGeneration_, _RegisterPolicy_>(_vector);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>&
basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::operator&=(const basic_simd& other) noexcept {
    return *this = (*this & other);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>&
basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::operator|=(const basic_simd& other) noexcept {
    return *this = (*this | other);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>&
basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::operator^=(const basic_simd& other) noexcept {
    return *this = (*this ^ other);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_> operator/(
    const basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>& left,
    const basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>& right) noexcept
{
    return _SimdDivide<_SimdGeneration_, _RegisterPolicy_, _Element_>(left._vector, right._vector);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_> operator+(
    const basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>& left,
    const basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>& right) noexcept
{
    return _SimdAdd<_SimdGeneration_, _RegisterPolicy_, _Element_>(left._vector, right._vector);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_> operator-(
    const basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>& left,
    const basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>& right) noexcept
{
    return _SimdSubstract<_SimdGeneration_, _RegisterPolicy_, _Element_>(left._vector, right._vector);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_> operator*(
    const basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>& left,
    const basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>& right) noexcept
{
    return _SimdMultiply<_SimdGeneration_, _RegisterPolicy_, _Element_>(left._vector, right._vector);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_> operator&(
    const basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>& left,
    const basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>& right) noexcept
{
    return _SimdBitAnd<_SimdGeneration_, _RegisterPolicy_>(left._vector, right._vector);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_> operator|(
    const basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>& left,
    const basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>& right) noexcept
{
    return _SimdBitOr<_SimdGeneration_, _RegisterPolicy_>(left._vector, right._vector);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_> operator^(
    const basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>& left,
    const basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>& right) noexcept
{
    return _SimdBitXor<_SimdGeneration_, _RegisterPolicy_>(left._vector, right._vector);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_> operator+(
    const basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>& left,
    const typename basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::value_type  right) noexcept
{
    return _SimdAdd<_SimdGeneration_, _RegisterPolicy_, _Element_>(
        left._vector, _SimdBroadcast<_SimdGeneration_, _RegisterPolicy_, 
            typename basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::vector_type>(right));
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_> operator-(
    const basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>& left,
    const typename basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::value_type  right) noexcept
{
    return _SimdSubstract<_SimdGeneration_, _RegisterPolicy_, _Element_>(
        left._vector, _SimdBroadcast<_SimdGeneration_, _RegisterPolicy_, 
            typename basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::vector_type>(right));
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_> operator*(
    const basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>& left,
    const typename basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::value_type  right) noexcept
{
    return _SimdMultiply<_SimdGeneration_, _RegisterPolicy_, _Element_>(
        left._vector, _SimdBroadcast<_SimdGeneration_, _RegisterPolicy_, 
            typename basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::vector_type>(right));
}


template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_> operator/(
    const basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>& left,
    const typename basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::value_type  right) noexcept
{
    return _SimdDivide<_SimdGeneration_, _RegisterPolicy_, _Element_>(
        left._vector, _SimdBroadcast<_SimdGeneration_, _RegisterPolicy_, 
            typename basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::vector_type>(right));
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>
basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::operator+() const noexcept
{
    return _vector;
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>
basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::operator-() const noexcept
{
    return _SimdNegate<_SimdGeneration_, _RegisterPolicy_, _Element_>(_vector);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
template <typename _DesiredType_>
simd_stl_always_inline _DesiredType_
basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::extract(const size_type index) const noexcept
{
    DebugAssert(index >= 0 && index < size<_DesiredType_>(), "simd_stl::numeric::basic_simd: Index out of range");
    return _SimdExtract<_SimdGeneration_, _RegisterPolicy_, _DesiredType_>(_vector, index);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
template <typename _DesiredType_>
simd_stl_always_inline BasicSimdElementReference<basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>>
basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::extractWrapped(const size_type index) noexcept
{
    DebugAssert(index >= 0 && index < size<_DesiredType_>(), "simd_stl::numeric::basic_simd: Index out of range");
    return BasicSimdElementReference<basic_simd>(this, index);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
template <typename _DesiredType_>
simd_stl_always_inline void basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::insert(
    const size_type                                         where,
    const typename std::type_identity<_DesiredType_>::type  value) noexcept
{
    return _SimdInsert<_SimdGeneration_, _RegisterPolicy_, _DesiredType_>(_vector, where, value);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>
basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::loadUnaligned(const void* where)  noexcept
{
    return _SimdLoadUnaligned<_SimdGeneration_, _RegisterPolicy_, vector_type>(where);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>
basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::loadAligned(const void* where) noexcept
{
    return _SimdLoadAligned<_SimdGeneration_, _RegisterPolicy_, vector_type>(where);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
simd_stl_always_inline void basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::storeUnaligned(void* where) const noexcept {
    return _SimdStoreUnaligned<_SimdGeneration_, _RegisterPolicy_, vector_type>(where, _vector);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
simd_stl_always_inline void basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::storeAligned(void* where) const noexcept {
    return _SimdStoreAligned<_SimdGeneration_, _RegisterPolicy_, vector_type>(where, _vector);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_> basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::loadUpperHalf(const void* where) noexcept {
    return _SimdLoadUpperHalf<_SimdGeneration_, _RegisterPolicy_, vector_type>(where);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_> basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::loadLowerHalf(const void* where) noexcept {
    return _SimdLoadLowerHalf<_SimdGeneration_, _RegisterPolicy_, vector_type>(where);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
template <typename _DesiredType_>
simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>
basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::maskLoadUnaligned(
    const void* where,
    const _Mask_type<_DesiredType_>   mask) noexcept
{
    return _SimdMaskLoadUnaligned<_SimdGeneration_, _RegisterPolicy_, vector_type, _DesiredType_>(
        reinterpret_cast<const _DesiredType_*>(where), mask);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
template <typename _DesiredType_>
simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>
basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::maskLoadAligned(
    const void* where,
    const _Mask_type<_DesiredType_>   mask) noexcept
{
    return _SimdMaskLoadAligned<_SimdGeneration_, _RegisterPolicy_, vector_type, _DesiredType_>(
        reinterpret_cast<const _DesiredType_*>(where), mask);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
template <typename _DesiredType_>
static simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>
basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::maskLoadUnaligned(
    const void* where,
    const basic_simd<_SimdGeneration_, _DesiredType_, _RegisterPolicy_>& mask) noexcept
{
    return _SimdMaskLoadUnaligned<_SimdGeneration_, _RegisterPolicy_, vector_type, _DesiredType_>(
        reinterpret_cast<const _DesiredType_*>(where), mask._vector);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
template <typename _DesiredType_>
static simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>
basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::maskLoadAligned(
    const void* where,
    const basic_simd<_SimdGeneration_, _DesiredType_, _RegisterPolicy_>& mask) noexcept
{
    return _SimdMaskLoadAligned<_SimdGeneration_, _RegisterPolicy_, vector_type, _DesiredType_>(
        reinterpret_cast<const _DesiredType_*>(where), mask._vector);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
template <typename _DesiredType_>
simd_stl_always_inline void basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::maskStoreUnaligned(
    void* where,
    const _Mask_type<_DesiredType_>    mask) const noexcept
{
    _SimdMaskStoreUnaligned<_SimdGeneration_, _RegisterPolicy_, _DesiredType_>(
        reinterpret_cast<_DesiredType_*>(where), mask, _vector);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
template <typename _DesiredType_>
simd_stl_always_inline void basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::maskStoreAligned(
    void* where,
    const _Mask_type<_DesiredType_>    mask) const noexcept
{
    _SimdMaskStoreAligned<_SimdGeneration_, _RegisterPolicy_, _DesiredType_>(
        reinterpret_cast<_DesiredType_*>(where), mask, _vector);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
template <typename _DesiredType_>
simd_stl_always_inline void basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::maskStoreUnaligned(
    void* where,
    const basic_simd<_SimdGeneration_, _DesiredType_, _RegisterPolicy_>& mask) const noexcept
{
    _SimdMaskStoreAligned<_SimdGeneration_, _RegisterPolicy_, _DesiredType_>(
        reinterpret_cast<_DesiredType_*>(where), mask._vector, _vector);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
template <typename _DesiredType_>
simd_stl_always_inline void basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::maskStoreAligned(
    void* where,
    const basic_simd<_SimdGeneration_, _DesiredType_, _RegisterPolicy_>& mask) const noexcept
{
    _SimdMaskStoreAligned<_SimdGeneration_, _RegisterPolicy_, _DesiredType_>(
        reinterpret_cast<_DesiredType_*>(where), mask._vector, _vector);
}


template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
template <typename _DesiredType_>
simd_stl_always_inline void basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::maskBlendStoreUnaligned(
    void* where,
    const _Mask_type<_DesiredType_>   mask,
    const basic_simd& source) const noexcept
{
    return _SimdMaskBlendStoreUnaligned<_SimdGeneration_, _RegisterPolicy_, _DesiredType_>(where, mask, _vector, source._vector);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
template <typename _DesiredType_>
simd_stl_always_inline void basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::maskBlendStoreAligned(
    void* where,
    const _Mask_type<_DesiredType_>   mask,
    const basic_simd& source) const noexcept
{
    return _SimdMaskBlendStoreAligned<_SimdGeneration_, _RegisterPolicy_, _DesiredType_>(where, mask, _vector, source._vector);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
template <typename _DesiredType_>
simd_stl_always_inline void basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::maskBlendStoreUnaligned(
    void* where,
    const basic_simd& mask,
    const basic_simd& source) const noexcept
{
    return _SimdMaskBlendStoreUnaligned<_SimdGeneration_, _RegisterPolicy_, _DesiredType_>(where, mask._vector, _vector, source._vector);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
template <typename _DesiredType_>
simd_stl_always_inline void basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::maskBlendStoreAligned(
    void* where,
    const basic_simd& mask,
    const basic_simd& source) const noexcept
{
    return _SimdMaskBlendStoreAligned<_SimdGeneration_, _RegisterPolicy_, _DesiredType_>(where, mask, _vector, source._vector);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_> operator>>(
    const basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>     left,
    const uint32                                                        shift) noexcept
{
    return _SimdShiftRightElements<_SimdGeneration_, _RegisterPolicy_, _Element_>(left._vector, shift);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_> operator<<(
    const basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>     left,
    const uint32                                                        shift) noexcept
{
    return _SimdShiftLeftElements<_SimdGeneration_, _RegisterPolicy_, _Element_>(left._vector, shift);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>&
basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::operator>>=(const uint32 shift) noexcept {
    return *this = (*this >> shift);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>&
basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::operator<<=(const uint32 shift) noexcept {
    return *this = (*this << shift);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
simd_stl_always_inline bool operator==(
    const basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>& left,
    const basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>& right) noexcept
{
    return left.isEqual(right);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
template <typename _DesiredType_>
simd_stl_always_inline bool basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::isEqual(const basic_simd& right) const noexcept {
    const auto _Mask = _SimdMaskCompare<_SimdGeneration_, _RegisterPolicy_, _DesiredType_, type_traits::equal_to<>>(_vector, right._vector);
    return basic_simd_mask<_SimdGeneration_, _DesiredType_, _RegisterPolicy_>(_Mask).allOf();
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
template <typename _DesiredType_>
simd_stl_always_inline basic_simd_mask<_SimdGeneration_, _DesiredType_, _RegisterPolicy_>
basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::maskNotEqual(const basic_simd& right) const noexcept
{
    return _SimdMaskCompare<_SimdGeneration_, _RegisterPolicy_, _DesiredType_, type_traits::not_equal_to<>>(_vector, right._vector);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
template <typename _DesiredType_>
simd_stl_always_inline basic_simd_mask<_SimdGeneration_, _DesiredType_, _RegisterPolicy_>
basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::maskEqual(const basic_simd& right) const noexcept
{
    return _SimdMaskCompare<_SimdGeneration_, _RegisterPolicy_, _DesiredType_, type_traits::equal_to<>>(_vector, right._vector);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
template <typename _DesiredType_>
simd_stl_always_inline basic_simd_mask<_SimdGeneration_, _DesiredType_, _RegisterPolicy_>
basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::maskGreater(const basic_simd& right) const noexcept
{
    return _SimdMaskCompare<_SimdGeneration_, _RegisterPolicy_, _DesiredType_, type_traits::greater<>>(_vector, right._vector);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
template <typename _DesiredType_>
simd_stl_always_inline basic_simd_mask<_SimdGeneration_, _DesiredType_, _RegisterPolicy_>
basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::maskLess(const basic_simd& right) const noexcept
{
    return _SimdMaskCompare<_SimdGeneration_, _RegisterPolicy_, _DesiredType_, type_traits::less<>>(_vector, right._vector);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
template <typename _DesiredType_>
simd_stl_always_inline basic_simd_mask<_SimdGeneration_, _DesiredType_, _RegisterPolicy_>
basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::maskGreaterEqual(const basic_simd& right) const noexcept
{
    return _SimdMaskCompare<_SimdGeneration_, _RegisterPolicy_, _DesiredType_, type_traits::greater_equal<>>(_vector, right._vector);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
template <typename _DesiredType_>
simd_stl_always_inline basic_simd_mask<_SimdGeneration_, _DesiredType_, _RegisterPolicy_>
basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::maskLessEqual(const basic_simd& right) const noexcept
{
    return _SimdMaskCompare<_SimdGeneration_, _RegisterPolicy_, _DesiredType_, type_traits::less_equal<>>(_vector, right._vector);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
template <typename _DesiredType_>
simd_stl_always_inline basic_simd<_SimdGeneration_, _DesiredType_, _RegisterPolicy_>
basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::notEqual(const basic_simd& right) const noexcept
{
    return _SimdCompare<_SimdGeneration_, _RegisterPolicy_, _DesiredType_, type_traits::not_equal_to<>>(_vector, right._vector);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
template <typename _DesiredType_>
simd_stl_always_inline basic_simd<_SimdGeneration_, _DesiredType_, _RegisterPolicy_>
basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::equal(const basic_simd& right) const noexcept
{
    return _SimdCompare<_SimdGeneration_, _RegisterPolicy_, _DesiredType_, type_traits::equal_to<>>(_vector, right._vector);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
template <typename _DesiredType_>
simd_stl_always_inline basic_simd<_SimdGeneration_, _DesiredType_, _RegisterPolicy_>
basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::greater(const basic_simd& right) const noexcept
{
    return _SimdCompare<_SimdGeneration_, _RegisterPolicy_, _DesiredType_, type_traits::greater<>>(_vector, right._vector);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
template <typename _DesiredType_>
simd_stl_always_inline basic_simd<_SimdGeneration_, _DesiredType_, _RegisterPolicy_>
basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::less(const basic_simd& right) const noexcept
{
    return _SimdCompare<_SimdGeneration_, _RegisterPolicy_, _DesiredType_, type_traits::less<>>(_vector, right._vector);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
template <typename _DesiredType_>
simd_stl_always_inline basic_simd<_SimdGeneration_, _DesiredType_, _RegisterPolicy_>
basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::greaterEqual(const basic_simd& right) const noexcept
{
    return _SimdCompare<_SimdGeneration_, _RegisterPolicy_, _DesiredType_, type_traits::greater_equal<>>(_vector, right._vector);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
template <typename _DesiredType_>
simd_stl_always_inline basic_simd<_SimdGeneration_, _DesiredType_, _RegisterPolicy_>
basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::lessEqual(const basic_simd& right) const noexcept
{
    return _SimdCompare<_SimdGeneration_, _RegisterPolicy_, _DesiredType_, type_traits::less_equal<>>(_vector, right._vector);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
template <typename _DesiredType_>
simd_stl_always_inline _Native_compare_return_type<basic_simd<_SimdGeneration_, _Element_,
    _RegisterPolicy_>, _DesiredType_, type_traits::not_equal_to<>>
    basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::nativeNotEqual(const basic_simd& right) const noexcept
{
    return _SimdNativeCompare<_SimdGeneration_, _RegisterPolicy_, _DesiredType_,
        type_traits::not_equal_to<>>(_vector, right._vector);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
template <typename _DesiredType_>
simd_stl_always_inline _Native_compare_return_type<basic_simd<_SimdGeneration_, _Element_,
    _RegisterPolicy_>, _DesiredType_, type_traits::equal_to<>>
    basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::nativeEqual(const basic_simd& right) const noexcept
{
    return _SimdNativeCompare<_SimdGeneration_, _RegisterPolicy_, _DesiredType_,
        type_traits::equal_to<>>(_vector, right._vector);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
template <typename _DesiredType_>
simd_stl_always_inline _Native_compare_return_type<basic_simd<_SimdGeneration_, _Element_,
    _RegisterPolicy_>, _DesiredType_, type_traits::greater<>>
    basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::nativeGreater(const basic_simd& right) const noexcept
{
    return _SimdNativeCompare<_SimdGeneration_, _RegisterPolicy_, _DesiredType_,
        type_traits::greater<>>(_vector, right._vector);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
template <typename _DesiredType_>
simd_stl_always_inline _Native_compare_return_type<basic_simd<_SimdGeneration_, _Element_,
    _RegisterPolicy_>, _DesiredType_, type_traits::less<>>
    basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::nativeLess(const basic_simd& right) const noexcept
{
    return _SimdNativeCompare<_SimdGeneration_, _RegisterPolicy_, _DesiredType_,
        type_traits::less<>>(_vector, right._vector);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
template <typename _DesiredType_>
simd_stl_always_inline _Native_compare_return_type<basic_simd<_SimdGeneration_, _Element_,
    _RegisterPolicy_>, _DesiredType_, type_traits::greater_equal<>>
    basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::nativeGreaterEqual(const basic_simd& right) const noexcept
{
    return _SimdNativeCompare<_SimdGeneration_, _RegisterPolicy_, _DesiredType_,
        type_traits::greater_equal<>>(_vector, right._vector);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
template <typename _DesiredType_>
simd_stl_always_inline _Native_compare_return_type<basic_simd<_SimdGeneration_, _Element_,
    _RegisterPolicy_>, _DesiredType_, type_traits::less_equal<>>
    basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::nativeLessEqual(const basic_simd& right) const noexcept
{
    return _SimdNativeCompare<_SimdGeneration_, _RegisterPolicy_, _DesiredType_,
        type_traits::less_equal<>>(_vector, right._vector);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
template <typename _DesiredType_>
simd_stl_always_inline basic_simd_mask<_SimdGeneration_, _DesiredType_, _RegisterPolicy_>
basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::toMask() const noexcept
{
    return _SimdToMask<_SimdGeneration_, _RegisterPolicy_, _DesiredType_>(_vector);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
template <typename _DesiredType_>
simd_stl_always_inline _Reduce_type<_DesiredType_> basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::reduce() const noexcept {
    return _SimdReduce<_SimdGeneration_, _RegisterPolicy_, _DesiredType_>(_vector);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
template <typename _ElementType_>
constexpr int basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::width() noexcept {
    static_assert(type_traits::__is_vector_type_supported_v<_ElementType_>, "Unsupported element type");

    constexpr auto width = sizeof(vector_type);
    return width;
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
template <typename _ElementType_>
constexpr int basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::size() noexcept {
    static_assert(type_traits::__is_vector_type_supported_v<_ElementType_>, "Unsupported element type");

    constexpr auto length = (sizeof(vector_type) / sizeof(_ElementType_));
    return length;
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
template <typename _ElementType_>
constexpr int basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::length() noexcept {
    return size<_ElementType_>();
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
constexpr int basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::registersCount() noexcept {
    if      constexpr (arch::__is_xmm_v<_SimdGeneration_>)
        return 8;
    else if constexpr (arch::__is_ymm_v<_SimdGeneration_>)
        return 16;
    else if constexpr (arch::__is_zmm_v<_SimdGeneration_>)
        return 32;
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::vector_type
basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::unwrap() const noexcept
{
    return _vector;
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
_Make_tail_mask_return_type<basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>>
basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::makeTailMask(uint32 bytes) noexcept
{
    return _SimdMakeTailMask<_SimdGeneration_, _RegisterPolicy_, _Element_>(bytes);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
void basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::streamingFence() noexcept {
    return _SimdStreamingFence<_SimdGeneration_>();
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_> basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::nonTemporalLoad(const void* where) noexcept {
    return _SimdNonTemporalLoad<_SimdGeneration_, _RegisterPolicy_, vector_type>(where);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
void basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::nonTemporalStore(void* where) const noexcept {
    _SimdNonTemporalStore<_SimdGeneration_, _RegisterPolicy_>(where, _vector);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
void basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::zeroUpper() noexcept {
    if constexpr (type_traits::is_zeroupper_required_v<_SimdGeneration_>)
        _mm256_zeroupper();
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
template <typename _DesiredType_>
_DesiredType_* basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::compressStoreUnaligned(
    void* where,
    _Mask_type<_DesiredType_>   mask) const noexcept
{
    return _SimdCompressStoreUnaligned<_SimdGeneration_, _RegisterPolicy_, _DesiredType_>(
        reinterpret_cast<_DesiredType_*>(where), mask, _vector);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
template <typename _DesiredType_>
_DesiredType_* basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::compressStoreAligned(
    void* where,
    _Mask_type<_DesiredType_>   mask) const noexcept
{
    return _SimdCompressStoreAligned<_SimdGeneration_, _RegisterPolicy_, _DesiredType_>(
        reinterpret_cast<_DesiredType_*>(where), mask, _vector);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
template <typename _DesiredType_>
simd_stl_always_inline void basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::blend(
    const basic_simd<_SimdGeneration_, _DesiredType_, _RegisterPolicy_>& _Vector,
    _Mask_type<_DesiredType_>                                               _Mask) noexcept
{
    _vector = _IntrinBitcast<vector_type>(_SimdBlend<_SimdGeneration_, _RegisterPolicy_, _DesiredType_>(
        _IntrinBitcast<decltype(_Vector._vector)>(_vector), _Vector._vector, _Mask));
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
template <typename _DesiredType_>
simd_stl_always_inline void basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::blend(
    const basic_simd<_SimdGeneration_, _DesiredType_, _RegisterPolicy_>& _Vector,
    const basic_simd<_SimdGeneration_, _DesiredType_, _RegisterPolicy_>& _Mask) noexcept
{
    _vector = _IntrinBitcast<vector_type>(_SimdBlend<_SimdGeneration_, _RegisterPolicy_, _DesiredType_>(
        _IntrinBitcast<decltype(_Vector._vector)>(_vector), _Vector._vector, _Mask._vector));
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
template <typename _DesiredType_>
simd_stl_always_inline basic_simd<_SimdGeneration_, _DesiredType_, _RegisterPolicy_> 
    basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::min(
        const basic_simd<_SimdGeneration_, _DesiredType_, _RegisterPolicy_>& _Other) const noexcept 
{
    return _SimdMin<_SimdGeneration_, _RegisterPolicy_, _DesiredType_>(_vector, _Other.unwrap());
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
template <typename _DesiredType_>
simd_stl_always_inline _DesiredType_ basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::min() const noexcept {
    return _SimdMin<_SimdGeneration_, _RegisterPolicy_, _DesiredType_>(_vector);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
template <typename _DesiredType_>
simd_stl_always_inline basic_simd<_SimdGeneration_, _DesiredType_, _RegisterPolicy_> 
    basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::max(
        const basic_simd<_SimdGeneration_, _DesiredType_, _RegisterPolicy_>& _Other) const noexcept
{
    return _SimdMax<_SimdGeneration_, _RegisterPolicy_, _DesiredType_>(_vector, _Other.unwrap());
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
template <typename _DesiredType_>
simd_stl_always_inline _DesiredType_ basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::max() const noexcept {
    return _SimdMax<_SimdGeneration_, _RegisterPolicy_, _DesiredType_>(_vector);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
template <typename _DesiredType_>
simd_stl_always_inline basic_simd<_SimdGeneration_, _DesiredType_, _RegisterPolicy_> 
    basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::abs() const noexcept 
{
    return _SimdAbs<_SimdGeneration_, _RegisterPolicy_, _DesiredType_>(_vector);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
template <typename _DesiredType_>
void basic_simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::reverse() noexcept {
    _vector = _SimdReverse<_SimdGeneration_, _RegisterPolicy_, _DesiredType_>(_vector);
}

template <arch::CpuFeature _SimdGeneration_>
zero_upper_at_exit_guard<_SimdGeneration_>::zero_upper_at_exit_guard() noexcept
{}

template <arch::CpuFeature _SimdGeneration_>
zero_upper_at_exit_guard<_SimdGeneration_>::~zero_upper_at_exit_guard() noexcept {
    basic_simd<_SimdGeneration_, int>::zeroUpper();
}

__SIMD_STL_NUMERIC_NAMESPACE_END
