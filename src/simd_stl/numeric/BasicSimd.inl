#pragma once 

__SIMD_STL_NUMERIC_NAMESPACE_BEGIN

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::simd() noexcept
{}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
template <
    typename _VectorType_,
    std::enable_if_t<_Is_intrin_type_v<_VectorType_> || _Is_valid_basic_simd_v<_VectorType_>, int>>
simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::simd(_VectorType_ _Other) noexcept {
    _vector = simd_cast<vector_type>(_Other);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::simd(value_type _Value) noexcept {
    fill<value_type>(_Value);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::~simd() noexcept
{}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
simd_stl_always_inline bool simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::is_supported() noexcept {
    return arch::ProcessorFeatures::isSupported<_SimdGeneration_>();
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
simd_stl_always_inline simd<_SimdGeneration_, _Element_, _RegisterPolicy_>& simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::clear() noexcept {
    return *this = _SimdBroadcastZeros<_SimdGeneration_, _RegisterPolicy_, vector_type>();
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
template <typename _DesiredType_>
simd_stl_always_inline simd<_SimdGeneration_, _Element_, _RegisterPolicy_>& simd<_SimdGeneration_, _Element_, _RegisterPolicy_>
    ::fill(typename std::type_identity<_DesiredType_>::type _Value) noexcept
{
    _vector = _SimdBroadcast<_SimdGeneration_, _RegisterPolicy_, vector_type>(_Value);
    return *this;
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
simd_stl_always_inline simd<_SimdGeneration_, _Element_, _RegisterPolicy_>&
    simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::operator+=(const simd& _Other) noexcept
{
    return *this = (*this + _Other);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
simd_stl_always_inline simd<_SimdGeneration_, _Element_, _RegisterPolicy_>&
    simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::operator-=(const simd& _Other) noexcept
{
    return *this = (*this - _Other);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
simd_stl_always_inline simd<_SimdGeneration_, _Element_, _RegisterPolicy_>&
    simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::operator*=(const simd& _Other) noexcept
{
    return *this = (*this * _Other);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
simd_stl_always_inline simd<_SimdGeneration_, _Element_, _RegisterPolicy_>&
    simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::operator/=(const simd& _Other) noexcept
{
    return *this = (*this / _Other);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
simd_stl_always_inline simd<_SimdGeneration_, _Element_, _RegisterPolicy_>&
    simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::operator=(const simd& _Left) noexcept
{
    _vector = _Left._vector;
    return *this;
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
simd_stl_always_inline _Element_
    simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::operator[](const size_type _Index) const noexcept 
{
    return _SimdExtract<_SimdGeneration_, _RegisterPolicy_, _Element_>(_vector, _Index);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
simd_stl_always_inline BasicSimdElementReference<simd<_SimdGeneration_, _Element_, _RegisterPolicy_>>
    simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::operator[](const size_type _Index) noexcept
{
    return BasicSimdElementReference<simd>(this, _Index);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
simd_stl_always_inline simd<_SimdGeneration_, _Element_, _RegisterPolicy_>
simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::operator++(int) noexcept {
    simd _Self = *this;
    *this += simd(1);
    return _Self;
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
simd_stl_always_inline simd<_SimdGeneration_, _Element_, _RegisterPolicy_>&
simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::operator++() noexcept {
    return *this += simd(1);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
simd_stl_always_inline simd<_SimdGeneration_, _Element_, _RegisterPolicy_>
simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::operator--(int) noexcept
{
    simd _Self = *this;
    *this -= simd(1);
    return _Self;
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
simd_stl_always_inline simd<_SimdGeneration_, _Element_, _RegisterPolicy_>&
simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::operator--() noexcept
{
    return *this -= simd(1);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
simd_stl_always_inline simd<_SimdGeneration_, _Element_, _RegisterPolicy_>
simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::operator~() const noexcept {
    return _SimdBitNot<_SimdGeneration_, _RegisterPolicy_>(_vector);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
simd_stl_always_inline simd<_SimdGeneration_, _Element_, _RegisterPolicy_>&
simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::operator&=(const simd& _Other) noexcept {
    return *this = (*this & _Other);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
simd_stl_always_inline simd<_SimdGeneration_, _Element_, _RegisterPolicy_>&
simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::operator|=(const simd& _Other) noexcept {
    return *this = (*this | _Other);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
simd_stl_always_inline simd<_SimdGeneration_, _Element_, _RegisterPolicy_>&
simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::operator^=(const simd& _Other) noexcept {
    return *this = (*this ^ _Other);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
simd_stl_always_inline simd<_SimdGeneration_, _Element_, _RegisterPolicy_> operator/(
    const simd<_SimdGeneration_, _Element_, _RegisterPolicy_>& _Left,
    const simd<_SimdGeneration_, _Element_, _RegisterPolicy_>& _Right) noexcept
{
    return _SimdDivide<_SimdGeneration_, _RegisterPolicy_, _Element_>(_Left._vector, _Right._vector);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
simd_stl_always_inline simd<_SimdGeneration_, _Element_, _RegisterPolicy_> operator+(
    const simd<_SimdGeneration_, _Element_, _RegisterPolicy_>& _Left,
    const simd<_SimdGeneration_, _Element_, _RegisterPolicy_>& _Right) noexcept
{
    return _SimdAdd<_SimdGeneration_, _RegisterPolicy_, _Element_>(_Left._vector, _Right._vector);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
simd_stl_always_inline simd<_SimdGeneration_, _Element_, _RegisterPolicy_> operator-(
    const simd<_SimdGeneration_, _Element_, _RegisterPolicy_>& _Left,
    const simd<_SimdGeneration_, _Element_, _RegisterPolicy_>& _Right) noexcept
{
    return _SimdSubstract<_SimdGeneration_, _RegisterPolicy_, _Element_>(_Left._vector, _Right._vector);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
simd_stl_always_inline simd<_SimdGeneration_, _Element_, _RegisterPolicy_> operator*(
    const simd<_SimdGeneration_, _Element_, _RegisterPolicy_>& _Left,
    const simd<_SimdGeneration_, _Element_, _RegisterPolicy_>& _Right) noexcept
{
    return _SimdMultiply<_SimdGeneration_, _RegisterPolicy_, _Element_>(_Left._vector, _Right._vector);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
simd_stl_always_inline simd<_SimdGeneration_, _Element_, _RegisterPolicy_> operator&(
    const simd<_SimdGeneration_, _Element_, _RegisterPolicy_>& _Left,
    const simd<_SimdGeneration_, _Element_, _RegisterPolicy_>& _Right) noexcept
{
    return _SimdBitAnd<_SimdGeneration_, _RegisterPolicy_>(_Left._vector, _Right._vector);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
simd_stl_always_inline simd<_SimdGeneration_, _Element_, _RegisterPolicy_> operator|(
    const simd<_SimdGeneration_, _Element_, _RegisterPolicy_>& _Left,
    const simd<_SimdGeneration_, _Element_, _RegisterPolicy_>& _Right) noexcept
{
    return _SimdBitOr<_SimdGeneration_, _RegisterPolicy_>(_Left._vector, _Right._vector);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
simd_stl_always_inline simd<_SimdGeneration_, _Element_, _RegisterPolicy_> operator^(
    const simd<_SimdGeneration_, _Element_, _RegisterPolicy_>& _Left,
    const simd<_SimdGeneration_, _Element_, _RegisterPolicy_>& _Right) noexcept
{
    return _SimdBitXor<_SimdGeneration_, _RegisterPolicy_>(_Left._vector, _Right._vector);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
simd_stl_always_inline simd<_SimdGeneration_, _Element_, _RegisterPolicy_> operator+(
    const simd<_SimdGeneration_, _Element_, _RegisterPolicy_>&                      _Left,
    const typename simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::value_type  _Right) noexcept
{
    return *this + simd(_Right);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
simd_stl_always_inline simd<_SimdGeneration_, _Element_, _RegisterPolicy_> operator-(
    const simd<_SimdGeneration_, _Element_, _RegisterPolicy_>& _Left,
    const typename simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::value_type  _Right) noexcept
{
    return _SimdSubstract<_SimdGeneration_, _RegisterPolicy_, _Element_>(
        _Left._vector, _SimdBroadcast<_SimdGeneration_, _RegisterPolicy_, 
            typename simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::vector_type>(_Right));
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
simd_stl_always_inline simd<_SimdGeneration_, _Element_, _RegisterPolicy_> operator*(
    const simd<_SimdGeneration_, _Element_, _RegisterPolicy_>& _Left,
    const typename simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::value_type  _Right) noexcept
{
    return _SimdMultiply<_SimdGeneration_, _RegisterPolicy_, _Element_>(
        _Left._vector, _SimdBroadcast<_SimdGeneration_, _RegisterPolicy_, 
            typename simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::vector_type>(_Right));
}


template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
simd_stl_always_inline simd<_SimdGeneration_, _Element_, _RegisterPolicy_> operator/(
    const simd<_SimdGeneration_, _Element_, _RegisterPolicy_>&                      _Left,
    const typename simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::value_type  _Right) noexcept
{
    return _SimdDivide<_SimdGeneration_, _RegisterPolicy_, _Element_>(
        _Left._vector, _SimdBroadcast<_SimdGeneration_, _RegisterPolicy_, 
            typename simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::vector_type>(_Right));
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
simd_stl_always_inline simd<_SimdGeneration_, _Element_, _RegisterPolicy_>
    simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::operator+() const noexcept 
{
    return _vector;
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
simd_stl_always_inline simd<_SimdGeneration_, _Element_, _RegisterPolicy_>
    simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::operator-() const noexcept
{
    return _SimdNegate<_SimdGeneration_, _RegisterPolicy_, _Element_>(_vector);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
template <typename _DesiredType_>
simd_stl_always_inline _DesiredType_
    simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::extract(size_type _Index) const noexcept
{
    DebugAssert(_Index >= 0 && _Index < size<_DesiredType_>(), "simd_stl::numeric::basic_simd: Index out of range");
    return _SimdExtract<_SimdGeneration_, _RegisterPolicy_, _DesiredType_>(_vector, _Index);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
template <typename _DesiredType_>
simd_stl_always_inline BasicSimdElementReference<simd<_SimdGeneration_, _Element_, _RegisterPolicy_>>
    simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::extract_wrapped(size_type _Index) noexcept
{
    DebugAssert(_Index >= 0 && _Index < size<_DesiredType_>(), "simd_stl::numeric::basic_simd: Index out of range");
    return BasicSimdElementReference<simd>(this, _Index);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
template <typename _DesiredType_>
simd_stl_always_inline void simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::insert(
    const size_type                                         where,
    const typename std::type_identity<_DesiredType_>::type  value) noexcept
{
    return _SimdInsert<_SimdGeneration_, _RegisterPolicy_, _DesiredType_>(_vector, where, value);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
template <class _AlignmentPolicy_>
simd_stl_always_inline simd<_SimdGeneration_, _Element_, _RegisterPolicy_>
simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::load(const void* _Address, _AlignmentPolicy_&&)  noexcept
{
    if constexpr (std::remove_cvref_t<_AlignmentPolicy_>::_Alignment)
        return _SimdLoadAligned<_SimdGeneration_, _RegisterPolicy_, vector_type>(_Address);
    else 
        return _SimdLoadUnaligned<_SimdGeneration_, _RegisterPolicy_, vector_type>(_Address);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
template <class _AlignmentPolicy_>
simd_stl_always_inline void simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::store(void* _Address, _AlignmentPolicy_&&) const noexcept {
    if constexpr (std::remove_cvref_t<_AlignmentPolicy_>::_Alignment)
        return _SimdStoreAligned<_SimdGeneration_, _RegisterPolicy_, vector_type>(_Address, _vector);
    else
        return _SimdStoreUnaligned<_SimdGeneration_, _RegisterPolicy_, vector_type>(_Address, _vector);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
template <
    typename    _DesiredType_,
    class       _MaskType_,
    class       _AlignmentPolicy_>
simd_stl_always_inline simd<_SimdGeneration_, _Element_, _RegisterPolicy_>
    simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::mask_load(
        const void*         _Address,
        const _MaskType_&   _Mask,
        _AlignmentPolicy_&&) noexcept
{
    if constexpr (std::remove_cvref_t<_AlignmentPolicy_>::_Alignment)
        return _SimdMaskLoadAligned<_SimdGeneration_, _RegisterPolicy_, vector_type, _DesiredType_>(
            reinterpret_cast<const _DesiredType_*>(_Address), _SimdUnwrapMask(_Mask), _SimdUnwrap(simd(0)));
    else 
        return _SimdMaskLoadUnaligned<_SimdGeneration_, _RegisterPolicy_, vector_type, _DesiredType_>(
            reinterpret_cast<const _DesiredType_*>(_Address), _SimdUnwrapMask(_Mask), _SimdUnwrap(simd(0)));
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
template <
    typename    _DesiredType_,
    class       _MaskType_,
    class       _VectorType_,
    class       _AlignmentPolicy_,
    std::enable_if_t<_Is_intrin_type_v<_VectorType_> || _Is_valid_basic_simd_v<_VectorType_>, int>>
simd_stl_always_inline simd<_SimdGeneration_, _Element_, _RegisterPolicy_> simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::mask_load(
    const void*         _Address,
    const _MaskType_&   _Mask,
    _VectorType_        _AdditionalSource,
    _AlignmentPolicy_&&) noexcept
{
    if constexpr (std::remove_cvref_t<_AlignmentPolicy_>::_Alignment)
        return _SimdMaskLoadAligned<_SimdGeneration_, _RegisterPolicy_, vector_type, _DesiredType_>(
            reinterpret_cast<const _DesiredType_*>(_Address), _SimdUnwrapMask(_Mask), _SimdUnwrap(_AdditionalSource));
    else
        return _SimdMaskLoadUnaligned<_SimdGeneration_, _RegisterPolicy_, vector_type, _DesiredType_>(
            reinterpret_cast<const _DesiredType_*>(_Address), _SimdUnwrapMask(_Mask), _SimdUnwrap(_AdditionalSource));
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
template <
    typename    _DesiredType_,
    class       _MaskType_,
    class       _AlignmentPolicy_>
simd_stl_always_inline void simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::mask_store(
    void*               _Address,
    const _MaskType_&   _Mask,
    _AlignmentPolicy_&&) const noexcept
{
    if constexpr (std::remove_cvref_t<_AlignmentPolicy_>::_Alignment)
        _SimdMaskStoreAligned<_SimdGeneration_, _RegisterPolicy_, _DesiredType_>(
            reinterpret_cast<_DesiredType_*>(_Address), _SimdUnwrapMask(_Mask), _vector);
    else
        _SimdMaskStoreUnaligned<_SimdGeneration_, _RegisterPolicy_, _DesiredType_>(
            reinterpret_cast<_DesiredType_*>(_Address), _SimdUnwrapMask(_Mask), _vector);
}

//template <
//    arch::CpuFeature	_SimdGeneration_,
//    typename			_Element_,
//    class               _RegisterPolicy_>
//simd_stl_always_inline simd<_SimdGeneration_, _Element_, _RegisterPolicy_> operator>>(
//    const simd<_SimdGeneration_, _Element_, _RegisterPolicy_>     _Left,
//    const uint32                                                  _Shift) noexcept
//{
//    return _SimdShiftRightElements<_SimdGeneration_, _RegisterPolicy_, _Element_>(_Left._vector, _Shift);
//}
//
//template <
//    arch::CpuFeature	_SimdGeneration_,
//    typename			_Element_,
//    class               _RegisterPolicy_>
//simd_stl_always_inline simd<_SimdGeneration_, _Element_, _RegisterPolicy_> operator<<(
//    const simd<_SimdGeneration_, _Element_, _RegisterPolicy_>   _Left,
//    const uint32                                                _Shift) noexcept
//{
//    return _SimdShiftLeftElements<_SimdGeneration_, _RegisterPolicy_, _Element_>(_Left._vector, _Shift);
//}
//
//template <
//    arch::CpuFeature	_SimdGeneration_,
//    typename			_Element_,
//    class               _RegisterPolicy_>
//simd_stl_always_inline simd<_SimdGeneration_, _Element_, _RegisterPolicy_>&
//simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::operator>>=(const uint32 shift) noexcept {
//    return *this = (*this >> shift);
//}
//
//template <
//    arch::CpuFeature	_SimdGeneration_,
//    typename			_Element_,
//    class               _RegisterPolicy_>
//simd_stl_always_inline simd<_SimdGeneration_, _Element_, _RegisterPolicy_>&
//simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::operator<<=(const uint32 shift) noexcept {
//    return *this = (*this << shift);
//}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
simd_stl_always_inline bool operator==(
    const simd<_SimdGeneration_, _Element_, _RegisterPolicy_>& _Left,
    const simd<_SimdGeneration_, _Element_, _RegisterPolicy_>& _Right) noexcept
{
    return _Left.mask_compare(_Right, type_traits::equal_to<>{}).allOf();
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
simd_stl_always_inline bool operator!=(
    const simd<_SimdGeneration_, _Element_, _RegisterPolicy_>& _Left,
    const simd<_SimdGeneration_, _Element_, _RegisterPolicy_>& _Right) noexcept
{
    return !(_Left == _Right);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
template <
    class       _Predicate_,
    typename    _DesiredType_>
simd_stl_always_inline simd_mask<_SimdGeneration_, _DesiredType_, _RegisterPolicy_>
    simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::mask_compare(const simd& _Right, _Predicate_&&) const noexcept
{
    return _SimdMaskCompare<_SimdGeneration_, _RegisterPolicy_, _DesiredType_, _Predicate_>(_vector, _Right._vector);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
template <
    class       _Predicate_,
    typename    _DesiredType_>
simd_stl_always_inline simd<_SimdGeneration_, _DesiredType_, _RegisterPolicy_>
    simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::vector_compare(const simd& _Right, _Predicate_&&) const noexcept
{
    return _SimdCompare<_SimdGeneration_, _RegisterPolicy_, _DesiredType_, _Predicate_>(_vector, _Right._vector);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
template <
    class       _Predicate_,
    typename    _DesiredType_>
simd_stl_always_inline _Native_compare_return_type<simd<_SimdGeneration_, _Element_,
    _RegisterPolicy_>, _DesiredType_, _Predicate_>
    simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::native_compare(const simd& _Right, _Predicate_&&) const noexcept
{
    return _SimdNativeCompare<_SimdGeneration_, _RegisterPolicy_, _DesiredType_, _Predicate_>(_vector, _Right._vector);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
template <typename _DesiredType_>
simd_stl_always_inline simd_mask<_SimdGeneration_, _DesiredType_, _RegisterPolicy_>
simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::to_mask() const noexcept
{
    return _SimdToMask<_SimdGeneration_, _RegisterPolicy_, _DesiredType_>(_vector);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
template <typename _DesiredType_>
simd_stl_always_inline _Reduce_type<_DesiredType_> simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::reduce_add() const noexcept {
    return _SimdReduce<_SimdGeneration_, _RegisterPolicy_, _DesiredType_>(_vector);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
template <typename _ElementType_>
constexpr int simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::width() noexcept {
    return sizeof(vector_type);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
template <typename _ElementType_>
constexpr int simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::size() noexcept {
    static_assert(type_traits::__is_vector_type_supported_v<_ElementType_>, "Unsupported element type");

    constexpr auto length = (sizeof(vector_type) / sizeof(_ElementType_));
    return length;
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
template <typename _ElementType_>
constexpr int simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::length() noexcept {
    return size<_ElementType_>();
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
simd_stl_always_inline simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::vector_type
simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::unwrap() const noexcept
{
    return _vector;
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
_Make_tail_mask_return_type<simd<_SimdGeneration_, _Element_, _RegisterPolicy_>>
simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::make_tail_mask(uint32 bytes) noexcept
{
    return _SimdMakeTailMask<_SimdGeneration_, _RegisterPolicy_, _Element_>(bytes);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
void simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::streaming_fence() noexcept {
    return _SimdStreamingFence<_SimdGeneration_>();
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
simd<_SimdGeneration_, _Element_, _RegisterPolicy_> simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::non_temporal_load(const void* _Address) noexcept
{
    return _SimdNonTemporalLoad<_SimdGeneration_, _RegisterPolicy_, vector_type>(_Address);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
void simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::non_temporal_store(void* _Address) const noexcept {
    _SimdNonTemporalStore<_SimdGeneration_, _RegisterPolicy_>(_Address, _vector);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
    template <
        typename    _DesiredType_,
        class       _MaskType_,
        class       _AlignmentPolicy_>
_DesiredType_* simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::compress_store(
    void*               _Address,
    const _MaskType_&   _Mask,
    _AlignmentPolicy_&&) const noexcept
{
    if constexpr (std::remove_cvref_t<_AlignmentPolicy_>::_Alignment)
        return _SimdCompressStoreAligned<_SimdGeneration_, _RegisterPolicy_, _DesiredType_>(
            reinterpret_cast<_DesiredType_*>(_Address), _SimdToMask<_SimdGeneration_, _RegisterPolicy_, _DesiredType_>(_SimdUnwrapMask(_Mask)), _vector);
    else 
        return _SimdCompressStoreUnaligned<_SimdGeneration_, _RegisterPolicy_, _DesiredType_>(
            reinterpret_cast<_DesiredType_*>(_Address), _SimdToMask<_SimdGeneration_, _RegisterPolicy_, _DesiredType_>(_SimdUnwrapMask(_Mask)), _vector);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
template <
    class       _MaskType_,
    typename    _DesiredType_>
simd_stl_always_inline void simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::blend(
    const simd<_SimdGeneration_, _DesiredType_, _RegisterPolicy_>&  _Vector,
    const _MaskType_&                                               _Mask) noexcept
{
    _vector = simd_cast<vector_type>(_SimdBlend<_SimdGeneration_, _RegisterPolicy_, _DesiredType_>(
        simd_cast<decltype(_Vector._vector)>(_vector), _Vector._vector, _SimdUnwrapMask(_Mask)));
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
template <typename _DesiredType_>
simd_stl_always_inline simd<_SimdGeneration_, _DesiredType_, _RegisterPolicy_> 
    simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::vertical_min(
        const simd<_SimdGeneration_, _DesiredType_, _RegisterPolicy_>& _Other) const noexcept 
{
    return _SimdVerticalMin<_SimdGeneration_, _RegisterPolicy_, _DesiredType_>(_vector, _Other.unwrap());
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
template <typename _DesiredType_>
simd_stl_always_inline _DesiredType_ simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::horizontal_min() const noexcept {
    return _SimdHorizontalMin<_SimdGeneration_, _RegisterPolicy_, _DesiredType_>(_vector);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
template <typename _DesiredType_>
simd_stl_always_inline simd<_SimdGeneration_, _DesiredType_, _RegisterPolicy_> 
    simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::vertical_max(
        const simd<_SimdGeneration_, _DesiredType_, _RegisterPolicy_>& _Other) const noexcept
{
    return _SimdVerticalMax<_SimdGeneration_, _RegisterPolicy_, _DesiredType_>(_vector, _Other.unwrap());
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
template <typename _DesiredType_>
simd_stl_always_inline _DesiredType_ simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::horizontal_max() const noexcept {
    return _SimdHorizontalMax<_SimdGeneration_, _RegisterPolicy_, _DesiredType_>(_vector);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
template <typename _DesiredType_>
simd_stl_always_inline simd<_SimdGeneration_, _DesiredType_, _RegisterPolicy_> 
    simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::abs() const noexcept 
{
    return _SimdAbs<_SimdGeneration_, _RegisterPolicy_, _DesiredType_>(_vector);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
template <typename _DesiredType_>
void simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::reverse() noexcept {
    _vector = _SimdReverse<_SimdGeneration_, _RegisterPolicy_, _DesiredType_>(_vector);
}

template <arch::CpuFeature _SimdGeneration_>
zero_upper_at_exit_guard<_SimdGeneration_>::zero_upper_at_exit_guard() noexcept
{}

template <arch::CpuFeature _SimdGeneration_>
zero_upper_at_exit_guard<_SimdGeneration_>::~zero_upper_at_exit_guard() noexcept {
    if constexpr (type_traits::is_zeroupper_required_v<_SimdGeneration_>)
        _mm256_zeroupper();
}

__SIMD_STL_NUMERIC_NAMESPACE_END
