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
    std::enable_if_t<__is_intrin_type_v<_VectorType_> || __is_valid_basic_simd_v<_VectorType_>, int>>
simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::simd(_VectorType_ __other) noexcept {
    _vector = simd_cast<vector_type>(__other);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::simd(value_type __value) noexcept {
    fill<value_type>(__value);
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
    return *this = __simd_broadcast_zeros<_SimdGeneration_, _RegisterPolicy_, vector_type>();
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
template <typename _DesiredType_>
simd_stl_always_inline simd<_SimdGeneration_, _Element_, _RegisterPolicy_>& simd<_SimdGeneration_, _Element_, _RegisterPolicy_>
    ::fill(typename std::type_identity<_DesiredType_>::type __value) noexcept
{
    _vector = __simd_broadcast<_SimdGeneration_, _RegisterPolicy_, vector_type>(__value);
    return *this;
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
simd_stl_always_inline simd<_SimdGeneration_, _Element_, _RegisterPolicy_>&
    simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::operator+=(const simd& __other) noexcept
{
    return *this = (*this + __other);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
simd_stl_always_inline simd<_SimdGeneration_, _Element_, _RegisterPolicy_>&
    simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::operator-=(const simd& __other) noexcept
{
    return *this = (*this - __other);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
simd_stl_always_inline simd<_SimdGeneration_, _Element_, _RegisterPolicy_>&
    simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::operator*=(const simd& __other) noexcept
{
    return *this = (*this * __other);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
simd_stl_always_inline simd<_SimdGeneration_, _Element_, _RegisterPolicy_>&
    simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::operator/=(const simd& __other) noexcept
{
    return *this = (*this / __other);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
simd_stl_always_inline simd<_SimdGeneration_, _Element_, _RegisterPolicy_>&
    simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::operator=(const simd& __left) noexcept
{
    _vector = __left._vector;
    return *this;
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
simd_stl_always_inline _Element_
    simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::operator[](const size_type __index) const noexcept 
{
    return __simd_extract<_SimdGeneration_, _RegisterPolicy_, _Element_>(_vector, __index);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
simd_stl_always_inline simd_element_reference<simd<_SimdGeneration_, _Element_, _RegisterPolicy_>>
    simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::operator[](const size_type __index) noexcept
{
    return simd_element_reference<simd>(this, __index);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
simd_stl_always_inline simd<_SimdGeneration_, _Element_, _RegisterPolicy_>
simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::operator++(int) noexcept {
    simd __self = *this;
    *this += simd(1);
    return __self;
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
    simd __self = *this;
    *this -= simd(1);
    return __self;
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
    return __simd_bit_not<_SimdGeneration_, _RegisterPolicy_>(_vector);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
simd_stl_always_inline simd<_SimdGeneration_, _Element_, _RegisterPolicy_>&
simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::operator&=(const simd& __other) noexcept {
    return *this = (*this & __other);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
simd_stl_always_inline simd<_SimdGeneration_, _Element_, _RegisterPolicy_>&
simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::operator|=(const simd& __other) noexcept {
    return *this = (*this | __other);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
simd_stl_always_inline simd<_SimdGeneration_, _Element_, _RegisterPolicy_>&
simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::operator^=(const simd& __other) noexcept {
    return *this = (*this ^ __other);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
simd_stl_always_inline simd<_SimdGeneration_, _Element_, _RegisterPolicy_> operator/(
    const simd<_SimdGeneration_, _Element_, _RegisterPolicy_>& __left,
    const simd<_SimdGeneration_, _Element_, _RegisterPolicy_>& __right) noexcept
{
    return __simd_divide<_SimdGeneration_, _RegisterPolicy_, _Element_>(__left._vector, __right._vector);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
simd_stl_always_inline simd<_SimdGeneration_, _Element_, _RegisterPolicy_> operator+(
    const simd<_SimdGeneration_, _Element_, _RegisterPolicy_>& __left,
    const simd<_SimdGeneration_, _Element_, _RegisterPolicy_>& __right) noexcept
{
    return __simd_add<_SimdGeneration_, _RegisterPolicy_, _Element_>(__left._vector, __right._vector);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
simd_stl_always_inline simd<_SimdGeneration_, _Element_, _RegisterPolicy_> operator-(
    const simd<_SimdGeneration_, _Element_, _RegisterPolicy_>& __left,
    const simd<_SimdGeneration_, _Element_, _RegisterPolicy_>& __right) noexcept
{
    return __simd_substract<_SimdGeneration_, _RegisterPolicy_, _Element_>(__left._vector, __right._vector);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
simd_stl_always_inline simd<_SimdGeneration_, _Element_, _RegisterPolicy_> operator*(
    const simd<_SimdGeneration_, _Element_, _RegisterPolicy_>& __left,
    const simd<_SimdGeneration_, _Element_, _RegisterPolicy_>& __right) noexcept
{
    return __simd_multiply<_SimdGeneration_, _RegisterPolicy_, _Element_>(__left._vector, __right._vector);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
simd_stl_always_inline simd<_SimdGeneration_, _Element_, _RegisterPolicy_> operator&(
    const simd<_SimdGeneration_, _Element_, _RegisterPolicy_>& __left,
    const simd<_SimdGeneration_, _Element_, _RegisterPolicy_>& __right) noexcept
{
    return __simd_bit_and<_SimdGeneration_, _RegisterPolicy_>(__left._vector, __right._vector);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
simd_stl_always_inline simd<_SimdGeneration_, _Element_, _RegisterPolicy_> operator|(
    const simd<_SimdGeneration_, _Element_, _RegisterPolicy_>& __left,
    const simd<_SimdGeneration_, _Element_, _RegisterPolicy_>& __right) noexcept
{
    return __simd_bit_or<_SimdGeneration_, _RegisterPolicy_>(__left._vector, __right._vector);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
simd_stl_always_inline simd<_SimdGeneration_, _Element_, _RegisterPolicy_> operator^(
    const simd<_SimdGeneration_, _Element_, _RegisterPolicy_>& __left,
    const simd<_SimdGeneration_, _Element_, _RegisterPolicy_>& __right) noexcept
{
    return __simd_bit_xor<_SimdGeneration_, _RegisterPolicy_>(__left._vector, __right._vector);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
simd_stl_always_inline simd<_SimdGeneration_, _Element_, _RegisterPolicy_> operator+(
    const simd<_SimdGeneration_, _Element_, _RegisterPolicy_>&                      __left,
    const typename simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::value_type  __right) noexcept
{
    return *this + simd(__right);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
simd_stl_always_inline simd<_SimdGeneration_, _Element_, _RegisterPolicy_> operator-(
    const simd<_SimdGeneration_, _Element_, _RegisterPolicy_>&                      __left,
    const typename simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::value_type  __right) noexcept
{
    return __simd_substract<_SimdGeneration_, _RegisterPolicy_, _Element_>(__left._vector, __simd_unwrap(simd(__right)));
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
simd_stl_always_inline simd<_SimdGeneration_, _Element_, _RegisterPolicy_> operator*(
    const simd<_SimdGeneration_, _Element_, _RegisterPolicy_>&                      __left,
    const typename simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::value_type  __right) noexcept
{
    return __simd_multiply<_SimdGeneration_, _RegisterPolicy_, _Element_>(__left._vector, __simd_unwrap(simd(__right)));
}


template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
simd_stl_always_inline simd<_SimdGeneration_, _Element_, _RegisterPolicy_> operator/(
    const simd<_SimdGeneration_, _Element_, _RegisterPolicy_>&                      __left,
    const typename simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::value_type  __right) noexcept
{
    return __simd_divide<_SimdGeneration_, _RegisterPolicy_, _Element_>(__left._vector, __simd_unwrap(simd(__right)));
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
    return __simd_negate<_SimdGeneration_, _RegisterPolicy_, _Element_>(_vector);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
template <typename _DesiredType_>
simd_stl_always_inline _DesiredType_
    simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::extract(size_type __index) const noexcept
{
    simd_stl_debug_assert(__index >= 0 && __index < size<_DesiredType_>(), "simd_stl::numeric::basic_simd: Index out of range");
    return __simd_extract<_SimdGeneration_, _RegisterPolicy_, _DesiredType_>(_vector, __index);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
template <typename _DesiredType_>
simd_stl_always_inline simd_element_reference<simd<_SimdGeneration_, _Element_, _RegisterPolicy_>>
    simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::extract_wrapped(size_type __index) noexcept
{
    simd_stl_debug_assert(__index >= 0 && __index < size<_DesiredType_>(), "simd_stl::numeric::basic_simd: Index out of range");
    return simd_element_reference<simd>(this, __index);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
template <typename _DesiredType_>
simd_stl_always_inline void simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::insert(
    const size_type                                         __address,
    const typename std::type_identity<_DesiredType_>::type  __value) noexcept
{
    return __simd_insert<_SimdGeneration_, _RegisterPolicy_, _DesiredType_>(_vector, __address, __value);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
template <class _AlignmentPolicy_>
simd_stl_always_inline simd<_SimdGeneration_, _Element_, _RegisterPolicy_>
simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::load(const void* __address, _AlignmentPolicy_&&)  noexcept
{
    if constexpr (std::remove_cvref_t<_AlignmentPolicy_>::__alignment)
        return __simd_load_aligned<_SimdGeneration_, _RegisterPolicy_, vector_type>(__address);
    else 
        return __simd_load_unaligned<_SimdGeneration_, _RegisterPolicy_, vector_type>(__address);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
template <class _AlignmentPolicy_>
simd_stl_always_inline void simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::store(void* __address, _AlignmentPolicy_&&) const noexcept {
    if constexpr (std::remove_cvref_t<_AlignmentPolicy_>::__alignment)
        return __simd_store_aligned<_SimdGeneration_, _RegisterPolicy_, vector_type>(__address, _vector);
    else
        return __simd_store_unaligned<_SimdGeneration_, _RegisterPolicy_, vector_type>(__address, _vector);
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
        const void*         __address,
        const _MaskType_&   __mask,
        _AlignmentPolicy_&&) noexcept
{
    if constexpr (std::remove_cvref_t<_AlignmentPolicy_>::__alignment)
        return __simd_mask_load_aligned<_SimdGeneration_, _RegisterPolicy_, vector_type, _DesiredType_>(
            reinterpret_cast<const _DesiredType_*>(__address), __simd_unwrap_mask(__mask), simd(0)._vector);
    else 
        return __simd_mask_load_unaligned<_SimdGeneration_, _RegisterPolicy_, vector_type, _DesiredType_>(
            reinterpret_cast<const _DesiredType_*>(__address), __simd_unwrap_mask(__mask), simd(0)._vector);
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
    std::enable_if_t<__is_intrin_type_v<_VectorType_> || __is_valid_basic_simd_v<_VectorType_>, int>>
simd_stl_always_inline simd<_SimdGeneration_, _Element_, _RegisterPolicy_> simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::mask_load(
    const void*         __address,
    const _MaskType_&   __mask,
    _VectorType_        __additional_source,
    _AlignmentPolicy_&&) noexcept
{
    if constexpr (std::remove_cvref_t<_AlignmentPolicy_>::__alignment)
        return __simd_mask_load_aligned<_SimdGeneration_, _RegisterPolicy_, vector_type, _DesiredType_>(
            reinterpret_cast<const _DesiredType_*>(__address), __simd_unwrap_mask(__mask), __simd_unwrap(__additional_source));
    else
        return __simd_mask_load_unaligned<_SimdGeneration_, _RegisterPolicy_, vector_type, _DesiredType_>(
            reinterpret_cast<const _DesiredType_*>(__address), __simd_unwrap_mask(__mask), __simd_unwrap(__additional_source));
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
    void*               __address,
    const _MaskType_& __mask,
    _AlignmentPolicy_&&) const noexcept
{
    if constexpr (std::remove_cvref_t<_AlignmentPolicy_>::__alignment)
        __simd_mask_store_aligned<_SimdGeneration_, _RegisterPolicy_, _DesiredType_>(
            reinterpret_cast<_DesiredType_*>(__address), __simd_unwrap_mask(__mask), _vector);
    else
        __simd_mask_store_unaligned<_SimdGeneration_, _RegisterPolicy_, _DesiredType_>(
            reinterpret_cast<_DesiredType_*>(__address), __simd_unwrap_mask(__mask), _vector);
}

//template <
//    arch::CpuFeature	_SimdGeneration_,
//    typename			_Element_,
//    class               _RegisterPolicy_>
//simd_stl_always_inline simd<_SimdGeneration_, _Element_, _RegisterPolicy_> operator>>(
//    const simd<_SimdGeneration_, _Element_, _RegisterPolicy_>     __left,
//    const uint32                                                  _Shift) noexcept
//{
//    return _SimdShiftRightElements<_SimdGeneration_, _RegisterPolicy_, _Element_>(__left._vector, _Shift);
//}
//
//template <
//    arch::CpuFeature	_SimdGeneration_,
//    typename			_Element_,
//    class               _RegisterPolicy_>
//simd_stl_always_inline simd<_SimdGeneration_, _Element_, _RegisterPolicy_> operator<<(
//    const simd<_SimdGeneration_, _Element_, _RegisterPolicy_>   __left,
//    const uint32                                                _Shift) noexcept
//{
//    return _SimdShiftLeftElements<_SimdGeneration_, _RegisterPolicy_, _Element_>(__left._vector, _Shift);
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
    const simd<_SimdGeneration_, _Element_, _RegisterPolicy_>& __left,
    const simd<_SimdGeneration_, _Element_, _RegisterPolicy_>& __right) noexcept
{
    return __left.mask_compare<simd_comparison::equal>(__right).all_of();
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
simd_stl_always_inline bool operator!=(
    const simd<_SimdGeneration_, _Element_, _RegisterPolicy_>& __left,
    const simd<_SimdGeneration_, _Element_, _RegisterPolicy_>& __right) noexcept
{
    return !(__left == __right);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
template <
    simd_comparison _Comparison_,
    typename        _DesiredType_>
simd_stl_always_inline simd_mask<_SimdGeneration_, _DesiredType_, _RegisterPolicy_>
    simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::mask_compare(const simd& __right) const noexcept
{
    return __simd_mask_compare<_SimdGeneration_, _RegisterPolicy_, _DesiredType_, _Comparison_>(_vector, __right._vector);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
template <
    simd_comparison _Comparison_,
    typename        _DesiredType_>
simd_stl_always_inline simd<_SimdGeneration_, _DesiredType_, _RegisterPolicy_>
    simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::vector_compare(const simd& __right) const noexcept
{
    return __simd_compare<_SimdGeneration_, _RegisterPolicy_, _DesiredType_, _Comparison_>(_vector, __right._vector);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
template <
    simd_comparison _Comparison_,
    typename        _DesiredType_>
simd_stl_always_inline __native_compare_return_type<simd<_SimdGeneration_, _Element_,
    _RegisterPolicy_>, _DesiredType_, _Comparison_>
    simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::native_compare(const simd& __right) const noexcept
{
    return __simd_native_compare<_SimdGeneration_, _RegisterPolicy_, _DesiredType_, _Comparison_>(_vector, __right._vector);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
template <typename _DesiredType_>
simd_stl_always_inline simd_mask<_SimdGeneration_, _DesiredType_, _RegisterPolicy_>
simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::to_mask() const noexcept
{
    return __simd_to_mask<_SimdGeneration_, _RegisterPolicy_, _DesiredType_>(_vector);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
template <typename _DesiredType_>
simd_stl_always_inline __reduce_type<_DesiredType_> simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::reduce_add() const noexcept {
    return __simd_reduce<_SimdGeneration_, _RegisterPolicy_, _DesiredType_>(_vector);
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

    constexpr auto __length = (sizeof(vector_type) / sizeof(_ElementType_));
    return __length;
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
__make_tail_mask_return_type<simd<_SimdGeneration_, _Element_, _RegisterPolicy_>>
simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::make_tail_mask(uint32 __bytes) noexcept
{
    return __simd_make_tail_mask<_SimdGeneration_, _RegisterPolicy_, _Element_>(__bytes);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
void simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::streaming_fence() noexcept {
    return __simd_streaming_fence<_SimdGeneration_>();
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
simd<_SimdGeneration_, _Element_, _RegisterPolicy_> simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::non_temporal_load(const void* __address) noexcept
{
    return __simd_non_temporal_load<_SimdGeneration_, _RegisterPolicy_, vector_type>(__address);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
void simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::non_temporal_store(void* __address) const noexcept {
    __simd_non_temporal_store<_SimdGeneration_, _RegisterPolicy_>(__address, _vector);
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
    void*               __address,
    const _MaskType_&   __mask,
    _AlignmentPolicy_&&) const noexcept
{
    if constexpr (std::remove_cvref_t<_AlignmentPolicy_>::__alignment)
        return __simd_compress_store_aligned<_SimdGeneration_, _RegisterPolicy_, _DesiredType_>(
            reinterpret_cast<_DesiredType_*>(__address), __simd_to_mask<_SimdGeneration_, _RegisterPolicy_, _DesiredType_>(__simd_unwrap_mask(__mask)), _vector);
    else 
        return __simd_compress_store_unaligned<_SimdGeneration_, _RegisterPolicy_, _DesiredType_>(
            reinterpret_cast<_DesiredType_*>(__address), __simd_to_mask<_SimdGeneration_, _RegisterPolicy_, _DesiredType_>(__simd_unwrap_mask(__mask)), _vector);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
template <
    class       _MaskType_,
    typename    _DesiredType_>
simd_stl_always_inline void simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::blend(
    const simd<_SimdGeneration_, _DesiredType_, _RegisterPolicy_>&  __vector,
    const _MaskType_&                                               __mask) noexcept
{
    _vector = simd_cast<vector_type>(__simd_blend<_SimdGeneration_, _RegisterPolicy_, _DesiredType_>(
        simd_cast<decltype(__vector._vector)>(_vector), __vector._vector, __simd_unwrap_mask(__mask)));
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
template <typename _DesiredType_>
simd_stl_always_inline simd<_SimdGeneration_, _DesiredType_, _RegisterPolicy_> 
    simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::vertical_min(
        const simd<_SimdGeneration_, _DesiredType_, _RegisterPolicy_>& __other) const noexcept 
{
    return __simd_vertical_min<_SimdGeneration_, _RegisterPolicy_, _DesiredType_>(_vector, __other._vector);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
template <typename _DesiredType_>
simd_stl_always_inline _DesiredType_ simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::horizontal_min() const noexcept {
    return __simd_horizontal_min<_SimdGeneration_, _RegisterPolicy_, _DesiredType_>(_vector);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
template <typename _DesiredType_>
simd_stl_always_inline simd<_SimdGeneration_, _DesiredType_, _RegisterPolicy_> 
    simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::vertical_max(
        const simd<_SimdGeneration_, _DesiredType_, _RegisterPolicy_>& __other) const noexcept
{
    return __simd_vertical_max<_SimdGeneration_, _RegisterPolicy_, _DesiredType_>(_vector, __other._vector);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
template <typename _DesiredType_>
simd_stl_always_inline _DesiredType_ simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::horizontal_max() const noexcept {
    return __simd_horizontal_max<_SimdGeneration_, _RegisterPolicy_, _DesiredType_>(_vector);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
template <typename _DesiredType_>
simd_stl_always_inline simd<_SimdGeneration_, _DesiredType_, _RegisterPolicy_> 
    simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::abs() const noexcept 
{
    return __simd_abs<_SimdGeneration_, _RegisterPolicy_, _DesiredType_>(_vector);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
template <typename _DesiredType_>
void simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::reverse() noexcept {
    _vector = __simd_reverse<_SimdGeneration_, _RegisterPolicy_, _DesiredType_>(_vector);
}

template <arch::CpuFeature _SimdGeneration_>
zero_upper_at_exit_guard<_SimdGeneration_>::zero_upper_at_exit_guard() noexcept
{}

template <arch::CpuFeature _SimdGeneration_>
zero_upper_at_exit_guard<_SimdGeneration_>::~zero_upper_at_exit_guard() noexcept {
    if constexpr (type_traits::__is_zeroupper_required_v<_SimdGeneration_>)
        _mm256_zeroupper();
}

__SIMD_STL_NUMERIC_NAMESPACE_END
