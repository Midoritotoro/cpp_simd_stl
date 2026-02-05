#pragma once 

simd_stl_disable_warning_msvc(4002)

__SIMD_STL_DATAPAR_NAMESPACE_BEGIN

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
simd<_SimdGeneration_, _Element_, _RegisterPolicy_> operator/(
    const simd<_SimdGeneration_, _Element_, _RegisterPolicy_>& __left,
    const simd<_SimdGeneration_, _Element_, _RegisterPolicy_>& __right) noexcept
{
    return __simd_divide<_SimdGeneration_, _RegisterPolicy_, _Element_>(__left._vector, __right._vector);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
simd<_SimdGeneration_, _Element_, _RegisterPolicy_> operator+(
    const simd<_SimdGeneration_, _Element_, _RegisterPolicy_>& __left,
    const simd<_SimdGeneration_, _Element_, _RegisterPolicy_>& __right) noexcept
{
    return __simd_add<_SimdGeneration_, _RegisterPolicy_, _Element_>(__left._vector, __right._vector);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
simd<_SimdGeneration_, _Element_, _RegisterPolicy_> operator-(
    const simd<_SimdGeneration_, _Element_, _RegisterPolicy_>& __left,
    const simd<_SimdGeneration_, _Element_, _RegisterPolicy_>& __right) noexcept
{
    return __simd_substract<_SimdGeneration_, _RegisterPolicy_, _Element_>(__left._vector, __right._vector);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
simd<_SimdGeneration_, _Element_, _RegisterPolicy_> operator*(
    const simd<_SimdGeneration_, _Element_, _RegisterPolicy_>& __left,
    const simd<_SimdGeneration_, _Element_, _RegisterPolicy_>& __right) noexcept
{
    return __simd_multiply<_SimdGeneration_, _RegisterPolicy_, _Element_>(__left._vector, __right._vector);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
simd<_SimdGeneration_, _Element_, _RegisterPolicy_> operator&(
    const simd<_SimdGeneration_, _Element_, _RegisterPolicy_>& __left,
    const simd<_SimdGeneration_, _Element_, _RegisterPolicy_>& __right) noexcept
{
    return __simd_bit_and<_SimdGeneration_, _RegisterPolicy_>(__left._vector, __right._vector);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
simd<_SimdGeneration_, _Element_, _RegisterPolicy_> operator|(
    const simd<_SimdGeneration_, _Element_, _RegisterPolicy_>& __left,
    const simd<_SimdGeneration_, _Element_, _RegisterPolicy_>& __right) noexcept
{
    return __simd_bit_or<_SimdGeneration_, _RegisterPolicy_>(__left._vector, __right._vector);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
simd<_SimdGeneration_, _Element_, _RegisterPolicy_> operator^(
    const simd<_SimdGeneration_, _Element_, _RegisterPolicy_>& __left,
    const simd<_SimdGeneration_, _Element_, _RegisterPolicy_>& __right) noexcept
{
    return __simd_bit_xor<_SimdGeneration_, _RegisterPolicy_>(__left._vector, __right._vector);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
simd<_SimdGeneration_, _Element_, _RegisterPolicy_> operator+(
    const simd<_SimdGeneration_, _Element_, _RegisterPolicy_>&                      __left,
    const typename simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::value_type  __right) noexcept
{
    return *this + simd(__right);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
simd<_SimdGeneration_, _Element_, _RegisterPolicy_> operator-(
    const simd<_SimdGeneration_, _Element_, _RegisterPolicy_>&                      __left,
    const typename simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::value_type  __right) noexcept
{
    return __simd_substract<_SimdGeneration_, _RegisterPolicy_, _Element_>(__left._vector, __simd_unwrap(simd<_SimdGeneration_, _Element_, _RegisterPolicy_>(__right)));
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
simd<_SimdGeneration_, _Element_, _RegisterPolicy_> operator*(
    const simd<_SimdGeneration_, _Element_, _RegisterPolicy_>&                      __left,
    const typename simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::value_type  __right) noexcept
{
    return __simd_multiply<_SimdGeneration_, _RegisterPolicy_, _Element_>(__left._vector, __simd_unwrap(simd<_SimdGeneration_, _Element_, _RegisterPolicy_>(__right)));
}


template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
simd<_SimdGeneration_, _Element_, _RegisterPolicy_> operator/(
    const simd<_SimdGeneration_, _Element_, _RegisterPolicy_>&                      __left,
    const typename simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::value_type  __right) noexcept
{
    return __simd_divide<_SimdGeneration_, _RegisterPolicy_, _Element_>(__left._vector, __simd_unwrap(simd<_SimdGeneration_, _Element_, _RegisterPolicy_>(__right)));
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
    simd_stl_debug_assert(__index >= 0 && __index < size<_DesiredType_>(), "simd_stl::datapar::basic_simd: Index out of range");
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
    simd_stl_debug_assert(__index >= 0 && __index < size<_DesiredType_>(), "simd_stl::datapar::basic_simd: Index out of range");
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
simd_compare_result<_SimdGeneration_, _Element_, _RegisterPolicy_, simd_comparison::equal> operator==(
    const simd<_SimdGeneration_, _Element_, _RegisterPolicy_>& __left,
    const simd<_SimdGeneration_, _Element_, _RegisterPolicy_>& __right) noexcept
{
    return simd_compare_result<_SimdGeneration_, _Element_, _RegisterPolicy_, simd_comparison::equal> {
        __simd_native_compare<_SimdGeneration_, _RegisterPolicy_, _Element_, simd_comparison::equal>(__left._vector, __right._vector) };
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
simd_compare_result<_SimdGeneration_, _Element_, _RegisterPolicy_, simd_comparison::not_equal> operator!=(
    const simd<_SimdGeneration_, _Element_, _RegisterPolicy_>& __left, 
    const simd<_SimdGeneration_, _Element_, _RegisterPolicy_>& __right) noexcept
{
    return simd_compare_result<_SimdGeneration_, _Element_, _RegisterPolicy_, simd_comparison::not_equal> {
        __simd_native_compare<_SimdGeneration_, _RegisterPolicy_, _Element_, simd_comparison::not_equal>(__left._vector, __right._vector) };
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
simd_compare_result<_SimdGeneration_, _Element_, _RegisterPolicy_, simd_comparison::less> operator<(
    const simd<_SimdGeneration_, _Element_, _RegisterPolicy_>& __left,
    const simd<_SimdGeneration_, _Element_, _RegisterPolicy_>& __right) noexcept
{
    return simd_compare_result<_SimdGeneration_, _Element_, _RegisterPolicy_, simd_comparison::less> {
        __simd_native_compare<_SimdGeneration_, _RegisterPolicy_, _Element_, simd_comparison::less>(__left._vector, __right._vector) };
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
simd_compare_result<_SimdGeneration_, _Element_, _RegisterPolicy_, simd_comparison::less_equal> operator<=(
    const simd<_SimdGeneration_, _Element_, _RegisterPolicy_>& __left,
    const simd<_SimdGeneration_, _Element_, _RegisterPolicy_>& __right) noexcept
{
    return simd_compare_result<_SimdGeneration_, _Element_, _RegisterPolicy_, simd_comparison::less_equal> {
        __simd_native_compare<_SimdGeneration_, _RegisterPolicy_, _Element_, simd_comparison::less_equal>(__left._vector, __right._vector) };
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
simd_compare_result<_SimdGeneration_, _Element_, _RegisterPolicy_, simd_comparison::greater> operator>(
    const simd<_SimdGeneration_, _Element_, _RegisterPolicy_>& __left, 
    const simd<_SimdGeneration_, _Element_, _RegisterPolicy_>& __right) noexcept
{
    return simd_compare_result<_SimdGeneration_, _Element_, _RegisterPolicy_, simd_comparison::greater> {
        __simd_native_compare<_SimdGeneration_, _RegisterPolicy_, _Element_, simd_comparison::greater>(__left._vector, __right._vector) };
}

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
simd_compare_result<_SimdGeneration_, _Element_, _RegisterPolicy_, simd_comparison::greater_equal> operator>=(
    const simd<_SimdGeneration_, _Element_, _RegisterPolicy_>& __left,
    const simd<_SimdGeneration_, _Element_, _RegisterPolicy_>& __right) noexcept
{
    return simd_compare_result<_SimdGeneration_, _Element_, _RegisterPolicy_, simd_comparison::greater_equal> {
        __simd_native_compare<_SimdGeneration_, _RegisterPolicy_, _Element_, simd_comparison::greater_equal>(__left._vector, __right._vector) };
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
simd_stl_always_inline auto simd<_SimdGeneration_, _Element_, _RegisterPolicy_>::to_index_mask() const noexcept {
    return __simd_to_index_mask<_SimdGeneration_, _RegisterPolicy_, _DesiredType_>(_vector);
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

__SIMD_STL_DATAPAR_NAMESPACE_END
