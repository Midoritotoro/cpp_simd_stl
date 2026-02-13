#pragma once 

__SIMD_STL_DATAPAR_NAMESPACE_BEGIN

template <
	arch::ISA	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_>
simd_mask<_SimdGeneration_, _Element_, _RegisterPolicy_>::simd_mask() noexcept
{}

template <
	arch::ISA	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_>
template <class _VectorMask_, std::enable_if_t<__is_valid_basic_simd_v<_VectorMask_> || __is_intrin_type_v<_VectorMask_>, int>>
simd_mask<_SimdGeneration_, _Element_, _RegisterPolicy_>::simd_mask(const _VectorMask_& __vector_mask) noexcept {
	if constexpr (__is_valid_basic_simd_v<_VectorMask_>)
		_mask = __vector_mask.to_mask().unwrap();
	else
		_mask = __simd_to_mask<_SimdGeneration_, _RegisterPolicy_, _Element_>(__vector_mask);
}

template <
	arch::ISA	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_>
simd_mask<_SimdGeneration_, _Element_, _RegisterPolicy_>::simd_mask(const mask_type __mask) noexcept :
	_mask(__mask)
{}

template <
	arch::ISA	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_>
simd_mask<_SimdGeneration_, _Element_, _RegisterPolicy_>::simd_mask(const simd_mask& __mask) noexcept :
	_mask(__mask._mask)
{}

template <
	arch::ISA	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_>
constexpr simd_mask<_SimdGeneration_, _Element_, _RegisterPolicy_>::mask_type 
	simd_mask<_SimdGeneration_, _Element_, _RegisterPolicy_>::unwrap() const noexcept 
{
	return _mask;
}

template <
	arch::ISA	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_>
constexpr bool simd_mask<_SimdGeneration_, _Element_, _RegisterPolicy_>::operator[](int32 __index) const noexcept {
	return (_mask >> __index) & 1;
}

template <
	arch::ISA	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_>
constexpr simd_mask<_SimdGeneration_, _Element_, _RegisterPolicy_> simd_mask<_SimdGeneration_, _Element_, _RegisterPolicy_>::operator++(int) noexcept {
	simd_mask __self = *this;
	++_mask;
	return __self;
}

template <
	arch::ISA	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_>
constexpr simd_mask<_SimdGeneration_, _Element_, _RegisterPolicy_>& simd_mask<_SimdGeneration_, _Element_, _RegisterPolicy_>::operator++() noexcept {
	++_mask;
	return *this;
}

template <
	arch::ISA	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_>
constexpr simd_mask<_SimdGeneration_, _Element_, _RegisterPolicy_> simd_mask<_SimdGeneration_, _Element_, _RegisterPolicy_>::operator--(int) noexcept {
	simd_mask __self = *this;
	--_mask;
	return __self;
}

template <
	arch::ISA	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_>
constexpr simd_mask<_SimdGeneration_, _Element_, _RegisterPolicy_>& simd_mask<_SimdGeneration_, _Element_, _RegisterPolicy_>::operator--() noexcept {
	--_mask;
	return *this;
}

template <
	arch::ISA	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_>
constexpr simd_mask<_SimdGeneration_, _Element_, _RegisterPolicy_>&
	simd_mask<_SimdGeneration_, _Element_, _RegisterPolicy_>::operator&=(const simd_mask& __other) noexcept
{
	return *this = (*this & __other);
}

template <
	arch::ISA	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_>
constexpr simd_mask<_SimdGeneration_, _Element_, _RegisterPolicy_>& 
	simd_mask<_SimdGeneration_, _Element_, _RegisterPolicy_>::operator|=(const simd_mask& __other) noexcept
{
	return *this = (*this | __other);
}

template <
	arch::ISA	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_>
constexpr simd_mask<_SimdGeneration_, _Element_, _RegisterPolicy_>& 
	simd_mask<_SimdGeneration_, _Element_, _RegisterPolicy_>::operator^=(const simd_mask& __other) noexcept
{
	return *this = (*this ^ __other);
}

template <
	arch::ISA	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_>
constexpr simd_mask<_SimdGeneration_, _Element_, _RegisterPolicy_>& 
	simd_mask<_SimdGeneration_, _Element_, _RegisterPolicy_>::operator>>=(const uint8 __shift) const noexcept
{
	return *this = (*this >> __shift);
}

template <
	arch::ISA	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_>
constexpr simd_mask<_SimdGeneration_, _Element_, _RegisterPolicy_>& 
	simd_mask<_SimdGeneration_, _Element_, _RegisterPolicy_>::operator<<=(const uint8 __shift) const noexcept
{
	return *this = (*this << __shift);
}

template <
	arch::ISA	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_>
template <uint8 _Shift_>
constexpr simd_mask<_SimdGeneration_, _Element_, _RegisterPolicy_>&
simd_mask<_SimdGeneration_, _Element_, _RegisterPolicy_>::operator>>=(const std::integral_constant<uint8, _Shift_> __shift) const noexcept
{
	return *this = (*this >> __shift);
}

template <
	arch::ISA	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_>
template <uint8 _Shift_>
constexpr simd_mask<_SimdGeneration_, _Element_, _RegisterPolicy_>&
simd_mask<_SimdGeneration_, _Element_, _RegisterPolicy_>::operator<<=(const std::integral_constant<uint8, _Shift_> __shift) const noexcept
{
	return *this = (*this << __shift);
}

template <
	arch::ISA	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_>
constexpr uint8 simd_mask<_SimdGeneration_, _Element_, _RegisterPolicy_>::bit_width() noexcept {
	return __used_bits;
}

template <
	arch::ISA	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_>
constexpr arch::ISA simd_mask<_SimdGeneration_, _Element_, _RegisterPolicy_>::generation() noexcept {
	return _SimdGeneration_;
}

__SIMD_STL_DATAPAR_NAMESPACE_END
