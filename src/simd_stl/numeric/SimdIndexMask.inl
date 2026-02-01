#pragma once 

__SIMD_STL_NUMERIC_NAMESPACE_BEGIN

template <
	arch::CpuFeature	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_>
simd_index_mask<_SimdGeneration_, _Element_, _RegisterPolicy_>::simd_index_mask() noexcept
{}

template <
	arch::CpuFeature	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_>
template <class _VectorMask_, std::enable_if_t<__is_valid_basic_simd_v<_VectorMask_> || __is_intrin_type_v<_VectorMask_>, int>>
simd_index_mask<_SimdGeneration_, _Element_, _RegisterPolicy_>::simd_index_mask(const _VectorMask_& __vector_mask) noexcept {
	if constexpr (__is_valid_basic_simd_v<_VectorMask_>)
		_mask = __vector_mask.to_index_mask().unwrap();
	else
		_mask = __simd_to_index_mask<_SimdGeneration_, _RegisterPolicy_, _Element_>(__vector_mask);
}

template <
	arch::CpuFeature	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_>
simd_index_mask<_SimdGeneration_, _Element_, _RegisterPolicy_>::simd_index_mask(const mask_type __mask) noexcept :
	_mask(__mask)
{}

template <
	arch::CpuFeature	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_>
constexpr bool simd_index_mask<_SimdGeneration_, _Element_, _RegisterPolicy_>::all_of() const noexcept {
	return __implementation::all_of(_mask);
}

template <
	arch::CpuFeature	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_>
constexpr bool simd_index_mask<_SimdGeneration_, _Element_, _RegisterPolicy_>::any_of() const noexcept {
	return __implementation::any_of(_mask);
}

template <
	arch::CpuFeature	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_>
constexpr bool simd_index_mask<_SimdGeneration_, _Element_, _RegisterPolicy_>::none_of() const noexcept {
	return __implementation::none_of(_mask);
}

template <
	arch::CpuFeature	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_>
constexpr simd_index_mask<_SimdGeneration_, _Element_, _RegisterPolicy_>::size_type
	simd_index_mask<_SimdGeneration_, _Element_, _RegisterPolicy_>::count_set() const noexcept
{
	return __implementation::count_set(_mask);
}

template <
	arch::CpuFeature	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_>
constexpr simd_index_mask<_SimdGeneration_, _Element_, _RegisterPolicy_>::size_type
	simd_index_mask<_SimdGeneration_, _Element_, _RegisterPolicy_>::count_trailing_zero_bits() const noexcept
{
	return __implementation::count_trailing_zero_bits(_mask);
}

template <
	arch::CpuFeature	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_>
constexpr simd_index_mask<_SimdGeneration_, _Element_, _RegisterPolicy_>::size_type
	simd_index_mask<_SimdGeneration_, _Element_, _RegisterPolicy_>::count_leading_zero_bits() const noexcept 
{
	return __implementation::count_leading_zero_bits(_mask);
}

template <
	arch::CpuFeature	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_>
constexpr simd_index_mask<_SimdGeneration_, _Element_, _RegisterPolicy_>::size_type
	simd_index_mask<_SimdGeneration_, _Element_, _RegisterPolicy_>::count_trailing_one_bits() const noexcept
{
	return __implementation::count_trailing_zero_bits(~_mask);
}

template <
	arch::CpuFeature	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_>
constexpr simd_index_mask<_SimdGeneration_, _Element_, _RegisterPolicy_>::size_type
	simd_index_mask<_SimdGeneration_, _Element_, _RegisterPolicy_>::count_leading_one_bits() const noexcept 
{
	return __implementation::count_leading_zero_bits(~_mask);
}

template <
	arch::CpuFeature	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_>
constexpr simd_index_mask<_SimdGeneration_, _Element_, _RegisterPolicy_>::mask_type
	simd_index_mask<_SimdGeneration_, _Element_, _RegisterPolicy_>::unwrap() const noexcept 
{
	return _mask;
}

template <
	arch::CpuFeature	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_>
constexpr simd_index_mask<_SimdGeneration_, _Element_, _RegisterPolicy_>::operator bool() const noexcept {
	return any_of();
}

template <
	arch::CpuFeature	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_>
constexpr void simd_index_mask<_SimdGeneration_, _Element_, _RegisterPolicy_>::clear_left_most_set_bit() noexcept {
	_mask = _mask & (_mask - 1);
}

template <
	arch::CpuFeature	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_>
constexpr bool simd_index_mask<_SimdGeneration_, _Element_, _RegisterPolicy_>
	::operator==(const simd_index_mask& __other) const noexcept
{
	return _mask == __other._mask;
}

template <
	arch::CpuFeature	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_>
constexpr bool simd_index_mask<_SimdGeneration_, _Element_, _RegisterPolicy_>
	::operator!=(const simd_index_mask& __other) const noexcept
{
	return _mask != __other._mask;
}

template <
	arch::CpuFeature	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_>
constexpr simd_index_mask<_SimdGeneration_, _Element_, _RegisterPolicy_> 
	simd_index_mask<_SimdGeneration_, _Element_, _RegisterPolicy_>::operator~() const noexcept
{
	return __implementation::__bit_not(_mask);
}

template <
	arch::CpuFeature	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_>
constexpr simd_index_mask<_SimdGeneration_, _Element_, _RegisterPolicy_> 
	simd_index_mask<_SimdGeneration_, _Element_, _RegisterPolicy_>::operator&(const simd_index_mask& __other) const noexcept
{
	return __implementation::__bit_and(_mask, __other._mask);
}

template <
	arch::CpuFeature	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_>
constexpr simd_index_mask<_SimdGeneration_, _Element_, _RegisterPolicy_> 
	simd_index_mask<_SimdGeneration_, _Element_, _RegisterPolicy_>::operator|(const simd_index_mask& __other) const noexcept
{
	return __implementation::__bit_or(_mask, __other._mask);
}

template <
	arch::CpuFeature	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_>
constexpr simd_index_mask<_SimdGeneration_, _Element_, _RegisterPolicy_> 
	simd_index_mask<_SimdGeneration_, _Element_, _RegisterPolicy_>::operator^(const simd_index_mask& __other) const noexcept
{
	return __implementation::__bit_xor(_mask, __other._mask);
}

template <
	arch::CpuFeature	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_>
constexpr simd_index_mask<_SimdGeneration_, _Element_, _RegisterPolicy_>&
	simd_index_mask<_SimdGeneration_, _Element_, _RegisterPolicy_>::operator&=(const simd_index_mask& __other) noexcept
{
	_mask &= __other._mask;
	return *this;
}

template <
	arch::CpuFeature	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_>
constexpr simd_index_mask<_SimdGeneration_, _Element_, _RegisterPolicy_>& 
	simd_index_mask<_SimdGeneration_, _Element_, _RegisterPolicy_>::operator|=(const simd_index_mask& __other) noexcept
{
	_mask |= __other._mask;
	return *this;
}

template <
	arch::CpuFeature	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_>
constexpr simd_index_mask<_SimdGeneration_, _Element_, _RegisterPolicy_>& 
	simd_index_mask<_SimdGeneration_, _Element_, _RegisterPolicy_>::operator^=(const simd_index_mask& __other) noexcept
{
	_mask ^= __other._mask;
	return *this;
}

__SIMD_STL_NUMERIC_NAMESPACE_END
