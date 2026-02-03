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
constexpr simd_index_mask<_SimdGeneration_, _Element_, _RegisterPolicy_>::mask_type
	simd_index_mask<_SimdGeneration_, _Element_, _RegisterPolicy_>::unwrap() const noexcept 
{
	return _mask;
}

template <
	arch::CpuFeature	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_>
constexpr arch::CpuFeature simd_index_mask<_SimdGeneration_, _Element_, _RegisterPolicy_>::generation() noexcept {
	return _SimdGeneration_;
}

template <
	arch::CpuFeature	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_>
constexpr uint8 simd_index_mask<_SimdGeneration_, _Element_, _RegisterPolicy_>::bit_width() noexcept {
	return __used_bits;
}

template <
	arch::CpuFeature	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_>
constexpr uint8 simd_index_mask<_SimdGeneration_, _Element_, _RegisterPolicy_>::divisor() noexcept {
	return __divisor;
}

template <
	arch::CpuFeature	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_>
constexpr simd_index_mask<_SimdGeneration_, _Element_, _RegisterPolicy_>&
	simd_index_mask<_SimdGeneration_, _Element_, _RegisterPolicy_>::operator&=(const simd_index_mask& __other) noexcept
{
	return *this = (*this & __other);
}

template <
	arch::CpuFeature	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_>
constexpr simd_index_mask<_SimdGeneration_, _Element_, _RegisterPolicy_>& 
	simd_index_mask<_SimdGeneration_, _Element_, _RegisterPolicy_>::operator|=(const simd_index_mask& __other) noexcept
{
	return *this = (*this | __other);
}

template <
	arch::CpuFeature	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_>
constexpr simd_index_mask<_SimdGeneration_, _Element_, _RegisterPolicy_>& 
	simd_index_mask<_SimdGeneration_, _Element_, _RegisterPolicy_>::operator^=(const simd_index_mask& __other) noexcept
{
	return *this = (*this ^ __other);
}


template <
	arch::CpuFeature	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_>
constexpr simd_index_mask<_SimdGeneration_, _Element_, _RegisterPolicy_>& 
	simd_index_mask<_SimdGeneration_, _Element_, _RegisterPolicy_>::operator>>=(const uint8 __shift) noexcept
{
	return *this = (*this >> __shift);
}

template <
	arch::CpuFeature	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_>
constexpr simd_index_mask<_SimdGeneration_, _Element_, _RegisterPolicy_>& 
	simd_index_mask<_SimdGeneration_, _Element_, _RegisterPolicy_>::operator<<=(const uint8 __shift) noexcept 
{
	return *this = (*this << __shift);
}

template <
	arch::CpuFeature	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_>
template <sizetype _Shift_>
constexpr simd_index_mask<_SimdGeneration_, _Element_, _RegisterPolicy_>& 
	simd_index_mask<_SimdGeneration_, _Element_, _RegisterPolicy_>::operator>>=(const std::integral_constant<uint8, _Shift_> __shift) noexcept 
{
	return *this = (*this >> __shift);
}

template <
	arch::CpuFeature	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_>
template <sizetype _Shift_>
constexpr simd_index_mask<_SimdGeneration_, _Element_, _RegisterPolicy_>& 
	simd_index_mask<_SimdGeneration_, _Element_, _RegisterPolicy_>::operator<<=(const std::integral_constant<uint8, _Shift_> __shift) noexcept
{
	return *this = (*this << __shift);
}

__SIMD_STL_NUMERIC_NAMESPACE_END
