#pragma once 

__SIMD_STL_NUMERIC_NAMESPACE_BEGIN

template <
	arch::CpuFeature	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_>
basic_simd_mask<_SimdGeneration_, _Element_, _RegisterPolicy_>::basic_simd_mask() noexcept
{}

template <
	arch::CpuFeature	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_>
basic_simd_mask<_SimdGeneration_, _Element_, _RegisterPolicy_>::basic_simd_mask(const mask_type mask) noexcept :
	_mask(mask)
{}

template <
	arch::CpuFeature	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_>
constexpr bool basic_simd_mask<_SimdGeneration_, _Element_, _RegisterPolicy_>::allOf() const noexcept {
	return implementation::allOf(_mask);
}

template <
	arch::CpuFeature	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_>
constexpr bool basic_simd_mask<_SimdGeneration_, _Element_, _RegisterPolicy_>::anyOf() const noexcept {
	return implementation::anyOf(_mask);
}

template <
	arch::CpuFeature	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_>
constexpr  bool basic_simd_mask<_SimdGeneration_, _Element_, _RegisterPolicy_>::noneOf() const noexcept {
	return implementation::noneOf(_mask);
}

template <
	arch::CpuFeature	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_>
constexpr basic_simd_mask<_SimdGeneration_, _Element_, _RegisterPolicy_>::size_type
	basic_simd_mask<_SimdGeneration_, _Element_, _RegisterPolicy_>::countSet() const noexcept
{
	return implementation::countSet(_mask);
}

template <
	arch::CpuFeature	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_>
constexpr basic_simd_mask<_SimdGeneration_, _Element_, _RegisterPolicy_>::size_type 
	basic_simd_mask<_SimdGeneration_, _Element_, _RegisterPolicy_>::countTrailingZeroBits() const noexcept
{
	return implementation::countTrailingZeroBits(_mask);
}

template <
	arch::CpuFeature	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_>
constexpr basic_simd_mask<_SimdGeneration_, _Element_, _RegisterPolicy_>::size_type
	basic_simd_mask<_SimdGeneration_, _Element_, _RegisterPolicy_>::countLeadingZeroBits() const noexcept 
{
	return implementation::countLeadingZeroBits(_mask);
}

template <
	arch::CpuFeature	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_>
constexpr basic_simd_mask<_SimdGeneration_, _Element_, _RegisterPolicy_>::size_type 
	basic_simd_mask<_SimdGeneration_, _Element_, _RegisterPolicy_>::countTrailingOneBits() const noexcept
{
	return implementation::countTrailingZeroBits(~_mask);
}

template <
	arch::CpuFeature	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_>
constexpr basic_simd_mask<_SimdGeneration_, _Element_, _RegisterPolicy_>::size_type 
	basic_simd_mask<_SimdGeneration_, _Element_, _RegisterPolicy_>::countLeadingOneBits() const noexcept 
{
	return implementation::countLeadingZeroBits(~_mask);
}

template <
	arch::CpuFeature	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_>
constexpr void basic_simd_mask<_SimdGeneration_, _Element_, _RegisterPolicy_>::clearLeftMostSetBit() noexcept {
	_mask = _mask & (_mask - 1);
}

template <
	arch::CpuFeature	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_>
constexpr basic_simd_mask<_SimdGeneration_, _Element_, _RegisterPolicy_>::mask_type 
	basic_simd_mask<_SimdGeneration_, _Element_, _RegisterPolicy_>::unwrap() const noexcept 
{
	return _mask;
}

template <
	arch::CpuFeature	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_>
constexpr bool basic_simd_mask<_SimdGeneration_, _Element_, _RegisterPolicy_>
	::operator==(const basic_simd_mask& other) const noexcept
{
	return _mask == other._mask;
}

template <
	arch::CpuFeature	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_>
constexpr bool basic_simd_mask<_SimdGeneration_, _Element_, _RegisterPolicy_>
	::operator!=(const basic_simd_mask& other) const noexcept 
{
	return _mask != other._mask;
}

template <
	arch::CpuFeature	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_>
constexpr bool basic_simd_mask<_SimdGeneration_, _Element_, _RegisterPolicy_>
	::operator<(const basic_simd_mask& other) const noexcept
{
	return _mask < other._mask;
}

template <
	arch::CpuFeature	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_>
constexpr bool basic_simd_mask<_SimdGeneration_, _Element_, _RegisterPolicy_>
	::operator<=(const basic_simd_mask& other) const noexcept
{
	return _mask <= other._mask;
}

template <
	arch::CpuFeature	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_>
constexpr bool basic_simd_mask<_SimdGeneration_, _Element_, _RegisterPolicy_>
	::operator>(const basic_simd_mask& other) const noexcept
{
	return _mask > other._mask;
}

template <
	arch::CpuFeature	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_>
constexpr bool basic_simd_mask<_SimdGeneration_, _Element_, _RegisterPolicy_>
	::operator>=(const basic_simd_mask& other) const noexcept 
{
	return _mask >= other._mask;
}

template <
	arch::CpuFeature	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_>
constexpr basic_simd_mask<_SimdGeneration_, _Element_, _RegisterPolicy_> 
	basic_simd_mask<_SimdGeneration_, _Element_, _RegisterPolicy_>::operator~() const noexcept
{
	return basic_simd_mask{ mask_type(~_mask) };
}

template <
	arch::CpuFeature	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_>
constexpr basic_simd_mask<_SimdGeneration_, _Element_, _RegisterPolicy_> 
	basic_simd_mask<_SimdGeneration_, _Element_, _RegisterPolicy_>::operator&(const basic_simd_mask& other) const noexcept
{
	return basic_simd_mask{ mask_type(_mask & other._mask) };
}

template <
	arch::CpuFeature	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_>
constexpr basic_simd_mask<_SimdGeneration_, _Element_, _RegisterPolicy_> 
	basic_simd_mask<_SimdGeneration_, _Element_, _RegisterPolicy_>::operator|(const basic_simd_mask& other) const noexcept
{
	return basic_simd_mask{ mask_type(_mask | other._mask) };
}

template <
	arch::CpuFeature	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_>
constexpr basic_simd_mask<_SimdGeneration_, _Element_, _RegisterPolicy_> 
	basic_simd_mask<_SimdGeneration_, _Element_, _RegisterPolicy_>::operator^(const basic_simd_mask& other) const noexcept
{
	return basic_simd_mask{ mask_type(_mask ^ other._mask) };
}

template <
	arch::CpuFeature	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_>
constexpr bool basic_simd_mask<_SimdGeneration_, _Element_, _RegisterPolicy_>::operator[](int32 _Index) const noexcept {
	return ((_mask >> _Index) & 1);
}

template <
	arch::CpuFeature	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_>
constexpr basic_simd_mask<_SimdGeneration_, _Element_, _RegisterPolicy_>&
	basic_simd_mask<_SimdGeneration_, _Element_, _RegisterPolicy_>::operator&=(const basic_simd_mask& other) noexcept 
{
	_mask &= other._mask;
	return *this;
}

template <
	arch::CpuFeature	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_>
constexpr basic_simd_mask<_SimdGeneration_, _Element_, _RegisterPolicy_>& 
	basic_simd_mask<_SimdGeneration_, _Element_, _RegisterPolicy_>::operator|=(const basic_simd_mask& other) noexcept
{
	_mask |= other._mask;
	return *this;
}

template <
	arch::CpuFeature	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_>
constexpr basic_simd_mask<_SimdGeneration_, _Element_, _RegisterPolicy_>& 
	basic_simd_mask<_SimdGeneration_, _Element_, _RegisterPolicy_>::operator=(mask_type other) noexcept 
{
	_mask = other;
	return *this;
}

template <
	arch::CpuFeature	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_>
constexpr basic_simd_mask<_SimdGeneration_, _Element_, _RegisterPolicy_>& 
	basic_simd_mask<_SimdGeneration_, _Element_, _RegisterPolicy_>::operator^=(const basic_simd_mask& other) noexcept
{
	_mask ^= other._mask;
	return *this;
}

template <
	arch::CpuFeature	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_>
constexpr basic_simd_mask<_SimdGeneration_, _Element_, _RegisterPolicy_>::operator bool() const noexcept {
	return anyOf();
}

__SIMD_STL_NUMERIC_NAMESPACE_END

