#pragma once 

__SIMD_STL_NUMERIC_NAMESPACE_BEGIN

template <
	arch::CpuFeature	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_>
constexpr bool BasicSimdMaskImplementation<_SimdGeneration_, _Element_, _RegisterPolicy_>::allOf(mask_type _Mask) noexcept {
	if constexpr (_UsedBits == (sizeof(mask_type) << 3))
		return (_Mask == math::__maximum_integral_limit<mask_type>());
	else
		return _Mask == ((mask_type(1) << _UsedBits) - 1);
}

template <
	arch::CpuFeature	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_>
constexpr bool BasicSimdMaskImplementation<_SimdGeneration_, _Element_, _RegisterPolicy_>::anyOf(mask_type _Mask) noexcept {
	return (_Mask != 0);
}

template <
	arch::CpuFeature	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_>
constexpr bool BasicSimdMaskImplementation<_SimdGeneration_, _Element_, _RegisterPolicy_>::noneOf(mask_type _Mask) noexcept {
	return (_Mask == 0);
}

template <
	arch::CpuFeature	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_>
constexpr uint8 BasicSimdMaskImplementation<_SimdGeneration_, _Element_, _RegisterPolicy_>::countSet(mask_type _Mask) noexcept {
	return math::PopulationCount(_Mask);
}

template <
	arch::CpuFeature	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_>
constexpr uint8 BasicSimdMaskImplementation<_SimdGeneration_, _Element_, _RegisterPolicy_>::countTrailingZeroBits(mask_type _Mask) noexcept {
	return math::CountTrailingZeroBits(_Mask);
}

template <
	arch::CpuFeature	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_>
constexpr uint8 BasicSimdMaskImplementation<_SimdGeneration_, _Element_, _RegisterPolicy_>::countLeadingZeroBits(mask_type _Mask) noexcept {
	return math::CountLeadingZeroBits(_Mask);
}


__SIMD_STL_NUMERIC_NAMESPACE_END

