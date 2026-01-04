#pragma once 

__SIMD_STL_NUMERIC_NAMESPACE_BEGIN

template <
	arch::CpuFeature	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_>
constexpr bool __simd_mask_implementation<_SimdGeneration_, _Element_, _RegisterPolicy_>::all_of(mask_type __mask) noexcept {
	if constexpr (__used_bits == (sizeof(mask_type) << 3))
		return (__mask == math::__maximum_integral_limit<mask_type>());
	else
		return __mask == ((mask_type(1) << __used_bits) - 1);
}

template <
	arch::CpuFeature	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_>
constexpr bool __simd_mask_implementation<_SimdGeneration_, _Element_, _RegisterPolicy_>::any_of(mask_type __mask) noexcept {
	return (__mask != 0);
}

template <
	arch::CpuFeature	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_>
constexpr bool __simd_mask_implementation<_SimdGeneration_, _Element_, _RegisterPolicy_>::none_of(mask_type __mask) noexcept {
	return (__mask == 0);
}

template <
	arch::CpuFeature	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_>
constexpr uint8 __simd_mask_implementation<_SimdGeneration_, _Element_, _RegisterPolicy_>::count_set(mask_type __mask) noexcept {
	return math::population_count(__mask);
}

template <
	arch::CpuFeature	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_>
constexpr uint8 __simd_mask_implementation<_SimdGeneration_, _Element_, _RegisterPolicy_>::count_trailing_zero_bits(mask_type __mask) noexcept {
	return math::count_trailing_zero_bits(__mask);
}

template <
	arch::CpuFeature	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_>
constexpr uint8 __simd_mask_implementation<_SimdGeneration_, _Element_, _RegisterPolicy_>::count_leading_zero_bits(mask_type __mask) noexcept {
	return math::count_leading_zero_bits(__mask);
}


__SIMD_STL_NUMERIC_NAMESPACE_END

