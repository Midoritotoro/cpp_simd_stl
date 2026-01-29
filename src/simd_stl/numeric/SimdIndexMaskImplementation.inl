#pragma once 

__SIMD_STL_NUMERIC_NAMESPACE_BEGIN

template <
	arch::CpuFeature	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_>
constexpr bool __simd_index_mask_implementation<_SimdGeneration_, _Element_, _RegisterPolicy_>::all_of(mask_type __mask) noexcept {
	if constexpr (__used_bits == (sizeof(mask_type) << 3))
		return (__mask == math::__maximum_integral_limit<mask_type>());
	else {
		const auto __max = mask_type(((mask_type(1) << __used_bits) - 1));
		return __mask == __max;
	}
}

template <
	arch::CpuFeature	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_>
constexpr bool __simd_index_mask_implementation<_SimdGeneration_, _Element_, _RegisterPolicy_>::any_of(mask_type __mask) noexcept {
	return (__mask != 0);
}

template <
	arch::CpuFeature	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_>
constexpr bool __simd_index_mask_implementation<_SimdGeneration_, _Element_, _RegisterPolicy_>::none_of(mask_type __mask) noexcept {
	return (__mask == 0);
}

template <
	arch::CpuFeature	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_>
constexpr uint8 __simd_index_mask_implementation<_SimdGeneration_, _Element_, _RegisterPolicy_>::count_set(mask_type __mask) noexcept {
	return math::__popcnt_n_bits<__used_bits>(__mask) / __divisor;
}

template <
	arch::CpuFeature	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_>
constexpr uint8 __simd_index_mask_implementation<_SimdGeneration_, _Element_, _RegisterPolicy_>::count_trailing_zero_bits(mask_type __mask) noexcept {
	return math::__ctz_n_bits<__used_bits>(__mask) / __divisor;
}

template <
	arch::CpuFeature	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_>
constexpr uint8 __simd_index_mask_implementation<_SimdGeneration_, _Element_, _RegisterPolicy_>::count_leading_zero_bits(mask_type __mask) noexcept {
	return math::__clz_n_bits<__used_bits>(__mask) / __divisor;
}


__SIMD_STL_NUMERIC_NAMESPACE_END

