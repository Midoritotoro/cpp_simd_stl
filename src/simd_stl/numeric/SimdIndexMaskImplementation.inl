#pragma once 

__SIMD_STL_NUMERIC_NAMESPACE_BEGIN

template <
	arch::CpuFeature	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_>
constexpr bool __simd_index_mask_implementation<_SimdGeneration_, _Element_, _RegisterPolicy_>::all_of(mask_type __mask) noexcept {
	return (__mask == mask_type(((mask_type(1) << __used_bits) - 1)));
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
	if constexpr (static_cast<int>(_SimdGeneration_) >= static_cast<int>(arch::CpuFeature::AVX2) && __used_bits >= 32)
		return math::__tzcnt_ctz(__mask) / __divisor;
	else
		return math::__ctz_n_bits<__used_bits>(__mask) / __divisor;
}

template <
	arch::CpuFeature	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_>
constexpr uint8 __simd_index_mask_implementation<_SimdGeneration_, _Element_, _RegisterPolicy_>::count_leading_zero_bits(mask_type __mask) noexcept {
	if constexpr (static_cast<int>(_SimdGeneration_) >= static_cast<int>(arch::CpuFeature::AVX2) && __used_bits >= 16)
		return math::__lzcnt_clz(__mask) / __divisor;
	else
		return math::__clz_n_bits<__used_bits>(__mask) / __divisor;
}


__SIMD_STL_NUMERIC_NAMESPACE_END

