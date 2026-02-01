#pragma once 

__SIMD_STL_NUMERIC_NAMESPACE_BEGIN

template <
	arch::CpuFeature	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_>
constexpr bool __simd_index_mask_implementation<_SimdGeneration_, _Element_, _RegisterPolicy_>::all_of(mask_type __mask) noexcept {
	static_assert(__used_bits <= 64);

	if constexpr (__is_k_register && sizeof(mask_type) == 2)
		return _kortestz_mask16_u8(__mask, static_cast<__mmask16>((__mmask16(1) << __used_bits) - 1));

	else if constexpr (__is_k_register && sizeof(mask_type) == 1 && __has_avx512dq_support_v<_SimdGeneration_>)
		return _kortestz_mask8_u8(__mask, static_cast<__mmask8>((__mmask8(1) << __used_bits) - 1));

	else if constexpr (__is_k_register && sizeof(mask_type) == 4 && __has_avx512bw_support_v<_SimdGeneration_>)
		return _kortestz_mask32_u8(__mask, static_cast<__mmask32>((__mmask32(1) << __used_bits) - 1));

	else if constexpr (__is_k_register && sizeof(mask_type) == 8 && __has_avx512bw_support_v<_SimdGeneration_>)
		return _kortestz_mask64_u8(__mask, static_cast<__mmask64>((__mmask64(1) << __used_bits) - 1));

	else
		return (__mask == mask_type(((mask_type(1) << __used_bits) - 1)));
}

template <
	arch::CpuFeature	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_>
constexpr bool __simd_index_mask_implementation<_SimdGeneration_, _Element_, _RegisterPolicy_>::any_of(mask_type __mask) noexcept {
	return !none_of(__mask);
}

template <
	arch::CpuFeature	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_>
constexpr bool __simd_index_mask_implementation<_SimdGeneration_, _Element_, _RegisterPolicy_>::none_of(mask_type __mask) noexcept {
	if constexpr (__is_k_register && sizeof(mask_type) == 2)
		return _kortestz_mask16_u8(__mask, __mask);

	else if constexpr (__is_k_register && sizeof(mask_type) == 1 && __has_avx512dq_support_v<_SimdGeneration_>)
		return _kortestz_mask8_u8(__mask, __mask);

	else if constexpr (__is_k_register && sizeof(mask_type) == 4 && __has_avx512bw_support_v<_SimdGeneration_>)
		return _kortestz_mask32_u8(__mask, __mask);

	else if constexpr (__is_k_register && sizeof(mask_type) == 8 && __has_avx512bw_support_v<_SimdGeneration_>)
		return _kortestz_mask64_u8(__mask, __mask);

	else
		return (__mask == 0);
}

template <
	arch::CpuFeature	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_>
constexpr uint8 __simd_index_mask_implementation<_SimdGeneration_, _Element_, _RegisterPolicy_>::count_set(mask_type __mask) noexcept {
	return math::__popcnt_n_bits<__used_bits>(__kmask_to_int(__mask)) / __divisor;
}

template <
	arch::CpuFeature	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_>
constexpr uint8 __simd_index_mask_implementation<_SimdGeneration_, _Element_, _RegisterPolicy_>::count_trailing_zero_bits(mask_type __mask) noexcept {
	if constexpr (static_cast<int>(_SimdGeneration_) >= static_cast<int>(arch::CpuFeature::AVX2))
		return math::__tzcnt_ctz_unsafe(__kmask_to_int(__mask)) / __divisor;
	else
		return math::__bsf_ctz_unsafe(__kmask_to_int(__mask)) / __divisor;
}

template <
	arch::CpuFeature	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_>
constexpr uint8 __simd_index_mask_implementation<_SimdGeneration_, _Element_, _RegisterPolicy_>::count_leading_zero_bits(mask_type __mask) noexcept {
	if constexpr (static_cast<int>(_SimdGeneration_) >= static_cast<int>(arch::CpuFeature::AVX2))
		return math::__lzcnt_clz(__kmask_to_int(__mask)) / __divisor;
	else
		return (math::__bsr_clz(__kmask_to_int(__mask)) - ((sizeof(mask_type) * 8) - __used_bits)) / __divisor;
}

template <
	arch::CpuFeature	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_>
constexpr uint8 __simd_index_mask_implementation<_SimdGeneration_, _Element_, _RegisterPolicy_>::count_trailing_one_bits(mask_type __mask) noexcept {
	return count_trailing_zero_bits(~__mask);
}

template <
	arch::CpuFeature	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_>
constexpr uint8 __simd_index_mask_implementation<_SimdGeneration_, _Element_, _RegisterPolicy_>::count_leading_one_bits(mask_type __mask) noexcept {
	return count_leading_zero_bits(~__mask);
}

template <
	arch::CpuFeature	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_>
constexpr __simd_index_mask_implementation<_SimdGeneration_, _Element_, _RegisterPolicy_>::mask_type 
	__simd_index_mask_implementation<_SimdGeneration_, _Element_, _RegisterPolicy_>::__bit_and(mask_type __first, mask_type __second) noexcept 
{
	if constexpr (__is_k_register && sizeof(mask_type) == 1 && __has_avx512dq_support_v<_SimdGeneration_>)
		return _kand_mask8(__first, __second);

	else if constexpr (__is_k_register && sizeof(mask_type) == 2)
		return _kand_mask16(__first, __second);

	else if constexpr (__is_k_register && sizeof(mask_type) == 4 && __has_avx512bw_support_v<_SimdGeneration_>)
		return _kand_mask32(__first, __second);

	else if constexpr (__is_k_register && sizeof(mask_type) == 8 && __has_avx512bw_support_v<_SimdGeneration_>)
		return _kand_mask64(__first, __second);

	else
		return __first & __second;
}

template <
	arch::CpuFeature	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_>
constexpr __simd_index_mask_implementation<_SimdGeneration_, _Element_, _RegisterPolicy_>::mask_type 
	__simd_index_mask_implementation<_SimdGeneration_, _Element_, _RegisterPolicy_>::__bit_or(mask_type __first, mask_type __second) noexcept
{
	if constexpr (__is_k_register && sizeof(mask_type) == 1 && __has_avx512dq_support_v<_SimdGeneration_>)
		return _kor_mask8(__first, __second);

	else if constexpr (__is_k_register && sizeof(mask_type) == 2)
		return _kor_mask16(__first, __second);

	else if constexpr (__is_k_register && sizeof(mask_type) == 4 && __has_avx512bw_support_v<_SimdGeneration_>)
		return _kor_mask32(__first, __second);

	else if constexpr (__is_k_register && sizeof(mask_type) == 8 && __has_avx512bw_support_v<_SimdGeneration_>)
		return _kor_mask64(__first, __second);

	else
		return __first | __second;
}

template <
	arch::CpuFeature	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_>
constexpr __simd_index_mask_implementation<_SimdGeneration_, _Element_, _RegisterPolicy_>::mask_type 
	__simd_index_mask_implementation<_SimdGeneration_, _Element_, _RegisterPolicy_>::__bit_xor(mask_type __first, mask_type __second) noexcept
{
	if constexpr (__is_k_register && sizeof(mask_type) == 1 && __has_avx512dq_support_v<_SimdGeneration_>)
		return _kxor_mask8(__first, __second);

	else if constexpr (__is_k_register && sizeof(mask_type) == 2)
		return _kxor_mask16(__first, __second);

	else if constexpr (__is_k_register && sizeof(mask_type) == 4 && __has_avx512bw_support_v<_SimdGeneration_>)
		return _kxor_mask32(__first, __second);

	else if constexpr (__is_k_register && sizeof(mask_type) == 8 && __has_avx512bw_support_v<_SimdGeneration_>)
		return _kxor_mask64(__first, __second);

	else
		return __first ^ __second;
}

template <
	arch::CpuFeature	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_>
constexpr __simd_index_mask_implementation<_SimdGeneration_, _Element_, _RegisterPolicy_>::mask_type
	__simd_index_mask_implementation<_SimdGeneration_, _Element_, _RegisterPolicy_>::__bit_not(mask_type __mask) noexcept
{
	if constexpr (__is_k_register && sizeof(mask_type) == 1 && __has_avx512dq_support_v<_SimdGeneration_>)
		return _knot_mask8(__mask);

	else if constexpr (__is_k_register && sizeof(mask_type) == 2)
		return _knot_mask16(__mask);

	else if constexpr (__is_k_register && sizeof(mask_type) == 4 && __has_avx512bw_support_v<_SimdGeneration_>)
		return _knot_mask32(__mask);

	else if constexpr (__is_k_register && sizeof(mask_type) == 8 && __has_avx512bw_support_v<_SimdGeneration_>)
		return _knot_mask64(__mask);

	else
		return ~__mask;
}

template <
	arch::CpuFeature	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_>
template <class _Type_>
constexpr auto __simd_index_mask_implementation<_SimdGeneration_, _Element_, _RegisterPolicy_>::__kmask_to_int(_Type_ __k) noexcept {
	if constexpr (__is_k_register && sizeof(_Type_) == 1 && __has_avx512dq_support_v<_SimdGeneration_>)
		return _cvtmask8_u32(__k);

	else if constexpr (__is_k_register && sizeof(_Type_) == 2 && __has_avx512f_support_v<_SimdGeneration_>)
		return _cvtmask16_u32(__k);

	else if constexpr (__is_k_register && sizeof(_Type_) == 4 && __has_avx512bw_support_v<_SimdGeneration_>)
		return _cvtmask32_u32(__k);

	else if constexpr (__is_k_register && sizeof(_Type_) == 8 && __has_avx512bw_support_v<_SimdGeneration_>)
		return _cvtmask64_u64(__k);

	else
		return __k;
}

template <
	arch::CpuFeature	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_>
template <class _Type_>
constexpr auto __simd_index_mask_implementation<_SimdGeneration_, _Element_, _RegisterPolicy_>::__int_to_kmask(_Type_ __integer) noexcept {
	if constexpr (__is_k_register && sizeof(_Type_) == 1 && __has_avx512dq_support_v<_SimdGeneration_>)
		return _cvtu32_mask8(__k);

	else if constexpr (__is_k_register && sizeof(_Type_) == 2 && __has_avx512f_support_v<_SimdGeneration_>)
		return _cvtu32_mask16(__k);

	else if constexpr (__is_k_register && sizeof(_Type_) == 4 && __has_avx512bw_support_v<_SimdGeneration_>)
		return _cvtu32_mask32(__k);

	else if constexpr (__is_k_register && sizeof(_Type_) == 8 && __has_avx512bw_support_v<_SimdGeneration_>)
		return _cvtu64_mask64(__k);

	else
		return __k;
}

__SIMD_STL_NUMERIC_NAMESPACE_END

