__SIMD_STL_NUMERIC_NAMESPACE_BEGIN

template <class _Derived_>
constexpr uint8 __simd_index_mask_operations<_Derived_>::__divisor() noexcept {
	return __self::divisor();
}

template <class _Derived_>
constexpr uint8 __simd_index_mask_operations<_Derived_>::count_set() const noexcept {
	return math::__popcnt_n_bits<__base::__bit_width()>(__base::__to_int()) / __divisor();
}

template <class _Derived_>
constexpr uint8 __simd_index_mask_operations<_Derived_>::count_trailing_zero_bits() const noexcept {
	if constexpr (__has_avx2_support_v<__base::__generation()>)
		return math::__tzcnt_ctz_unsafe(__base::__to_int()) / __divisor();
	else
		return math::__bsf_ctz_unsafe(__base::__to_int()) / __divisor();
}

template <class _Derived_>
constexpr uint8 __simd_index_mask_operations<_Derived_>::count_leading_zero_bits() const noexcept {
	if constexpr (__has_avx2_support_v<__base::__generation()>)
		return math::__lzcnt_clz(__base::__to_int()) / __divisor();
	else
		return (math::__bsr_clz(__base::__to_int())) / __divisor();
}

__SIMD_STL_NUMERIC_NAMESPACE_END
