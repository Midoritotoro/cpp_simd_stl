__SIMD_STL_NUMERIC_NAMESPACE_BEGIN

template <
	arch::CpuFeature	_SimdGeneration_,
	class				_Element_,
	class				_RegisterPolicy_,
	__simd_comparison	_Comparison_>
simd_compare_result<_SimdGeneration_, _Element_, _RegisterPolicy_, _Comparison_>::simd_compare_result(const native_type& __result) noexcept :
	_compare_result(__result)
{}

template <
	arch::CpuFeature	_SimdGeneration_,
	class				_Element_,
	class				_RegisterPolicy_,
	__simd_comparison	_Comparison_>
simd_compare_result<_SimdGeneration_, _Element_, _RegisterPolicy_, _Comparison_>::operator bool() const noexcept {
	return __simd_to_index_mask<_SimdGeneration_, _RegisterPolicy_, _Element_>(__simd_unwrap(_compare_result)) != 0;
}

__SIMD_STL_NUMERIC_NAMESPACE_END
