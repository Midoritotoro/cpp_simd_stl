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
	return static_cast<bool>(*this | as_index_mask);
}

template <
	arch::CpuFeature	_SimdGeneration_,
	class				_Element_,
	class				_RegisterPolicy_,
	__simd_comparison	_Comparison_>
simd_compare_result<_SimdGeneration_, _Element_, _RegisterPolicy_, _Comparison_> 
	simd_compare_result<_SimdGeneration_, _Element_, _RegisterPolicy_, _Comparison_>::operator~() noexcept
{
	return ~_compare_result;
}

__SIMD_STL_NUMERIC_NAMESPACE_END
