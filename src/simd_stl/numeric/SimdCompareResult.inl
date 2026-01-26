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
template <class _DesiredType_>
simd_stl_always_inline simd_index_mask<__simd_index_mask_divisor<_SimdGeneration_, _RegisterPolicy_, _DesiredType_>, _SimdGeneration_, _DesiredType_, _RegisterPolicy_> 
	simd_compare_result<_SimdGeneration_, _Element_, _RegisterPolicy_, _Comparison_>::as_index_mask() const noexcept
{
	return __simd_to_index_mask<_SimdGeneration_, _RegisterPolicy_, _DesiredType_>(_compare_result);
}

template <
	arch::CpuFeature	_SimdGeneration_,
	class				_Element_,
	class				_RegisterPolicy_,
	__simd_comparison	_Comparison_>
template <class _DesiredType_>
simd_stl_always_inline simd_mask<_SimdGeneration_, _DesiredType_, _RegisterPolicy_> 
	simd_compare_result<_SimdGeneration_, _Element_, _RegisterPolicy_, _Comparison_>::as_mask() const noexcept
{
	return __simd_to_mask<_SimdGeneration_, _RegisterPolicy_, _DesiredType_>(_compare_result);
}

template <
	arch::CpuFeature	_SimdGeneration_,
	class				_Element_,
	class				_RegisterPolicy_,
	__simd_comparison	_Comparison_>
template <class _DesiredType_>
simd_stl_always_inline simd<_SimdGeneration_, _DesiredType_, _RegisterPolicy_>
	simd_compare_result<_SimdGeneration_, _Element_, _RegisterPolicy_, _Comparison_>::as_simd() const noexcept 
{
	return __simd_to_vector<_SimdGeneration_, _RegisterPolicy_, typename simd<_SimdGeneration_,
		_DesiredType_, _RegisterPolicy_>::vector_type, _DesiredType_>(_compare_result);
}

template <
	arch::CpuFeature	_SimdGeneration_,
	class				_Element_,
	class				_RegisterPolicy_,
	__simd_comparison	_Comparison_>
simd_stl_always_inline simd_compare_result<_SimdGeneration_, _Element_, _RegisterPolicy_, _Comparison_>::native_type 
	simd_compare_result<_SimdGeneration_, _Element_, _RegisterPolicy_, _Comparison_>::native() const noexcept
{
	return _compare_result;
}

__SIMD_STL_NUMERIC_NAMESPACE_END
