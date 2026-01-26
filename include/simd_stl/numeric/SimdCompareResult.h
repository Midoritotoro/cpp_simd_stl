#pragma once 

#include <src/simd_stl/numeric/SimdCompare.h>


__SIMD_STL_NUMERIC_NAMESPACE_BEGIN

template <
	arch::CpuFeature	_SimdGeneration_,
	class				_Element_,
	class				_RegisterPolicy_,
	__simd_comparison	_Comparison_>
class simd_compare_result {
public:
	static inline constexpr auto __generation = _SimdGeneration_;
	
	using element_type		= _Element_;
	using register_policy	= _RegisterPolicy_;

	using native_type		= __simd_native_compare_return_type<simd<_SimdGeneration_, element_type, _RegisterPolicy_>, element_type, _Comparison_>;

	simd_compare_result(const native_type& __result) noexcept;

	simd_compare_result(const simd_compare_result&) = delete;
	simd_compare_result& operator=(const simd_compare_result&) = delete;
	
	simd_compare_result(simd_compare_result&&) noexcept = default; 
	simd_compare_result& operator=(simd_compare_result&&) noexcept = default;

	template <class _DesiredType_ = element_type>
	simd_stl_always_inline simd_index_mask<__simd_index_mask_divisor<_SimdGeneration_, _RegisterPolicy_, _DesiredType_>,
		_SimdGeneration_, _DesiredType_, _RegisterPolicy_> as_index_mask() const noexcept;

	template <class _DesiredType_ = element_type>
	simd_stl_always_inline simd_mask<_SimdGeneration_, _DesiredType_, _RegisterPolicy_> as_mask() const noexcept;

	template <class _DesiredType_ = element_type>
	simd_stl_always_inline simd<_SimdGeneration_, _DesiredType_, _RegisterPolicy_> as_simd() const noexcept;

	simd_stl_always_inline native_type native() const noexcept;
private:
	native_type _compare_result;
};

__SIMD_STL_NUMERIC_NAMESPACE_END

#include <src/simd_stl/numeric/SimdCompareResult.inl>
