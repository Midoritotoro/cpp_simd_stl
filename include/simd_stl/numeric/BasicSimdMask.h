#pragma once 

#include <simd_stl/numeric/BasicSimdMaskImplementation.h>


__SIMD_STL_NUMERIC_NAMESPACE_BEGIN

template <
	arch::CpuFeature	_SimdGeneration_,
	typename			_Element_>
class basic_simd_mask {
	static_assert(type_traits::__is_generation_supported_v<_SimdGeneration_>);
	static_assert(type_traits::__is_vector_type_supported_v<_Element_>);

	using __impl = BasicSimdMaskImplementation<_SimdGeneration_, _Element_>
public:
	using mask_type = typename __impl::mask_type;
	using size_type = typename __impl::size_type;

	simd_stl_constexpr_cxx20 simd_stl_always_inline bool all_of() const noexcept {

	}

	simd_stl_constexpr_cxx20 simd_stl_always_inline bool any_of() const noexcept {

	}

	simd_stl_constexpr_cxx20 simd_stl_always_inline bool none_of() const noexcept {

	}

	simd_stl_constexpr_cxx20 simd_stl_always_inline size_type countSet() const noexcept {

	}

	simd_stl_constexpr_cxx20 simd_stl_always_inline size_type countTrailingZeroBits() const noexcept {

	}

	simd_stl_constexpr_cxx20 simd_stl_always_inline size_type countLeadingZeroBits() const noexcept {

	}

private:
	mask_type _mask = 0;
};

__SIMD_STL_NUMERIC_NAMESPACE_END

