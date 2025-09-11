#pragma once 

#include <simd_stl/numeric/BasicSimdMaskImplementation.h>
#include <src/simd_stl/math/BitMath.h>


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

	simd_stl_constexpr_cxx20 simd_stl_always_inline bool allOf() const noexcept {
		return __impl::allOf(_mask);
	}

	simd_stl_constexpr_cxx20 simd_stl_always_inline bool anyOf() const noexcept {
		return __impl::anyOf(_mask);
	}

	simd_stl_constexpr_cxx20 simd_stl_always_inline bool noneOf() const noexcept {
		return __impl::noneOf(_mask);
	}

	simd_stl_constexpr_cxx20 simd_stl_always_inline size_type countSet() const noexcept {
		return __impl::countSet(_mask);
	}

	simd_stl_constexpr_cxx20 simd_stl_always_inline size_type countTrailingZeroBits() const noexcept {
		return __impl::countTrailingZeroBits(_mask);
	}

	simd_stl_constexpr_cxx20 simd_stl_always_inline size_type countLeadingZeroBits() const noexcept {
		return __impl::countLeadingZeroBits(_mask);
	}

	basic_simd_mask& operator>>=(const int8 shift) noexcept;
private:
	mask_type _mask = 0;
};

//basic_simd_mask& operator>>(const int8 shift) noexcept {

//}

__SIMD_STL_NUMERIC_NAMESPACE_END

