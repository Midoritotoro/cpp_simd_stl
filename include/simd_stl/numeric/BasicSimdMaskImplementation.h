#pragma once 

#include <src/simd_stl/type_traits/SimdTypeCheck.h>
#include <simd_stl/compatibility/Inline.h>

#include <src/simd_stl/math/BitMath.h>
#include <src/simd_stl/math/IntegralTypesConversions.h>

#include <bitset>


__SIMD_STL_NUMERIC_NAMESPACE_BEGIN

template <uint8 size>
using __deduce_simd_mask_type_helper =
	std::conditional_t<size <= 1, uint8_t,
		std::conditional_t<size <= 2, uint16_t,
			std::conditional_t<size <= 4, uint32_t,
				std::conditional_t<size <= 8, uint64_t, void>>>>;

template <
	arch::CpuFeature	_SimdGeneration_,
	typename			_Element_>
using __deduce_simd_mask_type =  __deduce_simd_mask_type_helper<(sizeof(type_traits::__deduce_simd_vector_type<_SimdGeneration_, _Element_>) / sizeof(_Element_))>;


template <
	arch::CpuFeature	_SimdGeneration_,
	typename			_Element_>
class BasicSimdMaskImplementation {
	static_assert(type_traits::__is_generation_supported_v<_SimdGeneration_>);
	static_assert(type_traits::__is_vector_type_supported_v<_Element_>);

	using mask_type = __deduce_simd_mask_type<_SimdGeneration_, _Element_>;
	static constexpr uint8 usedBits = sizeof(type_traits::__deduce_simd_vector_type<_SimdGeneration_, _Element_>) / sizeof(_Element_);

	static simd_stl_constexpr_cxx20 simd_stl_always_inline bool allOf(const mask_type mask) noexcept {
		return (mask == math::MaximumIntegralLimit<mask_type>());
	}

	static simd_stl_constexpr_cxx20 simd_stl_always_inline bool anyOf(const mask_type mask) noexcept {
		return (mask != 0);
	}

	static simd_stl_constexpr_cxx20 simd_stl_always_inline bool noneOf(const mask_type mask) noexcept {
		return (mask == 0);
	}

	static simd_stl_constexpr_cxx20 simd_stl_always_inline uint8 countSet(const mask_type mask) noexcept {
		return math::PopulationCount(mask);
	}

	static simd_stl_constexpr_cxx20 simd_stl_always_inline uint8 countTrailingZeroBits(const mask_type mask) noexcept {
		return math::CountTrailingZeroBits(mask);
	}

	static simd_stl_constexpr_cxx20 simd_stl_always_inline uint8 countLeadingZeroBits(const mask_type mask) noexcept {
		return math::CountLeadingZeroBits(mask);
	}

};

__SIMD_STL_NUMERIC_NAMESPACE_END

