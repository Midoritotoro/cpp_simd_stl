#pragma once 

#include <src/simd_stl/type_traits/SimdTypeCheck.h>
#include <simd_stl/compatibility/Inline.h>

#include <simd_stl/math/BitMath.h>
#include <simd_stl/math/IntegralTypesConversions.h>

#include <bitset>


__SIMD_STL_NUMERIC_NAMESPACE_BEGIN



template <
	arch::CpuFeature	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_ = numeric::_DefaultRegisterPolicy<_SimdGeneration_>>
class BasicSimdMaskImplementation {
public:
	static_assert(type_traits::__is_generation_supported_v<_SimdGeneration_>);
	static_assert(type_traits::__is_vector_type_supported_v<_Element_>);

	using mask_type = type_traits::__deduce_simd_mask_type<_SimdGeneration_, _Element_, _RegisterPolicy_>;
	using size_type = uint8;

	static constexpr uint8 usedBits = sizeof(type_traits::__deduce_simd_vector_type<_SimdGeneration_, _Element_, _RegisterPolicy_>) / sizeof(_Element_);

	static constexpr simd_stl_always_inline bool allOf(const mask_type mask) noexcept {
		if constexpr (usedBits == (sizeof(mask_type) << 3))
			return (mask == math::MaximumIntegralLimit<mask_type>());
		else
			return mask == ((mask_type(1) << usedBits) - 1);
	}

	static constexpr simd_stl_always_inline bool anyOf(const mask_type mask) noexcept {
		return (mask != 0);
	}

	static constexpr simd_stl_always_inline bool noneOf(const mask_type mask) noexcept {
		return (mask == 0);
	}

	static constexpr simd_stl_always_inline uint8 countSet(const mask_type mask) noexcept {
		return math::PopulationCount(mask);
	}

	static constexpr simd_stl_always_inline uint8 countTrailingZeroBits(const mask_type mask) noexcept {
		return math::CountTrailingZeroBits(mask);
	}

	static constexpr simd_stl_always_inline uint8 countLeadingZeroBits(const mask_type mask) noexcept {
		return math::CountLeadingZeroBits(mask);
	}
};

__SIMD_STL_NUMERIC_NAMESPACE_END

