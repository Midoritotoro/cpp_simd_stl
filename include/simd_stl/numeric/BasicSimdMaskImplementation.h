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
class __simd_mask_implementation {
public:
	static_assert(type_traits::__is_generation_supported_v<_SimdGeneration_>);
	static_assert(type_traits::__is_vector_type_supported_v<_Element_>);

	using mask_type = type_traits::__deduce_simd_mask_type<_SimdGeneration_, _Element_, _RegisterPolicy_>;
	using size_type = uint8;

	static constexpr uint8 __used_bits = sizeof(type_traits::__deduce_simd_vector_type<_SimdGeneration_,
		_Element_, _RegisterPolicy_>) / sizeof(_Element_);

	static constexpr simd_stl_always_inline bool all_of(mask_type __mask) noexcept;
	static constexpr simd_stl_always_inline bool any_of(mask_type __mask) noexcept;
	static constexpr simd_stl_always_inline bool none_of(mask_type __mask) noexcept;

	static constexpr simd_stl_always_inline uint8 count_set(mask_type __mask) noexcept;
	static constexpr simd_stl_always_inline uint8 count_trailing_zero_bits(mask_type __mask) noexcept;
	static constexpr simd_stl_always_inline uint8 count_leading_zero_bits(mask_type __mask) noexcept;
};

__SIMD_STL_NUMERIC_NAMESPACE_END

#include <src/simd_stl/numeric/BasicSimdMaskImplementation.inl>