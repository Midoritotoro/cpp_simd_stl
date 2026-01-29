#pragma once 

#include <src/simd_stl/type_traits/SimdTypeCheck.h>
#include <simd_stl/compatibility/Inline.h>

#include <simd_stl/math/BitMath.h>
#include <simd_stl/math/IntegralTypesConversions.h>

#include <src/simd_stl/numeric/SimdConvert.h>
#include <src/simd_stl/numeric/SimdCompare.h>

#include <bitset>


__SIMD_STL_NUMERIC_NAMESPACE_BEGIN

template <
	arch::CpuFeature	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_ = numeric::__default_register_policy<_SimdGeneration_>>
class __simd_index_mask_implementation {
public:
	using __native_compare_type = __simd_native_compare_return_type<
		simd<_SimdGeneration_, _Element_, _RegisterPolicy_>, _Element_, __simd_comparison::equal>;

	static constexpr bool __native_compare_returns_intrin = __is_intrin_type_v<__native_compare_type>;
	static constexpr bool __native_compare_returns_number = std::is_integral_v<__native_compare_type>;

	static constexpr auto __calculate_divisor() noexcept {
		return __simd_index_mask_divisor<_SimdGeneration_, _RegisterPolicy_, _Element_>;
	}

	static_assert(type_traits::__is_generation_supported_v<_SimdGeneration_>);
	static_assert(type_traits::__is_vector_type_supported_v<_Element_>);

	static constexpr uint8 __divisor	= __calculate_divisor();

	template <bool _ReturnsNumber_>
	struct __mask_type {
		using type = void;
	};

	template <>
	struct __mask_type<true> {
		static constexpr auto size = _RegisterPolicy_::__width / sizeof(_Element_) * __divisor;
		using type = typename IntegerForSize<((size <= 8) ? 1 : (size / 8))>::Unsigned;
	};

	template <>
	struct __mask_type<false> {
		static constexpr auto size = _RegisterPolicy_::__width / sizeof(_Element_) * __divisor;
		using type = typename IntegerForSize<((size <= 8) ? 1 : (size / 8))>::Unsigned;
	};

	static constexpr uint8 __used_bits = __mask_type<__native_compare_returns_number>::size;

	using mask_type = typename __mask_type<__native_compare_returns_number>::type;
	using size_type = uint8;

	static constexpr simd_stl_always_inline bool all_of(mask_type __mask) noexcept;
	static constexpr simd_stl_always_inline bool any_of(mask_type __mask) noexcept;
	static constexpr simd_stl_always_inline bool none_of(mask_type __mask) noexcept;

	static constexpr simd_stl_always_inline uint8 count_set(mask_type __mask) noexcept;
	static constexpr simd_stl_always_inline uint8 count_trailing_zero_bits(mask_type __mask) noexcept;
	static constexpr simd_stl_always_inline uint8 count_leading_zero_bits(mask_type __mask) noexcept;
};

__SIMD_STL_NUMERIC_NAMESPACE_END

#include <src/simd_stl/numeric/SimdIndexMaskImplementation.inl>