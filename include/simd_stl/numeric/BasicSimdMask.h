#pragma once 

#include <simd_stl/numeric/BasicSimdMaskImplementation.h>
#include <simd_stl/math/BitMath.h>


__SIMD_STL_NUMERIC_NAMESPACE_BEGIN

template <
	arch::CpuFeature	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_ = numeric::__default_register_policy<_SimdGeneration_>>
class simd_mask {
	static_assert(type_traits::__is_generation_supported_v<_SimdGeneration_>);
	static_assert(type_traits::__is_vector_type_supported_v<_Element_>);

	using __implementation = __simd_mask_implementation<_SimdGeneration_, _Element_, _RegisterPolicy_>;
public:
	using mask_type = typename __implementation::mask_type;
	using size_type = typename __implementation::size_type;

	simd_mask() noexcept;
	simd_mask(const mask_type __mask) noexcept;
	
	constexpr simd_stl_always_inline bool all_of() const noexcept;
	constexpr simd_stl_always_inline bool any_of() const noexcept;
	constexpr simd_stl_always_inline bool none_of() const noexcept;

	constexpr simd_stl_always_inline size_type count_set() const noexcept;
	constexpr simd_stl_always_inline size_type count_trailing_zero_bits() const noexcept;
	constexpr simd_stl_always_inline size_type count_leading_zero_bits() const noexcept;
	constexpr simd_stl_always_inline size_type count_trailing_one_bits() const noexcept;
	constexpr simd_stl_always_inline size_type count_leading_one_bits() const noexcept;
	constexpr simd_stl_always_inline void clear_left_most_set_bit() noexcept;

	constexpr simd_stl_always_inline mask_type unwrap() const noexcept;

	constexpr simd_stl_always_inline bool operator==(const simd_mask& __other) const noexcept;
	constexpr simd_stl_always_inline bool operator!=(const simd_mask& __other) const noexcept;
	constexpr simd_stl_always_inline bool operator<(const simd_mask& __other) const noexcept;
	constexpr simd_stl_always_inline bool operator<=(const simd_mask& __other) const noexcept;
	constexpr simd_stl_always_inline bool operator>(const simd_mask& __other) const noexcept;
	constexpr simd_stl_always_inline bool operator>=(const simd_mask& __other) const noexcept;

	constexpr simd_stl_always_inline simd_mask operator&(const simd_mask& __other) const noexcept;
	constexpr simd_stl_always_inline simd_mask operator|(const simd_mask& __other) const noexcept;
	constexpr simd_stl_always_inline simd_mask operator^(const simd_mask& __other) const noexcept;

	constexpr simd_stl_always_inline bool operator[](int32 __index) const noexcept;

	constexpr simd_stl_always_inline simd_mask& operator&=(const simd_mask& __other) noexcept;
	constexpr simd_stl_always_inline simd_mask& operator|=(const simd_mask& __other) noexcept;
	constexpr simd_stl_always_inline simd_mask& operator=(mask_type __other) noexcept;
	constexpr simd_stl_always_inline simd_mask& operator^=(const simd_mask& __other) noexcept;

	constexpr simd_stl_always_inline simd_mask operator~() const noexcept;

	constexpr simd_stl_always_inline explicit operator bool() const noexcept;
private:
	mask_type _mask = 0;
};

__SIMD_STL_NUMERIC_NAMESPACE_END

#include <src/simd_stl/numeric/BasicSimdMask.inl>