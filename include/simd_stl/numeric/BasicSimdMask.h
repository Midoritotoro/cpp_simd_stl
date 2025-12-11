#pragma once 

#include <simd_stl/numeric/BasicSimdMaskImplementation.h>
#include <simd_stl/math/BitMath.h>


__SIMD_STL_NUMERIC_NAMESPACE_BEGIN

template <
	arch::CpuFeature	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_ = numeric::_DefaultRegisterPolicy<_SimdGeneration_>>
class basic_simd_mask {
	static_assert(type_traits::__is_generation_supported_v<_SimdGeneration_>);
	static_assert(type_traits::__is_vector_type_supported_v<_Element_>);

	using implementation = BasicSimdMaskImplementation<_SimdGeneration_, _Element_, _RegisterPolicy_>;
public:
	using mask_type = typename implementation::mask_type;
	using size_type = typename implementation::size_type;

	basic_simd_mask() noexcept;
	basic_simd_mask(const mask_type mask) noexcept;

	constexpr simd_stl_always_inline bool allOf() const noexcept;
	constexpr simd_stl_always_inline bool anyOf() const noexcept;
	constexpr simd_stl_always_inline bool noneOf() const noexcept;

	constexpr simd_stl_always_inline size_type countSet() const noexcept;
	constexpr simd_stl_always_inline size_type countTrailingZeroBits() const noexcept;
	constexpr simd_stl_always_inline size_type countLeadingZeroBits() const noexcept;
	constexpr simd_stl_always_inline size_type countTrailingOneBits() const noexcept;
	constexpr simd_stl_always_inline size_type countLeadingOneBits() const noexcept;
	constexpr simd_stl_always_inline void clearLeftMostSetBit() noexcept;

	constexpr simd_stl_always_inline mask_type unwrap() const noexcept;

	constexpr simd_stl_always_inline bool operator==(const basic_simd_mask& other) const noexcept;
	constexpr simd_stl_always_inline bool operator!=(const basic_simd_mask& other) const noexcept;
	constexpr simd_stl_always_inline bool operator<(const basic_simd_mask& other) const noexcept;
	constexpr simd_stl_always_inline bool operator<=(const basic_simd_mask& other) const noexcept;
	constexpr simd_stl_always_inline bool operator>(const basic_simd_mask& other) const noexcept;
	constexpr simd_stl_always_inline bool operator>=(const basic_simd_mask& other) const noexcept;

	constexpr simd_stl_always_inline basic_simd_mask operator&(const basic_simd_mask& other) const noexcept;
	constexpr simd_stl_always_inline basic_simd_mask operator|(const basic_simd_mask& other) const noexcept;
	constexpr simd_stl_always_inline basic_simd_mask operator^(const basic_simd_mask& other) const noexcept;

	constexpr simd_stl_always_inline bool operator[](int32 _Index) const noexcept;

	constexpr simd_stl_always_inline basic_simd_mask& operator&=(const basic_simd_mask& other) noexcept;
	constexpr simd_stl_always_inline basic_simd_mask& operator|=(const basic_simd_mask& other) noexcept;
	constexpr simd_stl_always_inline basic_simd_mask& operator=(mask_type other) noexcept;
	constexpr simd_stl_always_inline basic_simd_mask& operator^=(const basic_simd_mask& other) noexcept;

	constexpr simd_stl_always_inline basic_simd_mask operator~() const noexcept;

	constexpr simd_stl_always_inline explicit operator bool() const noexcept;
private:
	mask_type _mask = 0;
};

__SIMD_STL_NUMERIC_NAMESPACE_END

#include <src/simd_stl/numeric/BasicSimdMask.inl>