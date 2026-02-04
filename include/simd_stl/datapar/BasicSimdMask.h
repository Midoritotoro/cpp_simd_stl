#pragma once 

#include <src/simd_stl/datapar/SimdElementMaskOperations.h>
#include <src/simd_stl/type_traits/TypeTraits.h>


__SIMD_STL_DATAPAR_NAMESPACE_BEGIN

template <
	arch::CpuFeature	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_>
class simd_mask: 
	public __simd_element_mask_operations<simd_mask<_SimdGeneration_, _Element_, _RegisterPolicy_>>
{
	static_assert(type_traits::__is_generation_supported_v<_SimdGeneration_>);
	static_assert(type_traits::__is_vector_type_supported_v<_Element_>);
public:
	static constexpr auto __generation = _SimdGeneration_;

	using element_type	= _Element_;
	using policy_type	= _RegisterPolicy_;
	using mask_type		= type_traits::__deduce_simd_mask_type<__generation, element_type, policy_type>;

	static constexpr uint8 __used_bits = policy_type::__width / sizeof(_Element_);

	simd_mask() noexcept;

	simd_mask(const mask_type __mask) noexcept;
	simd_mask(const simd_mask& __mask) noexcept;
	
	template <class _VectorMask_, std::enable_if_t<__is_valid_basic_simd_v<_VectorMask_> || __is_intrin_type_v<_VectorMask_>, int> = 0>
	simd_mask(const _VectorMask_& __vector_mask) noexcept;

	constexpr simd_stl_always_inline bool operator[](int32 __index) const noexcept;

	constexpr simd_stl_always_inline simd_mask operator++(int) noexcept;
	constexpr simd_stl_always_inline simd_mask& operator++() noexcept;

	constexpr simd_stl_always_inline simd_mask operator--(int) noexcept;
	constexpr simd_stl_always_inline simd_mask& operator--() noexcept;

	constexpr simd_stl_always_inline simd_mask& operator&=(const simd_mask& __other) noexcept;
	constexpr simd_stl_always_inline simd_mask& operator|=(const simd_mask& __other) noexcept;
	constexpr simd_stl_always_inline simd_mask& operator^=(const simd_mask& __other) noexcept;

	constexpr simd_stl_always_inline simd_mask& operator>>=(const uint8 __shift) const noexcept;
	constexpr simd_stl_always_inline simd_mask& operator<<=(const uint8 __shift) const noexcept;

	template <uint8 _Shift_>
	constexpr simd_stl_always_inline simd_mask& operator>>=(const std::integral_constant<uint8, _Shift_> __shift) const noexcept;
	
	template <uint8 _Shift_>
	constexpr simd_stl_always_inline simd_mask& operator<<=(const std::integral_constant<uint8, _Shift_> __shift) const noexcept;

	static constexpr simd_stl_always_inline uint8 bit_width() noexcept;
	static constexpr simd_stl_always_inline arch::CpuFeature generation() noexcept;
	constexpr simd_stl_always_inline mask_type unwrap() const noexcept;

private:
	mask_type _mask = 0;
};

__SIMD_STL_DATAPAR_NAMESPACE_END

#include <src/simd_stl/datapar/BasicSimdMask.inl>