#pragma once 

#include <src/simd_stl/datapar/SimdIndexMaskOperations.h>
#include <src/simd_stl/datapar/MaskTypeSelector.h>

__SIMD_STL_DATAPAR_NAMESPACE_BEGIN

template <
	arch::ISA	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_>
class simd_index_mask: public __simd_index_mask_operations<simd_index_mask<_SimdGeneration_, _Element_, _RegisterPolicy_>> {
public:
	static constexpr auto __generation = _SimdGeneration_;
	static constexpr auto __is_k_register = __has_avx512f_support_v<__generation>;

	using element_type	= _Element_;
	using policy_type	= _RegisterPolicy_;

	static constexpr bool __is_native_compare_returns_number = std::is_integral_v<__simd_native_compare_return_type<
		simd<__generation, element_type, policy_type>, element_type, __simd_comparison::equal>>;

	static constexpr uint8 __divisor = __simd_index_mask_divisor<__generation, policy_type, element_type>;
	static constexpr uint8 __used_bits = policy_type::__width / sizeof(element_type) * __divisor;

	using mask_type = __mmask_for_size_t<((__used_bits <= 8) ? 1 : (__used_bits / 8))>;

	simd_index_mask() noexcept;
	simd_index_mask(const mask_type __mask) noexcept;
	
	template <class _VectorMask_, std::enable_if_t<__is_valid_basic_simd_v<_VectorMask_> || __is_intrin_type_v<_VectorMask_>, int> = 0>
	simd_index_mask(const _VectorMask_& __vector_mask) noexcept;


	constexpr simd_stl_always_inline simd_index_mask& operator&=(const simd_index_mask& __other) noexcept;
	constexpr simd_stl_always_inline simd_index_mask& operator|=(const simd_index_mask& __other) noexcept;
	constexpr simd_stl_always_inline simd_index_mask& operator^=(const simd_index_mask& __other) noexcept;

	constexpr simd_stl_always_inline simd_index_mask& operator>>=(const uint8 __shift) noexcept;
	constexpr simd_stl_always_inline simd_index_mask& operator<<=(const uint8 __shift) noexcept;

	template <sizetype _Shift_>
	constexpr simd_stl_always_inline simd_index_mask& operator>>=(const std::integral_constant<uint8, _Shift_> __shift) noexcept;

	template <sizetype _Shift_>
	constexpr simd_stl_always_inline simd_index_mask& operator<<=(const std::integral_constant<uint8, _Shift_> __shift) noexcept;

	constexpr simd_stl_always_inline mask_type unwrap() const noexcept;
	static constexpr simd_stl_always_inline arch::ISA generation() noexcept;
	static constexpr simd_stl_always_inline uint8 bit_width() noexcept;
	static constexpr simd_stl_always_inline uint8 divisor() noexcept;
private:
	mask_type _mask = 0;
};

__SIMD_STL_DATAPAR_NAMESPACE_END

#include <src/simd_stl/datapar/SimdIndexMask.inl>