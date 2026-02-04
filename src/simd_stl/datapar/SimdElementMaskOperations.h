#pragma once 

#include <src/simd_stl/datapar/SimdMaskCommonOperations.h>
#include <simd_stl/math/BitMath.h>

__SIMD_STL_DATAPAR_NAMESPACE_BEGIN

template <class _Derived_>
struct __simd_element_mask_operations:
    public __simd_mask_common_operations<_Derived_> 
{
    using __self = std::remove_cvref_t<_Derived_>;
	using __base = __simd_mask_common_operations<_Derived_>;

	constexpr simd_stl_always_inline uint8 count_set() const noexcept;
	constexpr simd_stl_always_inline uint8 count_trailing_zero_bits() const noexcept;
	constexpr simd_stl_always_inline uint8 count_leading_zero_bits() const noexcept;
};

__SIMD_STL_DATAPAR_NAMESPACE_END

#include <src/simd_stl/datapar/SimdElementMaskOperations.inl>
