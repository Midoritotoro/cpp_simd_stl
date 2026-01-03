#pragma once 

#include <src/simd_stl/type_traits/IteratorCategory.h>
#include <src/simd_stl/type_traits/OperatorWrappers.h>

__SIMD_STL_TYPE_TRAITS_NAMESPACE_BEGIN

template <
	class __firstElement_,
	class _SecondElement_,
	bool = sizeof(__firstElement_) == sizeof(_SecondElement_) 
		&& std::is_integral_v<__firstElement_>
		&& std::is_integral_v<_SecondElement_>>
constexpr bool can_memcmp_elements_v =
		std::is_same_v<__firstElement_, bool> 
	||	std::is_same_v<_SecondElement_, bool> 
	||	static_cast<__firstElement_>(-1) == static_cast<_SecondElement_>(-1);

#ifdef __cpp_lib_byte
template <>
inline constexpr bool can_memcmp_elements_v<std::byte, std::byte, false> = true;
#endif // defined(__cpp_lib_byte)

template <
	class __firstType_,
	class _SecondType_>
constexpr bool can_memcmp_elements_v<__firstType_*, _SecondType_*, false> = 
	is_pointer_address_comparable_v<__firstType_, _SecondType_>;


template <
	class __firstElement_,
	class _SecondElement_>
constexpr bool can_memcmp_elements_v<__firstElement_, _SecondElement_, false> = false;

template <
	class __firstElement_,
	class _SecondElement_,
	class _Function_>
constexpr bool can_memcmp_elements_with_pred_v = false;

template <
	class __firstElement_,
	class _SecondElement_, 
	class _ThirdElement_>
constexpr bool can_memcmp_elements_with_pred_v<__firstElement_, _SecondElement_, std::equal_to<_ThirdElement_>> =
	IteratorCopyCategory<__firstElement_*, _ThirdElement_*>::BitcopyConstructible && 
	IteratorCopyCategory<_SecondElement_*, _ThirdElement_*>::BitcopyConstructible && 
	can_memcmp_elements_v<std::remove_cv_t<_ThirdElement_>, std::remove_cv_t<_ThirdElement_>
>;


template <
	class __firstElement_, 
	class _SecondElement_>
constexpr bool can_memcmp_elements_with_pred_v<__firstElement_, _SecondElement_, type_traits::equal_to<>> =
	can_memcmp_elements_v<__firstElement_, _SecondElement_>;

#if simd_stl_has_cxx20
template <
	class __firstElement_,
	class _SecondElement_>
constexpr bool can_memcmp_elements_with_pred_v<__firstElement_, _SecondElement_, std::ranges::equal_to> =
	can_memcmp_elements_v<__firstElement_, _SecondElement_>;
#endif // simd_stl_has_cxx20

template <
	class __firstIterator_,
	class _SecondIterator_,
	class _Function_>
constexpr bool equal_memcmp_is_safe_helper = 
	is_iterator_contiguous_v<__firstIterator_> && is_iterator_contiguous_v<_SecondIterator_> 
	&& !is_iterator_volatile_v<__firstIterator_> && !is_iterator_volatile_v<_SecondIterator_>
	&& can_memcmp_elements_with_pred_v<
		IteratorValueType<__firstIterator_>,
		IteratorValueType<_SecondIterator_>, _Function_>;

template <
	class __firstIterator_,
	class _SecondIterator_, 
	class _Function_>
constexpr bool equal_memcmp_is_safe_v =
	equal_memcmp_is_safe_helper<
		std::remove_const_t<__firstIterator_>,
		std::remove_const_t<_SecondIterator_>,
		std::remove_const_t<_Function_>
	>;


template <
	class __firstIterator_,
	class _SecondIterator_,
	class _Function_>
constexpr bool is_vectorized_search_algorithm_safe_v = equal_memcmp_is_safe_v<__firstIterator_, _SecondIterator_, _Function_>;


__SIMD_STL_TYPE_TRAITS_NAMESPACE_END
