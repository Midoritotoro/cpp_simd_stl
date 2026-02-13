#pragma once 

#include <src/simd_stl/algorithm/AlgorithmDebug.h>

#include <src/simd_stl/algorithm/vectorized/find/SearchVectorized.h>
#include <src/simd_stl/type_traits/CanMemcmpElements.h>

#include <src/simd_stl/algorithm/MsvcIteratorUnwrap.h>
#include <src/simd_stl/type_traits/OperatorWrappers.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <
	class _FirstUnwrappedForwardIterator_,
	class _SecondUnwrappedForwardIterator_,
	class _Predicate_> 
__simd_nodiscard_inline_constexpr _FirstUnwrappedForwardIterator_ __search_unchecked_forward(
	_FirstUnwrappedForwardIterator_		__first1_unwrapped,
	_FirstUnwrappedForwardIterator_		__last1_unwrapped,
	_SecondUnwrappedForwardIterator_	__first2_unwrapped,
	_SecondUnwrappedForwardIterator_	__last2_unwrapped,
	_Predicate_							__predicate) noexcept(
		type_traits::is_nothrow_invocable_v<
			_Predicate_,
			type_traits::iterator_value_type<_FirstUnwrappedForwardIterator_>,
			type_traits::iterator_value_type<_SecondUnwrappedForwardIterator_>>)
{
	const auto __last_possible = __last1_unwrapped - __iterators_difference(__first2_unwrapped, __last2_unwrapped);

	for (;; ++__first1_unwrapped) {
		auto __current_first = __first1_unwrapped;
		
		for (auto __current_second = __first2_unwrapped; ++__current_first; (void) ++__current_second) {
			if (__current_second == __last2_unwrapped)
				return __first1_unwrapped;

			if (__current_first == __last1_unwrapped)
				return __last1_unwrapped;

			if (__predicate(*__current_first, *__current_second) == false)
				break;
		}
	}

	return __last1_unwrapped;
}

template <
	class _FirstUnwrappedForwardIterator_,
	class _SecondUnwrappedForwardIterator_,
	class _Predicate_> 
__simd_nodiscard_inline_constexpr _FirstUnwrappedForwardIterator_ __search_unchecked(
	_FirstUnwrappedForwardIterator_		__first1_unwrapped,
	_FirstUnwrappedForwardIterator_		__last1_unwrapped,
	_SecondUnwrappedForwardIterator_	__first2_unwrapped,
	_SecondUnwrappedForwardIterator_	__last2_unwrapped,
	_Predicate_							__predicate) noexcept(
		type_traits::is_nothrow_invocable_v<
			_Predicate_,
			type_traits::iterator_value_type<_FirstUnwrappedForwardIterator_>,
			type_traits::iterator_value_type<_SecondUnwrappedForwardIterator_>>)
{
	using _ValueType = type_traits::iterator_value_type<_FirstUnwrappedForwardIterator_>;

	constexpr auto __is_random = type_traits::is_iterator_random_ranges_v<_FirstUnwrappedForwardIterator_> &&
		type_traits::is_iterator_random_ranges_v<_SecondUnwrappedForwardIterator_>;
		
	if constexpr (__is_random) {
		const auto __first1_address	= std::to_address(__first1_unwrapped);
		const auto __first2_address	= std::to_address(__first2_unwrapped);

		auto __first_range_length			= __iterators_difference(__first1_unwrapped, __last1_unwrapped);
		const auto __second_range_length	= __iterators_difference(__first2_unwrapped, __last2_unwrapped);

		if (__first_range_length < __second_range_length)
			return __last1_unwrapped;
	
		if constexpr (type_traits::__is_vectorized_search_algorithm_safe_v<
			_FirstUnwrappedForwardIterator_, _SecondUnwrappedForwardIterator_, _Predicate_>)
		{
#if simd_stl_has_cxx20
			if (type_traits::is_constant_evaluated() == false)
#endif // simd_stl_has_cxx20
			{
				const auto __position = __search_vectorized(
					__first1_address, __first_range_length, __first2_address, __second_range_length);

				return (__position == nullptr)
					? __last1_unwrapped
					: __first1_unwrapped + (reinterpret_cast<const _ValueType*>(__position) - __first1_address);
			}
		}

		const auto __position = _Search<arch::ISA::None>()(
			__first1_address, __first_range_length, __first2_address,
			__second_range_length, type_traits::__pass_function(_Predicate));

		return (__position == nullptr)
			? __last1_unwrapped
			: __first1_unwrapped + (reinterpret_cast<const _ValueType*>(__position) - __first1_address);
	}
	
	return __search_unchecked_forward(__first1_unwrapped, __last1_unwrapped,
		__first2_unwrapped, __last2_unwrapped, type_traits::__pass_function(__predicate));
}

__SIMD_STL_ALGORITHM_NAMESPACE_END
