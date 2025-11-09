#pragma once 

#include <src/simd_stl/algorithm/AlgorithmDebug.h>
#include <src/simd_stl/type_traits/SimdAlgorithmSafety.h>

#include <simd_stl/compatibility/Nodiscard.h>
#include <simd_stl/compatibility/Inline.h>

#include <src/simd_stl/algorithm/vectorized/EqualVectorized.h>
#include <src/simd_stl/algorithm/MsvcIteratorUnwrap.h>

#include <src/simd_stl/type_traits/CanMemcmpElements.h>
#include <src/simd_stl/type_traits/FunctionPass.h>

#include <cmath>

__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <
	class _FirstIterator_,
	class _SecondIterator_,
	class _Predicate_>
simd_stl_nodiscard simd_stl_always_inline simd_stl_constexpr_cxx20 bool equal(
	_FirstIterator_			first1,
	const _FirstIterator_	last1,
	_SecondIterator_		first2,
	_Predicate_				predicate) noexcept(
		type_traits::is_nothrow_invocable_v<
			_Predicate_,
			type_traits::IteratorValueType<_FirstIterator_>,
			type_traits::IteratorValueType<_SecondIterator_>
		>
	)
{
	__verifyRange(first1, last1);

	using _FirstIteratorUnwrappedType_	= unwrapped_iterator_type<_FirstIterator_>;
	using _SecondIteratorUnwrappedType_ = unwrapped_iterator_type<_SecondIterator_>;

	auto first1Unwrapped		= _UnwrapIterator(first1);
	const auto last1Unwrapped	= _UnwrapIterator(last1);

	auto first2Unwrapped		= _UnwrapIterator(first2);

	if constexpr (
		type_traits::is_iterator_random_ranges_v<_FirstIteratorUnwrappedType_>
		&& type_traits::is_iterator_random_ranges_v<_SecondIteratorUnwrappedType_>)
	{
		const auto length = IteratorsDifference(first1Unwrapped, last1Unwrapped);

		if constexpr (type_traits::is_vectorized_search_algorithm_safe_v<
			_FirstIteratorUnwrappedType_, _SecondIteratorUnwrappedType_, _Predicate_>)
		{
#if simd_stl_has_cxx20
			if (type_traits::is_constant_evaluated() == false)
#endif
			{
				using _ValueType_ = type_traits::IteratorValueType<_FirstIterator_>;

				return EqualVectorized<_ValueType_>(
					std::to_address(first1Unwrapped), std::to_address(first2Unwrapped), length);
			}
		}
		else {
			for (sizetype current = 0; current < length; ++current)
				if (predicate(*first1Unwrapped++, *first2Unwrapped++) == false)
					return false;

			return true;
		}
	}

	for (; first1Unwrapped != last1Unwrapped; ++first1Unwrapped, ++first2Unwrapped)
		if (predicate(*first1Unwrapped, *first2Unwrapped) == false)
			return false;
	
	return true;
}

template <
	class _FirstIterator_,
	class _SecondIterator_,
	class _Predicate_>
simd_stl_nodiscard simd_stl_always_inline simd_stl_constexpr_cxx20 bool equal(
	_FirstIterator_			first1,
	const _FirstIterator_	last1,
	_SecondIterator_		first2,
	const _SecondIterator_	last2,
	_Predicate_				predicate) noexcept(
		type_traits::is_nothrow_invocable_v<
			_Predicate_,
			type_traits::IteratorValueType<_FirstIterator_>,
			type_traits::IteratorValueType<_SecondIterator_>
		>
	)
{
	__verifyRange(first1, last1);
	__verifyRange(first2, last2);

	using _FirstIteratorUnwrappedType_	= unwrapped_iterator_type<_FirstIterator_>;
	using _SecondIteratorUnwrappedType_ = unwrapped_iterator_type<_SecondIterator_>;

	auto first1Unwrapped		= _UnwrapIterator(first1);
	const auto last1Unwrapped	= _UnwrapIterator(last1);

	auto first2Unwrapped		= _UnwrapIterator(first2);
	const auto last2Unwrapped	= _UnwrapIterator(last2);

	if constexpr (type_traits::is_iterator_random_ranges_v<_FirstIteratorUnwrappedType_>
		&& type_traits::is_iterator_random_ranges_v<_SecondIteratorUnwrappedType_>)
	{
		const auto length = IteratorsDifference(first1Unwrapped, last1Unwrapped);

		if (length != IteratorsDifference(first2Unwrapped, last2Unwrapped))
			return false;

		if constexpr (type_traits::is_vectorized_search_algorithm_safe_v<
			_FirstIteratorUnwrappedType_, _SecondIteratorUnwrappedType_, _Predicate_>)
		{
#if simd_stl_has_cxx20
			if (type_traits::is_constant_evaluated() == false)
#endif
			{
				using _ValueType_ = type_traits::IteratorValueType<_FirstIterator_>;

				return EqualVectorized<_ValueType_>(
					std::to_address(first1Unwrapped), std::to_address(first2Unwrapped), length);
			}
		}
		else {
			for (sizetype current = 0; current < length; ++current)
				if (predicate(*first1Unwrapped++, *first2Unwrapped++) == false)
					return false;

			return true;
		}
	}

	for (; first1Unwrapped != last1Unwrapped; ++first1Unwrapped, ++first2Unwrapped)
		if (predicate(*first1Unwrapped, *first2Unwrapped) == false)
			return false;
	
	return true;
}

template <
	class _FirstIterator_,
	class _SecondIterator_>
simd_stl_nodiscard simd_stl_always_inline simd_stl_constexpr_cxx20 bool equal(
	_FirstIterator_		first1,
	_FirstIterator_		last1,
	_SecondIterator_	first2) noexcept(
		type_traits::is_nothrow_invocable_v<
			type_traits::equal_to<>,
			type_traits::IteratorValueType<_FirstIterator_>,
			type_traits::IteratorValueType<_SecondIterator_>
		>
	)
{
	return simd_stl::algorithm::equal(first1, last1, first2, type_traits::equal_to<>{});
}

template <
	class _FirstIterator_,
	class _SecondIterator_>
simd_stl_nodiscard simd_stl_always_inline simd_stl_constexpr_cxx20 bool equal(
	_FirstIterator_		first1,
	_FirstIterator_		last1,
	_SecondIterator_	first2,
	_SecondIterator_	last2) noexcept(
		type_traits::is_nothrow_invocable_v<
			type_traits::equal_to<>,
			type_traits::IteratorValueType<_FirstIterator_>,
			type_traits::IteratorValueType<_SecondIterator_>
		>
	)
{
	return simd_stl::algorithm::equal(first1, last1, first2, last2, type_traits::equal_to<>{});
}

template <
	class _ExecutionPolicy_,
	class _FirstIterator_,
	class _SecondIterator_,
	class _Predicate_>
simd_stl_nodiscard simd_stl_always_inline simd_stl_constexpr_cxx20 bool equal(
	_ExecutionPolicy_&&,
	_FirstIterator_		first1,
	_FirstIterator_		last1,
	_SecondIterator_	first2,
	_Predicate_			predicate) noexcept(
		type_traits::is_nothrow_invocable_v<
			_Predicate_,
			type_traits::IteratorValueType<_FirstIterator_>,
			type_traits::IteratorValueType<_SecondIterator_>
		>
	)
{
	return simd_stl::algorithm::equal(first1, last1, first2, type_traits::passFunction(predicate));
}


template <
	class _ExecutionPolicy_,
	class _FirstIterator_,
	class _SecondIterator_,
	class _Predicate_>
simd_stl_nodiscard simd_stl_always_inline simd_stl_constexpr_cxx20 bool equal(
	_ExecutionPolicy_&&,
	_FirstIterator_		first1,
	_FirstIterator_		last1,
	_SecondIterator_	first2,
	_SecondIterator_	last2,
	_Predicate_			predicate) noexcept(
		type_traits::is_nothrow_invocable_v<
			_Predicate_,
			type_traits::IteratorValueType<_FirstIterator_>,
			type_traits::IteratorValueType<_SecondIterator_>
		>
	)
{
	return simd_stl::algorithm::equal(first1, last1, first2, last2, type_traits::passFunction(predicate));
}

template <
	class _ExecutionPolicy_,
	class _FirstIterator_,
	class _SecondIterator_>
simd_stl_nodiscard simd_stl_always_inline simd_stl_constexpr_cxx20 bool equal(
	_ExecutionPolicy_&&,
	_FirstIterator_		first1,
	_FirstIterator_		last1,
	_SecondIterator_	first2) noexcept(
		type_traits::is_nothrow_invocable_v<
			type_traits::equal_to<>,
			type_traits::IteratorValueType<_FirstIterator_>,
			type_traits::IteratorValueType<_SecondIterator_>
		>
	)
{
	return simd_stl::algorithm::equal(first1, last1, first2);
}


template <
	class _ExecutionPolicy_,
	class _FirstIterator_,
	class _SecondIterator_>
simd_stl_nodiscard simd_stl_always_inline simd_stl_constexpr_cxx20 bool equal(
	_ExecutionPolicy_&&,
	_FirstIterator_		first1,
	_FirstIterator_		last1,
	_SecondIterator_	first2,
	_SecondIterator_	last2) noexcept(
		type_traits::is_nothrow_invocable_v<
			type_traits::equal_to<>,
			type_traits::IteratorValueType<_FirstIterator_>,
			type_traits::IteratorValueType<_SecondIterator_>
		>
	)
{
	return simd_stl::algorithm::equal(first1, last1, first2, last2);
}

__SIMD_STL_ALGORITHM_NAMESPACE_END
