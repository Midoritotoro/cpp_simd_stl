#pragma once

#include <src/simd_stl/algorithm/AlgorithmDebug.h>
#include <src/simd_stl/type_traits/SimdAlgorithmSafety.h>

#include <src/simd_stl/algorithm/vectorized/find/EqualVectorized.h>
#include <src/simd_stl/algorithm/MsvcIteratorUnwrap.h>

#include <src/simd_stl/type_traits/CanMemcmpElements.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <
	class _FirstUnwrappedIterator_,
	class _SecondUnwrappedIterator_,
	class _Predicate_>
_Simd_nodiscard_inline_constexpr bool _EqualUnchecked(
	_FirstUnwrappedIterator_	_First1Unwrapped,
	_FirstUnwrappedIterator_	_Last1Unwrapped,
	_SecondUnwrappedIterator_	_First2Unwrapped,
	_Predicate_					_Predicate) noexcept(
		type_traits::is_nothrow_invocable_v<
			_Predicate_, 
			type_traits::IteratorValueType<_FirstUnwrappedIterator_>,
			type_traits::IteratorValueType<_SecondUnwrappedIterator_>>)
{
	if constexpr (
		type_traits::is_iterator_random_ranges_v<_FirstUnwrappedIterator_> &&
		type_traits::is_iterator_random_ranges_v<_SecondUnwrappedIterator_>)
	{
		const auto _Length = IteratorsDifference(_First1Unwrapped, _Last1Unwrapped);

		if constexpr (type_traits::is_vectorized_search_algorithm_safe_v<
			_FirstUnwrappedIterator_, _SecondUnwrappedIterator_, _Predicate_>)
		{
#if simd_stl_has_cxx20
			if (type_traits::is_constant_evaluated() == false)
#endif // simd_stl_has_cxx20
			{
				return _EqualVectorized<type_traits::IteratorValueType<_FirstUnwrappedIterator_>>(
					std::to_address(_First1Unwrapped), std::to_address(_First2Unwrapped), _Length);
			}
		}
		else {
			for (sizetype _Current = 0; _Current < _Length; ++_Current)
				if (_Predicate(*_First1Unwrapped++, *_First2Unwrapped++) == false)
					return false;

			return true;
		}
	}

	for (; _First1Unwrapped != _Last1Unwrapped; ++_First1Unwrapped, ++_First2Unwrapped)
		if (_Predicate(*_First1Unwrapped, *_First2Unwrapped) == false)
			return false;
	
	return true;
}

template <
	class _FirstUnwrappedIterator_,
	class _SecondUnwrappedIterator_,
	class _Predicate_>
_Simd_nodiscard_inline_constexpr bool _EqualUnchecked(
	_FirstUnwrappedIterator_	_First1Unwrapped,
	_FirstUnwrappedIterator_	_Last1Unwrapped,
	_SecondUnwrappedIterator_	_First2Unwrapped,
	_SecondUnwrappedIterator_	_Last2Unwrapped,
	_Predicate_					_Predicate) noexcept(
		type_traits::is_nothrow_invocable_v<
			_Predicate_,
			type_traits::IteratorValueType<_FirstUnwrappedIterator_>,
			type_traits::IteratorValueType<_FirstUnwrappedIterator_>>)
{
	if constexpr (type_traits::is_iterator_random_ranges_v<_FirstUnwrappedIterator_> &&
		type_traits::is_iterator_random_ranges_v<_SecondUnwrappedIterator_>)
	{
		const auto _Length = IteratorsDifference(_First1Unwrapped, _Last1Unwrapped);

		if (_Length != IteratorsDifference(_First2Unwrapped, _Last2Unwrapped))
			return false;

		if constexpr (type_traits::is_vectorized_search_algorithm_safe_v<
			_FirstUnwrappedIterator_, _SecondUnwrappedIterator_, _Predicate_>)
		{
#if simd_stl_has_cxx20
			if (type_traits::is_constant_evaluated() == false)
#endif // simd_stl_has_cxx20
			{
				return _EqualVectorized<type_traits::IteratorValueType<_FirstUnwrappedIterator_>>(
					std::to_address(_First1Unwrapped), std::to_address(_First2Unwrapped), _Length);
			}
		}
		else {
			for (sizetype _Current = 0; _Current < _Length; ++_Current)
				if (_Predicate(*_First1Unwrapped++, *_First2Unwrapped++) == false)
					return false;

			return true;
		}
	}

	for (; _First1Unwrapped != _Last1Unwrapped; ++_First1Unwrapped, ++_First2Unwrapped)
		if (_Predicate(*_First1Unwrapped, *_First2Unwrapped) == false)
			return false;
	
	return true;
}


__SIMD_STL_ALGORITHM_NAMESPACE_END
