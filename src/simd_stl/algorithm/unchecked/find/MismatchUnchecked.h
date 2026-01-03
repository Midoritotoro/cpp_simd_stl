#pragma once 

#include <src/simd_stl/algorithm/AlgorithmDebug.h>
#include <src/simd_stl/type_traits/SimdAlgorithmSafety.h>

#include <src/simd_stl/algorithm/vectorized/find/MismatchVectorized.h>
#include <src/simd_stl/algorithm/MsvcIteratorUnwrap.h>

#include <src/simd_stl/type_traits/CanMemcmpElements.h>
#include <src/simd_stl/type_traits/FunctionPass.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <
	class _FirstUnwrappedIterator_,
	class _SecondUnwrappedIterator_,
	class _Predicate_>
__simd_nodiscard_inline_constexpr std::pair<_FirstUnwrappedIterator_, _SecondUnwrappedIterator_> _MismatchUnchecked(
	_FirstUnwrappedIterator_		_First1Unwrapped,
	_FirstUnwrappedIterator_		_Last1Unwrapped,
	_SecondUnwrappedIterator_		_First2Unwrapped,
	_Predicate_						_Predicate) noexcept(
		type_traits::is_nothrow_invocable_v<
			_Predicate_,
			type_traits::IteratorValueType<_FirstUnwrappedIterator_>,
			type_traits::IteratorValueType<_SecondUnwrappedIterator_>>)
{
	if constexpr (type_traits::is_vectorized_search_algorithm_safe_v<
		_FirstUnwrappedIterator_, _SecondUnwrappedIterator_, _Predicate_>)
	{
#if simd_stl_has_cxx20
		if (type_traits::is_constant_evaluated() == false)
#endif // simd_stl_has_cxx20
		{
			const auto _Position = _MismatchVectorized<type_traits::IteratorValueType<_FirstUnwrappedIterator_>>(
				std::to_address(_First1Unwrapped), std::to_address(_First2Unwrapped), 
				__iterators_difference(_First1Unwrapped, _Last1Unwrapped));

			_First1Unwrapped += _Position;
			_First2Unwrapped += _Position;

			return { _First1Unwrapped, _First2Unwrapped };
		}
	}

	while (_First1Unwrapped != _Last1Unwrapped && _Predicate(*_First1Unwrapped, *_First2Unwrapped)) {
		++_First1Unwrapped;
		++_First2Unwrapped;
	}

	return { _First1Unwrapped, _First2Unwrapped };
}

template <
	class _FirstUnwrappedIterator_,
	class _SecondUnwrappedIterator_,
	class _Predicate_>
__simd_nodiscard_inline_constexpr std::pair<_FirstUnwrappedIterator_, _SecondUnwrappedIterator_> _MismatchUnchecked(
	_FirstUnwrappedIterator_	_First1Unwrapped,
	_FirstUnwrappedIterator_	_Last1Unwrapped,
	_SecondUnwrappedIterator_	_First2Unwrapped,
	_SecondUnwrappedIterator_	_Last2Unwrapped,
	_Predicate_					_Predicate) noexcept(
		type_traits::is_nothrow_invocable_v<
			_Predicate_,
			type_traits::IteratorValueType<_FirstUnwrappedIterator_>,
			type_traits::IteratorValueType<_SecondUnwrappedIterator_>>)
{
	if constexpr (type_traits::is_vectorized_search_algorithm_safe_v<
		_FirstUnwrappedIterator_, _SecondUnwrappedIterator_, _Predicate_>)
	{
#if simd_stl_has_cxx20
		if (type_traits::is_constant_evaluated() == false)
#endif // simd_stl_has_cxx20
		{
			const auto _Position = _MismatchVectorized<type_traits::IteratorValueType<_FirstUnwrappedIterator_>>(
				std::to_address(_First1Unwrapped), std::to_address(_First2Unwrapped), (std::min)(
					__iterators_difference(_First1Unwrapped, _Last1Unwrapped), __iterators_difference(_First2Unwrapped, _Last2Unwrapped)));
				
			_First1Unwrapped += _Position;
			_First2Unwrapped += _Position;

			return { _First1Unwrapped, _First2Unwrapped };
		}
	}

	while (_First1Unwrapped != _Last1Unwrapped && _Predicate(*_First1Unwrapped, *_First2Unwrapped)) {
		++_First1Unwrapped;
		++_First2Unwrapped;
	}

	return { _First1Unwrapped, _First2Unwrapped };
}

__SIMD_STL_ALGORITHM_NAMESPACE_END
