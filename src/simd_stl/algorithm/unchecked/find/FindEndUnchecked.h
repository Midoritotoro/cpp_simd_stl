#pragma once 

#include <src/simd_stl/algorithm/AlgorithmDebug.h>

#include <src/simd_stl/type_traits/CanMemcmpElements.h>

#include <src/simd_stl/algorithm/MsvcIteratorUnwrap.h>
#include <src/simd_stl/type_traits/OperatorWrappers.h>

#include <src/simd_stl/algorithm/vectorized/find/FindEndVectorized.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <
	class _FirstUnwrappedForwardIterator_,
	class _SecondUnwrappedForwardIterator_,
	class _Predicate_>
_Simd_nodiscard_inline_constexpr _FirstUnwrappedForwardIterator_ _FindEndUnchecked(
	_FirstUnwrappedForwardIterator_		_First1Unwrapped,
	_FirstUnwrappedForwardIterator_		_Last1Unwrapped,
	_SecondUnwrappedForwardIterator_	_First2Unwrapped,
	_SecondUnwrappedForwardIterator_	_Last2Unwrapped,
	_Predicate_							_Predicate) noexcept(
		type_traits::is_nothrow_invocable_v<
			_Predicate_,
			type_traits::IteratorValueType<_FirstUnwrappedForwardIterator_>,
			type_traits::IteratorValueType<_FirstUnwrappedForwardIterator_>>)
{
	using _ValueType = type_traits::IteratorValueType<_FirstUnwrappedForwardIterator_>;

	if constexpr (
		type_traits::is_iterator_random_ranges_v<_FirstUnwrappedForwardIterator_> &&
		type_traits::is_iterator_random_ranges_v<_SecondUnwrappedForwardIterator_>)
	{
		const auto _FirstRangeLength = IteratorsDifference(_First1Unwrapped, _Last1Unwrapped);
		const auto _SecondRangeLength = IteratorsDifference(_First2Unwrapped, _Last2Unwrapped);

		if (_FirstRangeLength < _SecondRangeLength || _SecondRangeLength == 0)
			return _Last1Unwrapped;

		if constexpr (type_traits::is_vectorized_search_algorithm_safe_v<
			_FirstUnwrappedForwardIterator_, _SecondUnwrappedForwardIterator_, _Predicate_>)
		{
			const auto _First1Address = std::to_address(_First1Unwrapped);
			const auto _First2Address = std::to_address(_First2Unwrapped);

			const auto _Position = _FindEndVectorized<_ValueType>(
				_First1Address, _FirstRangeLength, _First2Address, _SecondRangeLength);

			if constexpr (std::is_pointer_v<_FirstUnwrappedForwardIterator_>)
				_SeekPossiblyWrappedIterator(_First1Unwrapped, reinterpret_cast<const _ValueType*>(_Position));
			else
				_SeekPossiblyWrappedIterator(_First1Unwrapped, _First1Unwrapped + static_cast<type_traits::IteratorDifferenceType<_FirstUnwrappedForwardIterator_>>(
					reinterpret_cast<const _ValueType*>(_Position) - _First1Address));

			return _First1Unwrapped;
		}
	}
	else if constexpr (
		type_traits::is_iterator_bidirectional_ranges_v<_FirstUnwrappedForwardIterator_> &&
		type_traits::is_iterator_bidirectional_ranges_v<_FirstUnwrappedForwardIterator_>)
	{
		for (auto _CandidateUnwrapped = _Last1Unwrapped;; --_CandidateUnwrapped) {
			auto _Next1Unwrapped = _CandidateUnwrapped;
			auto _Next2Unwrapped = _Last2Unwrapped;

			for (;;) {
				if (_First2Unwrapped == _Next2Unwrapped) {
					_SeekPossiblyWrappedIterator(_First1, _Next1Unwrapped);
					return _First1Unwrapped;
				}

				if (_First1Unwrapped == _Next1Unwrapped)
					return _Last1Unwrapped;

				--_Next1Unwrapped;
				--_Next2Unwrapped;

				if (_Predicate(*_Next1Unwrapped, *_Next2Unwrapped) == false)
					break;
			}
		}
	}
	else
	{
		auto _ResultUnwrapped = _Last1Unwrapped;

		for (;;) {
			auto _Next1Unwrapped = _First1Unwrapped;
			auto _Next2Unwrapped = _First2Unwrapped;

			for (;;) {
				const auto _NeedleEnd = (_Next2Unwrapped == _Last2Unwrapped);

				if (_NeedleEnd)
					_ResultUnwrapped = _First1Unwrapped;

				if (_Next1Unwrapped == _Last1Unwrapped) {
					_SeekPossiblyWrappedIterator(_First1Unwrapped, _ResultUnwrapped);
					return _First1Unwrapped;
				}

				if (_NeedleEnd || _Predicate(*_Next1Unwrapped, *_Next2Unwrapped) == false)
					++_Next1Unwrapped;

				++_Next2Unwrapped;
			}

			++_First1Unwrapped;

			_SeekPossiblyWrappedIterator(_First1Unwrapped, _ResultUnwrapped);
			return _First1Unwrapped;
		}
	}
}

__SIMD_STL_ALGORITHM_NAMESPACE_END
