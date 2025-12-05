#pragma once 

#include <src/simd_stl/algorithm/AlgorithmDebug.h>

#include <src/simd_stl/algorithm/vectorized/SearchVectorized.h>
#include <src/simd_stl/type_traits/CanMemcmpElements.h>

#include <src/simd_stl/algorithm/MsvcIteratorUnwrap.h>
#include <src/simd_stl/type_traits/OperatorWrappers.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <
	class _FirstUnwrappedForwardIterator_,
	class _SecondUnwrappedForwardIterator_,
	class _Predicate_> 
_Simd_nodiscard_inline_constexpr _FirstUnwrappedForwardIterator_ _SearchUnchecked(
	_FirstUnwrappedForwardIterator_		_First1Unwrapped,
	_FirstUnwrappedForwardIterator_		_Last1Unwrapped,
	_SecondUnwrappedForwardIterator_	_First2Unwrapped,
	_SecondUnwrappedForwardIterator_	_Last2Unwrapped,
	_Predicate_							_Predicate) noexcept(
		type_traits::is_nothrow_invocable_v<
			_Predicate_,
			type_traits::IteratorValueType<_FirstUnwrappedForwardIterator_>,
			type_traits::IteratorValueType<_SecondUnwrappedForwardIterator_>>)
{
	using _Value_ = type_traits::IteratorValueType<_FirstUnwrappedForwardIterator_>;
		
	if constexpr (
		type_traits::is_iterator_random_ranges_v<_FirstUnwrappedForwardIterator_> &&
		type_traits::is_iterator_random_ranges_v<_SecondUnwrappedForwardIterator_>)
	{
		const auto _First1Address	= std::to_address(_First1Unwrapped);
		const auto _First2Address	= std::to_address(_First2Unwrapped);

		auto _FirstRangeLength			= IteratorsDifference(_First1Unwrapped, _Last1Unwrapped);
		const auto _SecondRangeLength	= IteratorsDifference(_First2Unwrapped, _Last2Unwrapped);

		if (_FirstRangeLength < _SecondRangeLength)
			return _Last1Unwrapped;
	
		if constexpr (type_traits::is_vectorized_search_algorithm_safe_v<
			_FirstUnwrappedForwardIterator_, _SecondUnwrappedForwardIterator_, _Predicate_>)
		{
#if simd_stl_has_cxx20
			if (type_traits::is_constant_evaluated() == false)
#endif // simd_stl_has_cxx20
			{
				const auto _Position = _SearchVectorized(_First1Address, _FirstRangeLength, _First2Address, _SecondRangeLength);

				return (_Position == nullptr) 
					? _Last1Unwrapped 
					: _First1Unwrapped + (reinterpret_cast<const _Value_*>(_Position) - _First1Address);
			}
		}

		const auto _Position = _Search<arch::CpuFeature::None>()(
			_First1Address, _FirstRangeLength, _First2Address,
			_SecondRangeLength, type_traits::passFunction(_Predicate));

		return (_Position == nullptr)
			? _Last1Unwrapped 
			: _First1Unwrapped + (reinterpret_cast<const _Value_*>(_Position) - _First1Address);
	}


	const auto _LastPossible = _Last1Unwrapped - IteratorsDifference(_First2Unwrapped, _Last2Unwrapped);
	auto _Mirst1Unwrapped = _First1Unwrapped;

	for (;; ++_First1Unwrapped) {
		auto _Mid1 = _First1Unwrapped;
		
		for (auto _Mid2 = _First2Unwrapped; ++_Mid1; (void) ++_Mid2) {
			if (_Mid2 == _Last2Unwrapped)
				return _First1Unwrapped;

			if (_Mid1 == _Last1Unwrapped)
				return _Last1Unwrapped;

			if (!_Predicate(*_Mid1, *_Mid2))
				break;
		}
	}

	return _Last1Unwrapped;
}

__SIMD_STL_ALGORITHM_NAMESPACE_END
