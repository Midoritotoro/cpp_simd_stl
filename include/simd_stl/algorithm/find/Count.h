#pragma once 

#include <src/simd_stl/algorithm/AlgorithmDebug.h>
#include <src/simd_stl/type_traits/SimdAlgorithmSafety.h>

#include <simd_stl/compatibility/Nodiscard.h>
#include <simd_stl/compatibility/Inline.h>

#include <src/simd_stl/algorithm/vectorized/CountVectorized.h>
#include <src/simd_stl/algorithm/MsvcIteratorUnwrap.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <
	class _Iterator_,
	class _Type_>
_Simd_nodiscard_inline_constexpr sizetype count(
	_Iterator_		_First,
	_Iterator_		_Last,
	const _Type_&	_Value) noexcept
{
	__verifyRange(_First, _Last);

	using _UnwrappedIteratorType = unwrapped_iterator_type<_Iterator_>;

	auto _FirstUnwrapped		= _UnwrapIterator(_First);
	const auto _LastUnwrapped	= _UnwrapIterator(_Last);

	if constexpr (type_traits::is_iterator_random_ranges_v<_UnwrappedIteratorType>) {
		const auto _Size = ByteLength(_FirstUnwrapped, _LastUnwrapped);

		if constexpr (type_traits::is_vectorized_find_algorithm_safe_v<_UnwrappedIteratorType, _Type_>) {
#if simd_stl_has_cxx20
			if (type_traits::is_constant_evaluated() == false)
#endif // simd_stl_has_cxx20
			{
				if (math::couldCompareEqualToValueType<_UnwrappedIteratorType>(_Value) == false)
					return 0;

				return _CountVectorized<_Type_>(std::to_address(_FirstUnwrapped), _Size, _Value);
			}
		}
	}

    type_traits::IteratorDifferenceType<_Iterator_> _Count = 0;

	for (; _FirstUnwrapped != _LastUnwrapped; ++_FirstUnwrapped)
		_Count += (*_FirstUnwrapped == _Value);

	return _Count;
}

template <
	class _InputIterator_,
	class _Predicate_>
_Simd_nodiscard_inline_constexpr type_traits::IteratorDifferenceType<_InputIterator_> count_if(
	_InputIterator_	_First,
	_InputIterator_	_Last,
	_Predicate_ 	_Predicate) noexcept(
		type_traits::is_nothrow_invocable_v<
			_Predicate_, type_traits::IteratorValueType<_InputIterator_>>)
{
	__verifyRange(_First, _Last);

	auto _FirstUnwrapped		= _UnwrapIterator(_First);
	const auto _LastUnwrapped	= _UnwrapIterator(_Last);

	auto _Count = type_traits::IteratorDifferenceType<_InputIterator_>(0);

	for (; _FirstUnwrapped != _LastUnwrapped; ++_FirstUnwrapped)
		if (_Predicate(*_FirstUnwrapped))
			++_Count;

	return _Count;
}

template <
	class _ExecutionPolicy_,
	class _Iterator_,
	class _Type_>
simd_stl_nodiscard sizetype count(
	_ExecutionPolicy_&&,
	_Iterator_		_First,
	_Iterator_		_Last,
	const _Type_&	_Value) noexcept
{
	return simd_stl::algorithm::count(_First, _Last, _Value);
}

template <
	class _ExecutionPolicy_,
	class _InputIterator_,
	class _Predicate_>
simd_stl_nodiscard type_traits::IteratorDifferenceType<_InputIterator_> count_if(
	_ExecutionPolicy_&&,
	_InputIterator_			_First,
	const _InputIterator_	_Last,
	_Predicate_ 			_Predicate) noexcept(
		type_traits::is_nothrow_invocable_v<
			_Predicate_, type_traits::IteratorValueType<_InputIterator_>>)
{
	return simd_stl::algorithm::count_if(_First, _Last, type_traits::passFunction(_Predicate));
}

__SIMD_STL_ALGORITHM_NAMESPACE_END
