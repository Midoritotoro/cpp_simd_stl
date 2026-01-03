#pragma once 

#include <src/simd_stl/algorithm/unchecked/remove/RemoveCopyUnchecked.h>
#include <src/simd_stl/algorithm/unchecked/remove/RemoveCopyIfUnchecked.h>

#include <simd_stl/concurrency/Execution.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <
	class _InputIterator_,
	class _OutputIterator_,
	class _Type_ = type_traits::IteratorValueType<_InputIterator_>>
__simd_nodiscard_inline_constexpr _OutputIterator_ remove_copy(
	_InputIterator_										_First,
	_InputIterator_										_Last,
	_OutputIterator_									_Destination,
	const typename std::type_identity<_Type_>::type&	_Value) noexcept
{
	__verifyRange(first, last);
	
	__seek_possibly_wrapped_iterator(_Destination, _RemoveCopyUnchecked(_UnwrapIterator(_First),
		_UnwrapIterator(_Last), _UnwrapIterator(_Destination), _Value));

	return _Destination;
}

template <
	class _InputIterator_,
	class _OutputIterator_,
	class _UnaryPredicate_>
__simd_nodiscard_inline_constexpr _OutputIterator_ remove_copy_if(
	_InputIterator_		_First,
	_InputIterator_		_Last,
	_OutputIterator_	_Destination,
	_UnaryPredicate_	_Predicate) noexcept
{
	__verifyRange(_First, _Last);
	
	__seek_possibly_wrapped_iterator(_Destination, _RemoveCopyIfUnchecked(_UnwrapIterator(_First),
		_UnwrapIterator(_Last), _UnwrapIterator(_Destination), type_traits::passFunction(_Predicate)));

	return _Destination;
}

template <
	class _ExecutionPolicy_,
	class _InputIterator_,
	class _OutputIterator_,
	class _Type_,
	concurrency::enable_if_execution_policy<_ExecutionPolicy_> = 0>
__simd_nodiscard_inline _OutputIterator_ remove_copy(
	_ExecutionPolicy_&&,
	_InputIterator_		first,
	_InputIterator_		last,
	_OutputIterator_	destination,
	const _Type_&		value) noexcept
{
	return simd_stl::algorithm::remove_copy(first, last, destination, value);
}

template <
	class _ExecutionPolicy_,
	class _InputIterator_,
	class _OutputIterator_,
	class _UnaryPredicate_,
	concurrency::enable_if_execution_policy<_ExecutionPolicy_> = 0>
__simd_nodiscard_inline _OutputIterator_ remove_copy_if(
	_ExecutionPolicy_&&,
	_InputIterator_		first,
	_InputIterator_		last,
	_OutputIterator_	destination,
	_UnaryPredicate_	predicate) noexcept
{
	return simd_stl::algorithm::remove_copy_if(first, last, destination, type_traits::passFunction(predicate));
}

__SIMD_STL_ALGORITHM_NAMESPACE_END
