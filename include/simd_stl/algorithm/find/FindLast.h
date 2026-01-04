#pragma once 

#include <src/simd_stl/algorithm/unchecked/find/FindLastUnchecked.h>
#include <src/simd_stl/algorithm/unchecked/find/FindLastIfUnchecked.h>

#include <src/simd_stl/algorithm/unchecked/find/FindLastIfNotUnchecked.h>
#include <simd_stl/concurrency/Execution.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <
	class _Iterator_,
	class _Type_>
__simd_nodiscard_inline_constexpr _Iterator_ find_last(
	_Iterator_		_First,
	_Iterator_		_Last,
	const _Type_&	_Value) noexcept
{
	__verifyRange(_First, _Last);

	__seek_possibly_wrapped_iterator(_First, _FindLastUnchecked(_UnwrapIterator(_First), _UnwrapIterator(_Last), _Value));
	return _First;
}

template <
	class _InputIterator_,
	class _Predicate_>
__simd_nodiscard_inline_constexpr _InputIterator_ find_last_if_not(
	_InputIterator_	_First, 
	_InputIterator_	_Last, 
	_Predicate_		_Predicate) noexcept(
		type_traits::is_nothrow_invocable_v<
			_Predicate_, type_traits::iterator_value_type<_InputIterator_>>)
{
	__verifyRange(_First, _Last);

	__seek_possibly_wrapped_iterator(_First, _FindLastIfNotUnchecked(_UnwrapIterator(_First), _UnwrapIterator(_Last),
		type_traits::passFunction(_Predicate)));

	return _First;
}

template <
	class _InputIterator_,
	class _Predicate_>
__simd_nodiscard_inline_constexpr _InputIterator_ find_last_if(
	_InputIterator_	_First, 
	_InputIterator_	_Last,
	_Predicate_		_Predicate) noexcept(
		type_traits::is_nothrow_invocable_v<
			_Predicate_, type_traits::iterator_value_type<_InputIterator_>>)
{
	__verifyRange(_First, _Last);
	
	__seek_possibly_wrapped_iterator(_First, _FindLastIfUnchecked(_UnwrapIterator(_First), _UnwrapIterator(_Last),
		type_traits::passFunction(_Predicate)));

	return _First;
}

template <
	class _ExecutionPolicy_,
	class _Iterator_,
	class _Type_,
	concurrency::enable_if_execution_policy<_ExecutionPolicy_> = 0>
__simd_nodiscard_inline_constexpr _Iterator_ find_last(
	_ExecutionPolicy_&&,
	_Iterator_		_First,
	_Iterator_		_Last,
	const _Type_&	_Value) noexcept
{
	return simd_stl::algorithm::find_last(_First, _Last, _Value);
}

template <
	class _ExecutionPolicy_,
	class _InputIterator_,
	class _Predicate_,
	concurrency::enable_if_execution_policy<_ExecutionPolicy_> = 0>
__simd_nodiscard_inline_constexpr _InputIterator_ find_last_if_not(
	_ExecutionPolicy_&&,
	_InputIterator_	_First, 
	_InputIterator_	_Last, 
	_Predicate_		_Predicate) noexcept(
		type_traits::is_nothrow_invocable_v<
			_Predicate_, type_traits::iterator_value_type<_InputIterator_>>)
{
	return simd_stl::algorithm::find_last_if_not(_First, _Last, type_traits::passFunction(_Predicate));
}

template <
	class _ExecutionPolicy_,
	class _InputIterator_,
	class _Predicate_,
	concurrency::enable_if_execution_policy<_ExecutionPolicy_> = 0>
__simd_nodiscard_inline_constexpr _InputIterator_ find_last_if(
	_ExecutionPolicy_&&,
	_InputIterator_	_First, 
	_InputIterator_	_Last,
	_Predicate_		_Predicate) noexcept(
		type_traits::is_nothrow_invocable_v<
			_Predicate_, type_traits::iterator_value_type<_InputIterator_>>)
{
	return simd_stl::algorithm::find_last_if(_First, _Last, type_traits::passFunction(_Predicate));
}

__SIMD_STL_ALGORITHM_NAMESPACE_END
