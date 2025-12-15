#pragma once 

#include <src/simd_stl/algorithm/unchecked/find/CountIfUnchecked.h>
#include <src/simd_stl/algorithm/unchecked/find/CountUnchecked.h>

#include <simd_stl/concurrency/Execution.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <
	class _Iterator_,
	class _Type_ = type_traits::IteratorValueType<_Iterator_>>
_Simd_nodiscard_inline_constexpr sizetype count(
	_Iterator_											_First,
	_Iterator_											_Last,
	const typename std::type_identity<_Type_>::type&	_Value) noexcept
{
	__verifyRange(_First, _Last);
	return _CountUnchecked(_UnwrapIterator(_First), _UnwrapIterator(_Last), _Value);
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
	return _CountIfUnchecked(_UnwrapIterator(_First), _UnwrapIterator(_Last), type_traits::passFunction(_Predicate));
}

template <
	class _ExecutionPolicy_,
	class _Iterator_,
	class _Type_ = type_traits::IteratorValueType<_Iterator_>,
	concurrency::enable_if_execution_policy<_ExecutionPolicy_> = 0>
simd_stl_nodiscard sizetype count(
	_ExecutionPolicy_&&,
	_Iterator_											_First,
	_Iterator_											_Last,
	const typename std::type_identity<_Type_>::type&	_Value) noexcept
{
	return simd_stl::algorithm::count(_First, _Last, _Value);
}

template <
	class _ExecutionPolicy_,
	class _InputIterator_,
	class _Predicate_,
	concurrency::enable_if_execution_policy<_ExecutionPolicy_> = 0>
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
