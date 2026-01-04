#pragma once 

#include <src/simd_stl/algorithm/unchecked/find/CountIfUnchecked.h>
#include <src/simd_stl/algorithm/unchecked/find/CountUnchecked.h>

#include <simd_stl/concurrency/Execution.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <
	class _Iterator_,
	class _Type_ = type_traits::iterator_value_type<_Iterator_>>
__simd_nodiscard_inline_constexpr sizetype count(
	_Iterator_											__first,
	_Iterator_											_Last,
	const typename std::type_identity<_Type_>::type&	_Value) noexcept
{
	__verifyRange(__first, _Last);
	return _CountUnchecked(_UnwrapIterator(__first), _UnwrapIterator(_Last), _Value);
}

template <
	class _InputIterator_,
	class _Predicate_>
__simd_nodiscard_inline_constexpr type_traits::iterator_difference_type<_InputIterator_> count_if(
	_InputIterator_	__first,
	_InputIterator_	_Last,
	_Predicate_ 	_Predicate) noexcept(
		type_traits::is_nothrow_invocable_v<
			_Predicate_, type_traits::iterator_value_type<_InputIterator_>>)
{
	__verifyRange(__first, _Last);
	return _CountIfUnchecked(_UnwrapIterator(__first), _UnwrapIterator(_Last), type_traits::passFunction(_Predicate));
}

template <
	class _ExecutionPolicy_,
	class _Iterator_,
	class _Type_ = type_traits::iterator_value_type<_Iterator_>,
	concurrency::enable_if_execution_policy<_ExecutionPolicy_> = 0>
simd_stl_nodiscard sizetype count(
	_ExecutionPolicy_&&,
	_Iterator_											__first,
	_Iterator_											_Last,
	const typename std::type_identity<_Type_>::type&	_Value) noexcept
{
	return simd_stl::algorithm::count(__first, _Last, _Value);
}

template <
	class _ExecutionPolicy_,
	class _InputIterator_,
	class _Predicate_,
	concurrency::enable_if_execution_policy<_ExecutionPolicy_> = 0>
simd_stl_nodiscard type_traits::iterator_difference_type<_InputIterator_> count_if(
	_ExecutionPolicy_&&,
	_InputIterator_			__first,
	const _InputIterator_	_Last,
	_Predicate_ 			_Predicate) noexcept(
		type_traits::is_nothrow_invocable_v<
			_Predicate_, type_traits::iterator_value_type<_InputIterator_>>)
{
	return simd_stl::algorithm::count_if(__first, _Last, type_traits::passFunction(_Predicate));
}

__SIMD_STL_ALGORITHM_NAMESPACE_END
