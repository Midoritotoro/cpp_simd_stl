#pragma once 

#include <src/simd_stl/algorithm/unchecked/remove/RemoveUnchecked.h>
#include <src/simd_stl/algorithm/unchecked/remove/RemoveIfUnchecked.h>

#include <simd_stl/concurrency/Execution.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <
	class _InputIterator_,
	class _Type_ = type_traits::IteratorValueType<_InputIterator_>>
simd_stl_nodiscard simd_stl_always_inline simd_stl_constexpr_cxx20 _InputIterator_ remove(
	_InputIterator_										_First,
	_InputIterator_										_Last,
	const typename std::type_identity<_Type_>::type&	_Value) noexcept
{
	__verifyRange(_First, _Last);

	__seek_possibly_wrapped_iterator(_First, _RemoveUnchecked(
		_UnwrapIterator(_First), _UnwrapIterator(_Last), _Value));

	return _First;
}

template <
	class _Iterator_,
	class _UnaryPredicate_>
simd_stl_nodiscard simd_stl_always_inline simd_stl_constexpr_cxx20 _Iterator_ remove_if(
	_Iterator_			_First,
	_Iterator_			_Last,
	_UnaryPredicate_	_Predicate) noexcept(
		type_traits::is_nothrow_invocable_v<_UnaryPredicate_,
			type_traits::IteratorValueType<_Iterator_>>)
{
	__verifyRange(_First, _Last);

	__seek_possibly_wrapped_iterator(_First, _RemoveIfUnchecked(
		_UnwrapIterator(_First), _UnwrapIterator(_Last), type_traits::passFunction(_Predicate)));

	return _First;
}

template <
	class _ExecutionPolicy_,
	class _Iterator_,
	class _Type_,
	concurrency::enable_if_execution_policy<_ExecutionPolicy_> = 0>
simd_stl_nodiscard simd_stl_always_inline simd_stl_constexpr_cxx20 _Iterator_ remove(
	_ExecutionPolicy_&&,
	_Iterator_		first,
	_Iterator_		last,
	const _Type_&	value) noexcept
{
	return simd_stl::algorithm::remove(first, last, value);
}

template <
	class _ExecutionPolicy_,
	class _Iterator_,
	class _UnaryPredicate_,
	concurrency::enable_if_execution_policy<_ExecutionPolicy_> = 0>
simd_stl_nodiscard simd_stl_always_inline simd_stl_constexpr_cxx20 _Iterator_ remove_if(
	_ExecutionPolicy_&&,
	_Iterator_			first,
	_Iterator_			last,
	_UnaryPredicate_	predicate) noexcept(
		type_traits::is_nothrow_invocable_v<_UnaryPredicate_,
			type_traits::IteratorValueType<_Iterator_>>)
{
	return simd_stl::algorithm::remove_if(first, last, type_traits::passFunction(predicate));
}

__SIMD_STL_ALGORITHM_NAMESPACE_END
