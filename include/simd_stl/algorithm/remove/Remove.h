#pragma once 

#include <src/simd_stl/algorithm/unchecked/remove/RemoveUnchecked.h>
#include <src/simd_stl/algorithm/unchecked/remove/RemoveIfUnchecked.h>

#include <simd_stl/concurrency/Execution.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <
	class _InputIterator_,
	class _Type_ = type_traits::iterator_value_type<_InputIterator_>>
simd_stl_nodiscard simd_stl_always_inline simd_stl_constexpr_cxx20 _InputIterator_ remove(
	_InputIterator_										__first,
	_InputIterator_										__last,
	const typename std::type_identity<_Type_>::type&	__value) noexcept
{
	__verify_range(__first, __last);

	__seek_possibly_wrapped_iterator(__first, __remove_unchecked(
		__unwrap_iterator(__first), __unwrap_iterator(__last), __value));

	return __first;
}

template <
	class _Iterator_,
	class _UnaryPredicate_>
simd_stl_nodiscard simd_stl_always_inline simd_stl_constexpr_cxx20 _Iterator_ remove_if(
	_Iterator_			__first,
	_Iterator_			__last,
	_UnaryPredicate_	__predicate) noexcept(
		type_traits::is_nothrow_invocable_v<_UnaryPredicate_,
			type_traits::iterator_value_type<_Iterator_>>)
{
	__verify_range(__first, __last);

	__seek_possibly_wrapped_iterator(__first, __remove_if_unchecked(
		__unwrap_iterator(__first), __unwrap_iterator(__last), type_traits::__pass_function(__predicate)));

	return __first;
}

template <
	class _ExecutionPolicy_,
	class _Iterator_,
	class _Type_,
	concurrency::enable_if_execution_policy<_ExecutionPolicy_> = 0>
simd_stl_nodiscard simd_stl_always_inline simd_stl_constexpr_cxx20 _Iterator_ remove(
	_ExecutionPolicy_&&,
	_Iterator_		__first,
	_Iterator_		__last,
	const _Type_&	__value) noexcept
{
	return simd_stl::algorithm::remove(__first, __last, __value);
}

template <
	class _ExecutionPolicy_,
	class _Iterator_,
	class _UnaryPredicate_,
	concurrency::enable_if_execution_policy<_ExecutionPolicy_> = 0>
simd_stl_nodiscard simd_stl_always_inline simd_stl_constexpr_cxx20 _Iterator_ remove_if(
	_ExecutionPolicy_&&,
	_Iterator_			__first,
	_Iterator_			__last,
	_UnaryPredicate_	__predicate) noexcept(
		type_traits::is_nothrow_invocable_v<_UnaryPredicate_,
			type_traits::iterator_value_type<_Iterator_>>)
{
	return simd_stl::algorithm::remove_if(__first, __last, type_traits::__pass_function(__predicate));
}

__SIMD_STL_ALGORITHM_NAMESPACE_END
