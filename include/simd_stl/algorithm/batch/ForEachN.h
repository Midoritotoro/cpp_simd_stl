#pragma once 

#include <src/simd_stl/algorithm/unchecked/batch/ForEachNUnchecked.h>
#include <simd_stl/concurrency/Execution.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <
	class _InputIterator_,
	class _SizeType_,
	class _UnaryFunction_>
__simd_inline_constexpr _UnaryFunction_ for_each_n(
	_InputIterator_	__first,
	_SizeType_		__count,
	_UnaryFunction_	__function) noexcept(
		type_traits::is_nothrow_invocable_v<_UnaryFunction_,
			type_traits::iterator_value_type<_InputIterator_>>)
{
	__verify_range(__first, __last);
	__for_each_n_unchecked(__unwrap_iterator(__first), __count, type_traits::__pass_function(__function));

	return __function;
}

template <
	class _ExecutionPolicy_,
	class _InputIterator_,
	class _SizeType_,
	class _UnaryFunction_,
	concurrency::enable_if_execution_policy<_ExecutionPolicy_> = 0>
simd_stl_always_inline void for_each_n(
	_ExecutionPolicy_&&,
	_InputIterator_	__first,
	_SizeType_		__count,
	_UnaryFunction_	__function) noexcept(
		type_traits::is_nothrow_invocable_v<_UnaryFunction_,
			type_traits::iterator_value_type<_InputIterator_>>)
{
	return simd_stl::algorithm::for_each_n(__first, __count, type_traits::__pass_function(__function));
}

__SIMD_STL_ALGORITHM_NAMESPACE_END
