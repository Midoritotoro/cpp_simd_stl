#pragma once 

#include <src/simd_stl/unchecked/ForEachUnchecked.h>
#include <simd_stl/concurrency/Execution.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <
	class _InputIterator_,
	class _UnaryFunction_>
_Simd_inline_constexpr _UnaryFunction_ for_each(
	_InputIterator_	_First,
	_InputIterator_	_Last,
	_UnaryFunction_	_Function) noexcept(
		type_traits::is_nothrow_invocable< _UnaryFunction_,
			type_traits::IteratorValueType<_UnwrappedInputIterator>)
{
	__verifyRange(_First, _Last);

	_ForEachUnchecked(_UnwrapIterator(_First), _UnwrapIterator(_Last),
		type_traits::passFunction(_Function));

	return _Function;
}

template <
	class _ExecutionPolicy_,
	class _InputIterator_,
	class _UnaryFunction_,
	concurrency::enable_if_execution_policy<_ExecutionPolicy_> = 0>
simd_stl_always_inline void for_each(
	_ExecutionPolicy_&&,
	_InputIterator_	_First,
	_InputIterator_	_Last,
	_UnaryFunction_	_Function) noexcept(
		type_traits::is_nothrow_invocable< _UnaryFunction_,
			type_traits::IteratorValueType<_UnwrappedInputIterator>)

__SIMD_STL_ALGORITHM_NAMESPACE_END
