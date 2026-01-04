#pragma once 

#include <src/simd_stl/algorithm/unchecked/batch/ForEachNUnchecked.h>
#include <simd_stl/concurrency/Execution.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <
	class _InputIterator_,
	class _SizeType_,
	class _UnaryFunction_>
__simd_inline_constexpr _UnaryFunction_ for_each_n(
	_InputIterator_	_First,
	_SizeType_		_Count,
	_UnaryFunction_	_Function) noexcept(
		type_traits::is_nothrow_invocable_v<_UnaryFunction_,
			type_traits::iterator_value_type<_InputIterator_>>)
{
	__verifyRange(_First, _Last);

	_ForEachNUnchecked(_UnwrapIterator(_First), _Count, type_traits::passFunction(_Function));

	return _Function;
}

template <
	class _ExecutionPolicy_,
	class _InputIterator_,
	class _SizeType_,
	class _UnaryFunction_,
	concurrency::enable_if_execution_policy<_ExecutionPolicy_> = 0>
simd_stl_always_inline void for_each_n(
	_ExecutionPolicy_&&,
	_InputIterator_	_First,
	_SizeType_		_Count,
	_UnaryFunction_	_Function) noexcept(
		type_traits::is_nothrow_invocable_v<_UnaryFunction_,
			type_traits::iterator_value_type<_InputIterator_>>)
{
	return simd_stl::algorithm::for_each_n(_First, _Count, type_traits::passFunction(_Function));
}

__SIMD_STL_ALGORITHM_NAMESPACE_END
