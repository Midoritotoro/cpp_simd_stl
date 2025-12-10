#pragma once 

#include <src/simd_stl/algorithm/AlgorithmDebug.h>
#include <src/simd_stl/algorithm/MsvcIteratorUnwrap.h>

#include <simd_stl/concurrency/Execution.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <
	class _UnwrappedInputIterator_,
	class _UnaryFunction_>
_Simd_inline_constexpr void _ForEachUnchecked(
	_UnwrappedInputIterator_	_FirstUnwrapped,
	_UnwrappedInputIterator_	_LastUnwrapped,
	_UnaryFunction_				_Function) noexcept(
		type_traits::is_nothrow_invocable< _UnaryFunction_,
			type_traits::IteratorValueType<_UnwrappedInputIterator>)
{
	for (; _FirstUnwrapped != _LastUnwrapped; ++_FirstUnwrapped)
		_Function(*_FirstUnwrapped);
}

__SIMD_STL_ALGORITHM_NAMESPACE_END
