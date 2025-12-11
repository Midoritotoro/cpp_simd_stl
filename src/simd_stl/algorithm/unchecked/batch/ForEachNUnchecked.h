#pragma once 

#include <src/simd_stl/algorithm/AlgorithmDebug.h>
#include <src/simd_stl/algorithm/MsvcIteratorUnwrap.h>

#include <simd_stl/concurrency/Execution.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <
	class _UnwrappedInputIterator_,
	class _SizeType_,
	class _UnaryFunction_>
_Simd_inline_constexpr void _ForEachNUnchecked(
	_UnwrappedInputIterator_	_FirstUnwrapped,
	_SizeType_					_Count,
	_UnaryFunction_				_Function) noexcept(
		type_traits::is_nothrow_invocable_v<_UnaryFunction_,
			type_traits::IteratorValueType<_UnwrappedInputIterator_>>)
{
	for (auto _Index = 0; _Index < _Count; ++_Index, ++_FirstUnwrapped)
		_Function(*_FirstUnwrapped);
}

__SIMD_STL_ALGORITHM_NAMESPACE_END
