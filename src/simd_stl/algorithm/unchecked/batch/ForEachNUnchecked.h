#pragma once 

#include <src/simd_stl/algorithm/AlgorithmDebug.h>
#include <src/simd_stl/algorithm/MsvcIteratorUnwrap.h>

#include <simd_stl/concurrency/Execution.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <
	class _UnwrappedInputIterator_,
	class _SizeType_,
	class _UnaryFunction_>
__simd_inline_constexpr void __for_each_n_unchecked(
	_UnwrappedInputIterator_	__first_unwrapped,
	_SizeType_					__count,
	_UnaryFunction_				__function) noexcept(
		type_traits::is_nothrow_invocable_v<_UnaryFunction_,
			type_traits::IteratorValueType<_UnwrappedInputIterator_>>)
{
	for (auto __index = 0; __index < __count; ++__index, ++__first_unwrapped)
		__function(*__first_unwrapped);
}

__SIMD_STL_ALGORITHM_NAMESPACE_END
