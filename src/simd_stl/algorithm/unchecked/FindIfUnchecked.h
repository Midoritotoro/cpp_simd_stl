#pragma once 

#include <src/simd_stl/algorithm/AlgorithmDebug.h>
#include <src/simd_stl/type_traits/SimdAlgorithmSafety.h>

#include <src/simd_stl/algorithm/vectorized/FindVectorized.h>

#include <src/simd_stl/algorithm/MsvcIteratorUnwrap.h>
#include <simd_stl/concurrency/Execution.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN


template <
	class _InputUnwrappedIterator_,
	class _Predicate_>
simd_stl_nodiscard simd_stl_constexpr_cxx20 simd_stl_always_inline _InputUnwrappedIterator_ _FindIfUnchecked(
	_InputUnwrappedIterator_		firstUnwrapped,
	const _InputUnwrappedIterator_	lastUnwrapped,
	_Predicate_						predicate) noexcept(
		type_traits::is_nothrow_invocable_v<
			_Predicate_,
			type_traits::IteratorValueType<_InputUnwrappedIterator_>>)
{
	for (; firstUnwrapped != lastUnwrapped; ++firstUnwrapped)
		if (predicate(*firstUnwrapped))
			break;

	return firstUnwrapped;
}

__SIMD_STL_ALGORITHM_NAMESPACE_END
