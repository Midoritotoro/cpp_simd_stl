#pragma once 

#include <src/simd_stl/algorithm/AlgorithmDebug.h>
#include <src/simd_stl/type_traits/SimdAlgorithmSafety.h>

#include <src/simd_stl/algorithm/MsvcIteratorUnwrap.h>
#include <simd_stl/concurrency/Execution.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN


template <
	class _InputUnwrappedIterator_,
	class _Predicate_>
simd_stl_nodiscard simd_stl_constexpr_cxx20 simd_stl_always_inline _InputUnwrappedIterator_ _FindIfUnchecked(
	_InputUnwrappedIterator_	_FirstUnwrapped,
	_InputUnwrappedIterator_	_LastUnwrapped,
	_Predicate_					_Predicate) noexcept(
		type_traits::is_nothrow_invocable_v<
			_Predicate_, type_traits::IteratorValueType<_InputUnwrappedIterator_>>)
{
	for (; _FirstUnwrapped != _LastUnwrapped; ++_FirstUnwrapped)
		if (_Predicate(*_FirstUnwrapped))
			break;

	return _FirstUnwrapped;
}

__SIMD_STL_ALGORITHM_NAMESPACE_END
