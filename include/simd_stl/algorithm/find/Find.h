#pragma once 

#include <src/core/algorithm/AlgorithmDebug.h>
#include <base/core/type_traits/SimdAlgorithmSafety.h>

__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN


template <
	class _Iterator_,
	class _Type_>
simd_stl_nodiscard simd_stl_constexpr_cxx20 _Iterator_ find(
	_Iterator_			firstIterator,
	const _Iterator_	lastIterator,
	const _Type_&		value) noexcept
{

}

__SIMD_STL_ALGORITHM_NAMESPACE_END
