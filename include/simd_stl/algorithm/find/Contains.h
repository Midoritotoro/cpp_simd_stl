#pragma once 

#include <src/simd_stl/algorithm/AlgorithmDebug.h>
#include <src/simd_stl/type_traits/SimdAlgorithmSafety.h>

#include <simd_stl/compatibility/Nodiscard.h>
#include <simd_stl/compatibility/Inline.h>

#include <src/simd_stl/algorithm/vectorized/ContainsVectorized.h>
#include <src/simd_stl/algorithm/MsvcIteratorUnwrap.h>

__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <
	class _Iterator_,
	class _Type_>
simd_stl_nodiscard simd_stl_always_inline simd_stl_constexpr_cxx20 bool contains(
	_Iterator_			first,
	const _Iterator_	last,
	const _Type_&		value) noexcept
{
	__verifyRange(first, last);

#if defined(simd_stl_cpp_msvc)
	using _IteratorUnwrappedType_ = std::_Unwrapped_t<_Iterator_>;
#else 
	using _IteratorUnwrappedType_ = _Iterator_;
#endif // defined(simd_stl_cpp_msvc) 

	if constexpr (type_traits::is_vectorized_find_algorithm_safe_v<_IteratorUnwrappedType_, _Type_>) {
		auto firstUnwrapped			= __unwrapIterator(first);
		const auto lastUnwrapped	= __unwrapIterator(last);

#if simd_stl_has_cxx20
		if (type_traits::is_constant_evaluated() == false)
#endif
			return ContainsVectorized(std::to_address(firstUnwrapped), std::to_address(lastUnwrapped), value);
	}

	for (; first != last; ++first)
		if (*first == value)
			return true;

	return false;
}

__SIMD_STL_ALGORITHM_NAMESPACE_END
