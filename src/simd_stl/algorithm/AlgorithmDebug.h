#pragma once 

#include <src/simd_stl/utility/Assert.h>
#include <simd_stl/compatibility/Inline.h>

#include <src/simd_stl/type_traits/IteratorCheck.h>
#include <src/simd_stl/type_traits/IntegralProperties.h>

#include <simd_stl/math/IntegralTypesConversions.h>

__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <class _Type_>
simd_stl_constexpr_cxx20 void _VerifyRange(
	const _Type_* const firstPointer,
	const _Type_* const lastPointer) noexcept
{
	DebugAssertLog(firstPointer <= lastPointer, "transposed pointer range");
}

template <
	class _Iterator_,
	class _Sentinel_>
simd_stl_constexpr_cxx20 void _VerifyRange(
	const _Iterator_& firstIterator,
	const _Sentinel_& lastIterator) noexcept
{
#if !defined(NDEBUG)
	if constexpr (std::is_pointer_v<_Iterator_> && std::is_pointer_v<_Sentinel_>) {
		DebugAssertLog(firstIterator <= lastIterator, "transposed pointer range");
		return;
	}
	else if constexpr (type_traits::is_range_verifiable_v<_Iterator_, _Sentinel_>) {
		_VerifyRange(
			const_cast<const char*>(reinterpret_cast<const volatile char*>(std::to_address(firstIterator))),
			const_cast<const char*>(reinterpret_cast<const volatile char*>(std::to_address(lastIterator))));
	}
#endif
}

#if !defined(__verifyRange)
#  if !defined(NDEBUG)
#    define __verifyRange(first, last)  _VerifyRange(first, last) 
#  else
#    define __verifyRange(first, last) 
#  endif // !defined(NDEBUG)
#endif // !defined(__verifyRange)

__SIMD_STL_ALGORITHM_NAMESPACE_END
