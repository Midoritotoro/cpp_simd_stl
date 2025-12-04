#pragma once 

#include <src/simd_stl/utility/Assert.h>
#include <simd_stl/compatibility/Inline.h>

#include <src/simd_stl/type_traits/IteratorCheck.h>
#include <src/simd_stl/type_traits/IntegralProperties.h>

#include <simd_stl/math/IntegralTypesConversions.h>

__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <class _Type_>
simd_stl_constexpr_cxx20 void _VerifyRange(
	const _Type_* const _FirstPointer,
	const _Type_* const _LastPointer) noexcept
{
	DebugAssertLog(_FirstPointer <= _LastPointer, "transposed pointer range");
}

template <
	class _Iterator_,
	class _Sentinel_>
simd_stl_constexpr_cxx20 void _VerifyRange(
	const _Iterator_& _FirstIterator,
	const _Sentinel_& _LastIterator) noexcept
{
#if !defined(NDEBUG)
	if constexpr (std::is_pointer_v<_Iterator_> && std::is_pointer_v<_Sentinel_>) {
		DebugAssertLog(_FirstIterator <= _LastIterator, "transposed pointer range");
		return;
	}
	else if constexpr (type_traits::is_range_verifiable_v<_Iterator_, _Sentinel_>) {
		_VerifyRange(
			const_cast<const char*>(reinterpret_cast<const volatile char*>(std::to_address(_FirstIterator))),
			const_cast<const char*>(reinterpret_cast<const volatile char*>(std::to_address(_LastIterator))));
	}
#endif
}

#if !defined(__verifyRange)
#  if !defined(NDEBUG)
#    define __verifyRange(_First, _Last)  _VerifyRange(_First, _Last) 
#  else
#    define __verifyRange(_First, _Last) 
#  endif // !defined(NDEBUG)
#endif // !defined(__verifyRange)

__SIMD_STL_ALGORITHM_NAMESPACE_END
