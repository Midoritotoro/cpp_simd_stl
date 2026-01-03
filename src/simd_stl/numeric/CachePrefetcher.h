#pragma once

#include <simd_stl/compatibility/Compatibility.h>


__SIMD_STL_NUMERIC_NAMESPACE_BEGIN

enum class __prefetch_hint: int32 {
	NTA = _MM_HINT_NTA,
	T0  = _MM_HINT_T0,
	T1  = _MM_HINT_T1,
	T2  = _MM_HINT_T2
};

template <__prefetch_hint _Hint_>
class __cache_prefetcher {
public:
	template <class _Pointer_>
	simd_stl_always_inline void operator()(_Pointer_ __pointer) const noexcept {
		_mm_prefetch(const_cast<const char*>(reinterpret_cast<const volatile char*>(std::to_address(__pointer))), static_cast<int>(_Hint_));
	}
};

__SIMD_STL_NUMERIC_NAMESPACE_END