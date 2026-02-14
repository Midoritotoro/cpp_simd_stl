#pragma once 

#include <src/simd_stl/datapar/arithmetic/Sub.h>
#include <src/simd_stl/datapar/shuffle/BroadcastZeros.h>


__SIMD_STL_DATAPAR_NAMESPACE_BEGIN 

template <arch::ISA	_ISA_>
struct _Simd_streaming_fence {
	simd_stl_static_operator simd_stl_always_inline void operator()() simd_stl_const_operator noexcept {
#if defined(simd_stl_processor_x86_64)
		_mm_sfence();
#endif // defined(simd_stl_processor_x86)
	}
};

__SIMD_STL_DATAPAR_NAMESPACE_BEGIN
