#pragma once 

#include <simd_stl/compatibility/Compatibility.h>
#include <simd_stl/arch/ProcessorDetection.h>


__SIMD_STL_CONCURRENCY_NAMESPACE_BEGIN

simd_stl_always_inline void _Yield() noexcept {
#if __has_builtin(__yield)
        __yield();

#elif defined(_YIELD_PROCESSOR) && defined(simd_stl_cpp_msvc)
        _YIELD_PROCESSOR();

#elif __has_builtin(__builtin_ia32_pause)
        __builtin_ia32_pause();

#elif defined(simd_stl_processor_x86) && defined(simd_stl_cpp_gnu)
        __builtin_ia32_pause();

#elif defined(simd_stl_processor_x86) && defined(simd_stl_cpp_msvc)
        _mm_pause();

#elif defined(simd_stl_processor_x86)
        asm("pause");

#elif __has_builtin(__builtin_arm_yield)
        __builtin_arm_yield();

#elif defined(simd_stl_processor_arm) && simd_stl_processor_arm >= 7 && defined(simd_stl_cpp_gnu)
        asm("yield");        
#endif
}

__SIMD_STL_CONCURRENCY_NAMESPACE_END
