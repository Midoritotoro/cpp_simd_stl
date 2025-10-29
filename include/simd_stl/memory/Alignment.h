#pragma once 

#include <simd_stl/Types.h>

#include <src/simd_stl/algorithm/MsvcIteratorUnwrap.h>
#include <simd_stl/compatibility/SystemDetection.h>

#include <simd_stl/memory/PointerToIntegral.h>


__SIMD_STL_MEMORY_NAMESPACE_BEGIN

#if defined(simd_stl_cpp_gnu)
#  if simd_stl_cpp_gnu > 430
#    define ALLOC_SIZE(...) __attribute__((alloc_size(__VA_ARGS__)))
#  else
#    define ALLOC_SIZE(...)
#  endif
#else 
#  define ALLOC_SIZE(...) 
#endif

#if defined(simd_stl_os_windows) && !defined(aligned_malloc)
#  define aligned_malloc		                _aligned_malloc
#elif !defined(aligned_malloc)
#  define aligned_malloc(size, alignment)		malloc(size)
#endif

#if defined(simd_stl_os_windows) && !defined(aligned_realloc)
#  define aligned_realloc                       _aligned_realloc
#elif !defined(aligned_realloc)
#  define aligned_realloc(block, size, align)   realloc(block, size)
#endif

#if defined (simd_stl_os_windows) && !defined(aligned_free)
#  define aligned_free(pointer)                 _aligned_free(pointer)
#elif !defined(aligned_free)
#  define aligned_free(pointer)                 free(pointer)
#endif

simd_stl_always_inline bool isAlignment(std::size_t value) noexcept {
    return (value > 0) && ((value & (value - 1)) == 0);
}

simd_stl_always_inline bool isAligned(
    const void* pointer,
    sizetype    alignment) noexcept
{
    DebugAssert(isAlignment(alignment));
    return (pointerToIntegral(pointer) & (alignment - 1)) == 0;
}

simd_stl_always_inline void* alignDown(
    void*       pointer, 
    sizetype    alignment) noexcept
{
    DebugAssert(isAlignment(alignment));
    return reinterpret_cast<void*>(~(alignment - 1) & pointerToIntegral(pointer));

}

simd_stl_always_inline void* alignUp(
    void*       pointer,
    sizetype    alignment) noexcept
{
    DebugAssert(isAlignment(alignment));
    return reinterpret_cast<void*>(~(alignment - 1) & (pointerToIntegral(pointer) + alignment - 1));
}

template <std::size_t N>
struct IsAlignmentConstant : 
    std::integral_constant<bool, (N > 0) && ((N & (N - 1)) == 0)> 
{};

__SIMD_STL_MEMORY_NAMESPACE_END
