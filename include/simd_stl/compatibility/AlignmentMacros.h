#pragma once 

#include <simd_stl/compatibility/CompilerDetection.h>
#include <simd_stl/arch/ProcessorDetection.h>


#if !defined(simd_stl_unaligned)
#  if defined(simd_stl_processor_x86_64) || defined(simd_stl_processor_arm) || (simd_stl_processor_arm == 8) // x64 ARM
#    if defined(simd_stl_os_windows) && defined(simd_stl_cpp_msvc)
#      define simd_stl_unaligned __unaligned
#    else
#      define simd_stl_unaligned
#    endif // defined(simd_stl_os_windows) && defined(simd_stl_cpp_msvc)
#  else 
#    define simd_stl_unaligned
#  endif // defined(simd_stl_processor_x86_64) || defined(simd_stl_processor_arm) || (simd_stl_processor_arm == 8)
#endif // !defined(simd_stl_unaligned)


#if !defined(simd_stl_aligned_type)
#  if defined(simd_stl_cpp_gnu)
#    define simd_stl_aligned_type(alignment) __attribute__((aligned(alignment)))
#  elif defined(simd_stl_cpp_msvc) || defined(simd_stl_cpp_clang) 
#    define simd_stl_aligned_type(alignment) __declspec(align(alignment))
#  elif simd_stl_has_cxx11
#    define simd_stl_aligned_type(alignment) alignas(alignment)
#  endif // defined(simd_stl_cpp_gnu) || defined(simd_stl_cpp_msvc) || defined(simd_stl_cpp_clang) || simd_stl_has_cxx11
#endif // !defined(simd_stl_aligned_type)