#pragma once 

#include <base/core/compatibility/CompilerDetection.h>
#include <base/core/compatibility/Nodiscard.h>


#if !defined(simd_stl_restrict)
#  if defined(simd_stl_cpp_msvc) || defined (simd_stl_cpp_clang)
#    define simd_stl_restrict   __declspec(restrict)
#  elif defined(simd_stl_cpp_gnu)
#    define simd_stl_restrict   __restrict  
#  endif // defined(simd_stl_cpp_msvc) || defined (simd_stl_cpp_clang) || defined(simd_stl_cpp_gnu)
#endif // !defined(simd_stl_restrict)

#if !defined(simd_stl_sizeof_in_bits)
#  define simd_stl_sizeof_in_bits(type) (sizeof(type) * 8)
#endif // !defined(simd_stl_sizeof_in_bits)