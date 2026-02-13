#pragma once 

#include <simd_stl/compatibility/CompilerDetection.h>

#if (defined(simd_stl_cpp_clang) && simd_stl_cpp_clang < 1600) || defined(simd_stl_cpp_gnu)

#  if !defined(simd_stl_static_operator)
#    define simd_stl_static_operator
#  endif // !defined(simd_stl_static_operator)

#  if !defined(simd_stl_const_operator)
#    define simd_stl_const_operator const
#  endif // !defined(simd_stl_const_operator)

#  if !defined(simd_stl_static_labmda)
#    define simd_stl_static_labmda
#  endif // !defined(simd_stl_static_labmda)

#else

#  if !defined(simd_stl_static_operator)
#    define simd_stl_static_operator static
#  endif // !defined(simd_stl_static_operator)

#  if !defined(simd_stl_const_operator)
#    define simd_stl_const_operator
#  endif // !defined(simd_stl_const_operator)

#  if !defined(simd_stl_static_labmda)
#    define simd_stl_static_labmda static
#  endif // !defined(simd_stl_static_labmda)

#endif // (defined(simd_stl_cpp_clang) && simd_stl_cpp_clang < 1600) || defined(simd_stl_cpp_gnu)