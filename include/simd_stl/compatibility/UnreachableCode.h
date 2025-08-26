#pragma once 

#include <base/core/compatibility/CompilerDetection.h>


#if !defined(simd_stl_maybe_unused_attribute)
#  if defined(__has_cpp_attribute) && __has_cpp_attribute(maybe_unused) >= 201603L
#    define simd_stl_maybe_unused_attribute     [[maybe_unused]]
#  elif defined(simd_stl_cpp_gnu) || defined(simd_stl_cpp_clang)
#    define simd_stl_maybe_unused_attribute     __attribute__(unused)
#  endif // defined(__has_cpp_attribute) && __has_cpp_attribute(maybe_unused) >= 201603L
#endif // !defined(simd_stl_maybe_unused_attribute)


#if !defined(simd_stl_unused)
#  define simd_stl_unused(variable) ((void)(variable))
#endif // !defined(simd_stl_unused)

#if !defined(simd_stl_unreachable)
#  if defined(simd_stl_cpp_gnu) || defined(simd_stl_cpp_clang)
#    define simd_stl_unreachable() __builtin_unreachable()
#  elif defined(simd_stl_cpp_msvc)
#    define simd_stl_unreachable() (__assume(0))
#  else
#    define simd_stl_unreachable() ((void)0)
#  endif // defined(simd_stl_cpp_gnu) || defined(simd_stl_cpp_clang) || defined(simd_stl_cpp_msvc)
#endif // !defined(simd_stl_unreachable)

