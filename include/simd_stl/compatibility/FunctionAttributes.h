#pragma once

#include <base/core/compatibility/CompilerDetection.h>


#if defined(simd_stl_cpp_msvc) || defined(simd_stl_cpp_clang)
#  include <stdnoreturn.h> 
#endif // defined(simd_stl_cpp_msvc) || defined(simd_stl_cpp_clang)


#if defined(__cpp_conditional_explicit)
#  if !defined(simd_stl_implicit)
#    define simd_stl_implicit explicit(false)
#  else
#    define simd_stl_implicit
#  endif // simd_stl_implicit
#endif // defined(__cpp_conditional_explicit)


#if !defined(simd_stl_noreturn)
#  if defined(simd_stl_cpp_gnu)
#    define simd_stl_noreturn           __attribute__((__noreturn__))
#  elif defined(simd_stl_cpp_msvc) || defined(simd_stl_cpp_clang)
#    define simd_stl_noreturn           __declspec(noreturn)
#  else 
#    define simd_stl_noreturn       
#  endif // defined(simd_stl_cpp_gnu) || defined(simd_stl_cpp_msvc) || defined(simd_stl_cpp_clang)
#endif // !defined(simd_stl_noreturn)


#if !defined(simd_stl_declare_pure_function)
#  if defined(simd_stl_cpp_gnu) || defined(simd_stl_cpp_clang)
#    define simd_stl_declare_pure_function __attribute__((pure))
#  else
#    define simd_stl_declare_pure_function
#  endif // defined(simd_stl_cpp_gnu) || defined(simd_stl_cpp_clang)
#endif // !defined(simd_stl_declare_pure_function)


#if !defined(simd_stl_declare_const_function)
#  if defined(simd_stl_cpp_msvc)
// The "noalias" attribute tells the compiler optimizer that pointers going into these hand-vectorized algorithms
// won't be stored beyond the lifetime of the function, and that the function will only reference arrays denoted by
// those pointers. The optimizer also assumes in that case that a pointer parameter is not returned to the caller via
// the return value, so functions using "noalias" must usually return void. This attribute is valuable because these
// functions are in native code objects that the compiler cannot analyze. In the absence of the noalias attribute, the
// compiler has to assume that the denoted arrays are "globally address taken", and that any later calls to
// unanalyzable routines may modify those arrays.
#    define simd_stl_declare_const_function __declspec(noalias)
#  elif defined(simd_stl_cpp_gnu) || defined(simd_stl_cpp_clang)
#    define simd_stl_declare_const_function __attribute__((const))
#  else
#    define simd_stl_declare_const_function
#  endif // defined(simd_stl_cpp_gnu) || defined(simd_stl_cpp_clang) || defined(simd_stl_cpp_msvc)
#endif // !defined(simd_stl_declare_const_function)


#if !defined(simd_stl_declare_cold_function)
#  if defined(simd_stl_cpp_gnu) || defined(simd_stl_cpp_clang)
#    define simd_stl_declare_cold_function __attribute__((cold))
#  else 
#    define simd_stl_declare_cold_function
#  endif // defined(simd_stl_cpp_gnu) || defined(simd_stl_cpp_clang)
#endif // !defined(simd_stl_declare_cold_function)
