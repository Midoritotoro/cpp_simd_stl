#pragma once

#include <simd_stl/compatibility/CompilerDetection.h>


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
