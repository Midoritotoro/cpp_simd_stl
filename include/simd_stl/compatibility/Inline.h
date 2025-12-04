#pragma once 

#include <simd_stl/compatibility/CxxVersionDetection.h>
#include <simd_stl/compatibility/CompilerDetection.h>


#if !defined(simd_stl_never_inline)
#  if defined(simd_stl_cpp_msvc) || defined(simd_stl_cpp_clang)
#     define simd_stl_never_inline __declspec(noinline)
#  elif defined(simd_stl_cpp_gnu) 
#    define simd_stl_never_inline __attribute__((noinline))
#  else 
#    define simd_stl_never_inline 
#  endif // defined(simd_stl_cpp_msvc) || defined(simd_stl_cpp_clang) || defined(simd_stl_cpp_gnu)
#endif // !defined(simd_stl_never_inline)


#if !defined(simd_stl_always_inline)
#  if defined(simd_stl_cpp_msvc)
#    define simd_stl_always_inline __forceinline
#  elif defined(simd_stl_cpp_gnu) || defined(simd_stl_cpp_clang)
#    define simd_stl_always_inline inline __attribute__((always_inline))
#  else 
#    define simd_stl_always_inline inline
#  endif // defined(simd_stl_cpp_msvc) || defined(simd_stl_cpp_clang) || defined(simd_stl_cpp_gnu)
#endif // !defined(simd_stl_always_inline)


#if !defined(simd_stl_constexpr_cxx20)
#  if simd_stl_has_cxx20
#    define simd_stl_constexpr_cxx20 constexpr
#  else
#    define simd_stl_constexpr_cxx20
#  endif // simd_stl_has_cxx20
#endif // !defined(simd_stl_constexpr_cxx20)


#if !defined(simd_stl_clang_constexpr_cxx20)
#  if defined(simd_stl_cpp_clang)
#    define simd_stl_clang_constexpr_cxx20 simd_stl_constexpr_cxx20
#  else
#    define simd_stl_clang_constexpr_cxx20 
#  endif // defined(simd_stl_cpp_clang)
#endif // !defined(simd_stl_clang_constexpr_cxx20)