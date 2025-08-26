#pragma once 


#if !defined(simd_stl_has_cxx11)
#  if __cplusplus >= 201103L
#    define simd_stl_has_cxx11 1
#  else
#    define simd_stl_has_cxx11 0
#  endif // __cplusplus >= 201103L
#endif // !defined(simd_stl_has_cxx11)


#if !defined(simd_stl_has_cxx14)
#  if __cplusplus >= 201402L
#    define simd_stl_has_cxx14 1
#  else
#    define simd_stl_has_cxx14 0
#  endif // __cplusplus >= 201402L
#endif // !defined(simd_stl_has_cxx14)


#if !defined(simd_stl_has_cxx17)
#  if __cplusplus >= 201703L
#    define simd_stl_has_cxx17 1
#  else
#    define simd_stl_has_cxx17 0
#  endif // __cplusplus >= 201703L
#endif // !defined(simd_stl_has_cxx17)


#if !defined(simd_stl_has_cxx20)
#  if simd_stl_has_cxx17 && __cplusplus >= 202002L
#    define simd_stl_has_cxx20 1
#  else
#    define simd_stl_has_cxx20 0
#  endif // simd_stl_has_cxx17 && __cplusplus >= 202002L
#endif // !defined(simd_stl_has_cxx20)


#if !defined(simd_stl_has_cxx23)
#  if simd_stl_has_cxx20 && __cplusplus > 202002L
#    define simd_stl_has_cxx23 1
#  else
#    define simd_stl_has_cxx23 0
#  endif // simd_stl_has_cxx20 && __cplusplus > 202002L
#endif // !defined(simd_stl_has_cxx23)