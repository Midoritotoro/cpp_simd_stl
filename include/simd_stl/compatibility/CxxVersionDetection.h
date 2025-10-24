#pragma once 

#if defined(__cplusplus)
    #if defined(_MSVC_LANG) && _MSVC_LANG > __cplusplus
        #define LANGUAGE_VERSION _MSVC_LANG
    #else
        #define LANGUAGE_VERSION __cplusplus
    #endif
#else
    #define LANGUAGE_VERSION 0L
#endif


#if !defined(simd_stl_has_cxx11)
#  if LANGUAGE_VERSION >= 201103L
#    define simd_stl_has_cxx11 1
#  else
#    define simd_stl_has_cxx11 0
#  endif // LANGUAGE_VERSION >= 201103L
#endif // !defined(simd_stl_has_cxx11)


#if !defined(simd_stl_has_cxx14)
#  if LANGUAGE_VERSION >= 201402L
#    define simd_stl_has_cxx14 1
#  else
#    define simd_stl_has_cxx14 0
#  endif // LANGUAGE_VERSION >= 201402L
#endif // !defined(simd_stl_has_cxx14)


#if !defined(simd_stl_has_cxx17)
#  if LANGUAGE_VERSION >= 201703L
#    define simd_stl_has_cxx17 1
#  else
#    define simd_stl_has_cxx17 0
#  endif // LANGUAGE_VERSION >= 201703L
#endif // !defined(simd_stl_has_cxx17)


#if !defined(simd_stl_has_cxx20)
#  if simd_stl_has_cxx17 && LANGUAGE_VERSION >= 202002L
#    define simd_stl_has_cxx20 1
#  else
#    define simd_stl_has_cxx20 0
#  endif // simd_stl_has_cxx17 && LANGUAGE_VERSION >= 202002L
#endif // !defined(simd_stl_has_cxx20)


#if !defined(simd_stl_has_cxx23)
#  if simd_stl_has_cxx20 && LANGUAGE_VERSION > 202002L
#    define simd_stl_has_cxx23 1
#  else
#    define simd_stl_has_cxx23 0
#  endif // simd_stl_has_cxx20 && LANGUAGE_VERSION > 202002L
#endif // !defined(simd_stl_has_cxx23)