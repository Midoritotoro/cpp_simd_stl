#pragma once 

#include <simd_stl/compatibility/CompilerDetection.h>
#include <simd_stl/compatibility/LanguageFeatures.h>


#if !defined(simd_stl_likely)
#  if defined(simd_stl_cpp_gnu) || defined(simd_stl_cpp_clang)
#    define simd_stl_likely(expression) __builtin_expect(!!(expression), true)
#  elif defined(simd_stl_cpp_msvc) && defined(__has_cpp_attribute) && __has_cpp_attribute(likely) >= 201803L
#    define simd_stl_likely(expression)                     \
       (                                                \
         ([](bool value){                               \
           switch (value) {                             \
             [[unlikely]] case true: return true;       \
             [[likely]] case false: return false;       \
         }                                              \
       })(expression))
#  else
#    define simd_stl_likely(expression) (!!(expression))
#  endif // defined(simd_stl_cpp_gnu) || defined(simd_stl_cpp_clang)
#endif // !defined(simd_stl_likely)


#if !defined(simd_stl_unlikely)
#  if defined(simd_stl_cpp_gnu) || defined(simd_stl_cpp_clang)
#    define simd_stl_unlikely(expression) __builtin_expect(!!(expression), false)
#  elif defined(simd_stl_cpp_msvc) && defined(__has_cpp_attribute) && __has_cpp_attribute(unlikely) >= 201803L
#    define simd_stl_unlikely(expression)                   \
       (                                                \
         ([](bool value){                               \
           switch (value) {                             \
             [[likely]] case true: return true;         \
             [[unlikely]] case false: return false;     \
         }                                              \
       })(expression))
#  else
#    define simd_stl_unlikely(expression) (!!(expression))
#  endif // defined(simd_stl_cpp_gnu) || defined(simd_stl_cpp_clang)
#endif // !defined(simd_stl_unlikely)


#if !defined(simd_stl_likely_attribute)
#  if defined(__has_cpp_attribute) && __has_cpp_attribute(likely) >= 201803L
#    define simd_stl_likely_attribute [[likely]]
#  else
#    define simd_stl_likely_attribute
#  endif
#endif // !defined(simd_stl_likely_attribute)


#if !defined(simd_stl_unlikely_attribute)
#  if defined(__has_cpp_attribute) && __has_cpp_attribute(unlikely) >= 201803L
#    define simd_stl_unlikely_attribute [[unlikely]]
#  else
#    define simd_stl_unlikely_attribute
#  endif
#endif // !defined(simd_stl_unlikely_attribute)
