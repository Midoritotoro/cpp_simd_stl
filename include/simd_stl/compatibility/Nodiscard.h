#pragma once 

#include <simd_stl/compatibility/CompilerDetection.h>
#include <simd_stl/compatibility/LanguageFeatures.h>


#if !defined(simd_stl_has_nodiscard)
#  if !defined(__has_cpp_attribute)
#    define simd_stl_has_nodiscard 0
#  elif __has_cpp_attribute(nodiscard) >= 201603L
#    define simd_stl_has_nodiscard 1
#  else
#    define simd_stl_has_nodiscard 0
#  endif
#endif // !defined(simd_stl_has_nodiscard)


#if !defined(simd_stl_nodiscard)
#  if simd_stl_has_nodiscard
#    define simd_stl_nodiscard  [[nodiscard]]
#  elif defined(simd_stl_cpp_gnu)
#    define simd_stl_nodiscard  __attribute__((__warn_unused_result__))
#  elif defined(simd_stl_cpp_clang)
#    define simd_stl_nodiscard  __attribute__(warn_unused_result)
#  elif defined(simd_stl_cpp_msvc)
#    define simd_stl_nodiscard  _Check_return_
#  else
#    define simd_stl_nodiscard
#  endif // simd_stl_has_nodiscard || defined(simd_stl_cpp_gnu) || defined(simd_stl_cpp_clang) || defined(simd_stl_cpp_msvc)
#endif // !defined(simd_stl_nodiscard)


#if !defined(simd_stl_nodiscard_with_warning)
#  if defined(__has_cpp_attribute) && __has_cpp_attribute(nodiscard) >= 201907L
#    define simd_stl_nodiscard_with_warning(message)    [[nodiscard(message)]]
#  elif defined(__has_cpp_attribute) && __has_cpp_attribute(nodiscard) >= 201603L
#    define simd_stl_nodiscard_with_warning(message)    simd_stl_nodiscard
#  else
#    define simd_stl_nodiscard_with_warning(message)
#  endif //     defined(__has_cpp_attribute) && __has_cpp_attribute(nodiscard) >= 201907L 
//  || defined(__has_cpp_attribute) && __has_cpp_attribute(nodiscard) >= 201603L
#endif // !defined(simd_stl_nodiscard_with_warning)


#if !defined(simd_stl_nodiscard_constructor)
// https://open-std.org/jtc1/sc22/wg21/docs/papers/2019/p1771r1.pdf
#  if defined(__has_cpp_attribute) && __has_cpp_attribute(nodiscard) >= 201907L
#    define simd_stl_nodiscard_constructor simd_stl_nodiscard
#  else
#    define simd_stl_nodiscard_constructor
#  endif // defined(__has_cpp_attribute) && __has_cpp_attribute(nodiscard) >= 201907L
#endif // !defined(simd_stl_nodiscard_constructor)


#if !defined(simd_stl_nodiscard_constructor_with_warning)
#  if defined(__has_cpp_attribute) && __has_cpp_attribute(nodiscard) >= 201907L
#    define simd_stl_nodiscard_constructor_with_warning(message) simd_stl_nodiscard_with_warning(message)
#  else
#    define simd_stl_nodiscard_constructor_with_warning(message) 
#  endif // defined(__has_cpp_attribute) && __has_cpp_attribute(nodiscard) >= 201907L
#endif // !defined(simd_stl_nodiscard_constructor_with_warning)