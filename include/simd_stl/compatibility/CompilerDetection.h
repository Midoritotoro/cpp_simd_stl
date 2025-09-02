#pragma once 

#if !defined(SIMD_STL_ECHO)
#  define SIMD_STL_ECHO(...) __VA_ARGS__
#endif

#if defined(_MSC_VER)
#  define simd_stl_cpp_msvc			(_MSC_VER)
#  define simd_stl_cpp_msvc_only	simd_stl_cpp_msvc

#  if defined(__clang__)
#    undef simd_stl_cpp_msvc_only
#    define simd_stl_cpp_clang ((__clang_major__ * 100) + __clang_minor__)
#    define simd_stl_cpp_clang_only simd_stl_cpp_clang
#  endif // defined(__clang__)
#

#elif defined(__GNUC__)
#  define simd_stl_cpp_gnu          (__GNUC__ * 100 + __GNUC_MINOR__)

#  if defined(__MINGW32__)
#    define simd_stl_cpp_mingw
#  endif

#  if defined(__clang__)
#    if defined(__apple_build_version__)

#      if __apple_build_version__   >= 14030022 // Xcode 14.3
#        define simd_stl_cpp_clang 1500

#      elif __apple_build_version__ >= 14000029 // Xcode 14.0
#        define simd_stl_cpp_clang 1400

#      elif __apple_build_version__ >= 13160021 // Xcode 13.3
#        define simd_stl_cpp_clang 1300

#      elif __apple_build_version__ >= 13000029 // Xcode 13.0
#        define simd_stl_cpp_clang 1200

#      elif __apple_build_version__ >= 12050022 // Xcode 12.5
#        define simd_stl_cpp_clang 1110

#      elif __apple_build_version__ >= 12000032 // Xcode 12.0
#        define simd_stl_cpp_clang 1000

#      elif __apple_build_version__ >= 11030032 // Xcode 11.4
#        define simd_stl_cpp_clang 900

#      elif __apple_build_version__ >= 11000033 // Xcode 11.0
#        define simd_stl_cpp_clang 800

#      else
#        error ""
#      endif // __apple_build_version__

#    else
#      define simd_stl_cpp_clang ((__clang_major__ * 100) + __clang_minor__)
#    endif // defined(__apple_build_version__)

#    define simd_stl_cpp_clang_only simd_stl_cpp_clang


#    if !defined(__has_extension)
#      define __has_extension __has_feature
#    endif // !defined(__has_extension)
#  else

#    define simd_stl_cpp_gnu_only simd_stl_cpp_gnu
#  endif // defined(__clang__)

#endif // defined(_MSC_VER) || defined(__GNUC__)