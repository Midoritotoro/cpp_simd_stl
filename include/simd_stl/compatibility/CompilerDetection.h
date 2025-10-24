#pragma once 

#if !defined(SIMD_STL_ECHO)
#  define SIMD_STL_ECHO(...) __VA_ARGS__
#endif


#define _PP_CAT(a,b) a##b
#define PP_CAT(a,b) _PP_CAT(a,b)

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

#define __SIMD_STL_REPEAT__(X) X

#define __SIMD_STL_REPEAT_1(X)  __SIMD_STL_REPEAT__(X)
#define __SIMD_STL_REPEAT_2(X)  __SIMD_STL_REPEAT_1(X), __SIMD_STL_REPEAT__(X)
#define __SIMD_STL_REPEAT_3(X)  __SIMD_STL_REPEAT_2(X), __SIMD_STL_REPEAT__(X)
#define __SIMD_STL_REPEAT_4(X)  __SIMD_STL_REPEAT_3(X), __SIMD_STL_REPEAT__(X)
#define __SIMD_STL_REPEAT_5(X)  __SIMD_STL_REPEAT_4(X), __SIMD_STL_REPEAT__(X)
#define __SIMD_STL_REPEAT_6(X)  __SIMD_STL_REPEAT_5(X), __SIMD_STL_REPEAT__(X)
#define __SIMD_STL_REPEAT_7(X)  __SIMD_STL_REPEAT_6(X), __SIMD_STL_REPEAT__(X)
#define __SIMD_STL_REPEAT_8(X)  __SIMD_STL_REPEAT_7(X), __SIMD_STL_REPEAT__(X)
#define __SIMD_STL_REPEAT_9(X)  __SIMD_STL_REPEAT_8(X), __SIMD_STL_REPEAT__(X)
#define __SIMD_STL_REPEAT_10(X) __SIMD_STL_REPEAT_9(X), __SIMD_STL_REPEAT__(X)
#define __SIMD_STL_REPEAT_11(X) __SIMD_STL_REPEAT_10(X), __SIMD_STL_REPEAT__(X)
#define __SIMD_STL_REPEAT_12(X) __SIMD_STL_REPEAT_11(X), __SIMD_STL_REPEAT__(X)
#define __SIMD_STL_REPEAT_13(X) __SIMD_STL_REPEAT_12(X), __SIMD_STL_REPEAT__(X)
#define __SIMD_STL_REPEAT_14(X) __SIMD_STL_REPEAT_13(X), __SIMD_STL_REPEAT__(X)
#define __SIMD_STL_REPEAT_15(X) __SIMD_STL_REPEAT_14(X), __SIMD_STL_REPEAT__(X)
#define __SIMD_STL_REPEAT_16(X) __SIMD_STL_REPEAT_15(X), __SIMD_STL_REPEAT__(X)
#define __SIMD_STL_REPEAT_17(X) __SIMD_STL_REPEAT_16(X), __SIMD_STL_REPEAT__(X)
#define __SIMD_STL_REPEAT_18(X) __SIMD_STL_REPEAT_17(X), __SIMD_STL_REPEAT__(X)
#define __SIMD_STL_REPEAT_19(X) __SIMD_STL_REPEAT_18(X), __SIMD_STL_REPEAT__(X)
#define __SIMD_STL_REPEAT_20(X) __SIMD_STL_REPEAT_19(X), __SIMD_STL_REPEAT__(X)
#define __SIMD_STL_REPEAT_21(X) __SIMD_STL_REPEAT_20(X), __SIMD_STL_REPEAT__(X)
#define __SIMD_STL_REPEAT_22(X) __SIMD_STL_REPEAT_21(X), __SIMD_STL_REPEAT__(X)
#define __SIMD_STL_REPEAT_23(X) __SIMD_STL_REPEAT_22(X), __SIMD_STL_REPEAT__(X)
#define __SIMD_STL_REPEAT_24(X) __SIMD_STL_REPEAT_23(X), __SIMD_STL_REPEAT__(X)
#define __SIMD_STL_REPEAT_25(X) __SIMD_STL_REPEAT_24(X), __SIMD_STL_REPEAT__(X)
#define __SIMD_STL_REPEAT_26(X) __SIMD_STL_REPEAT_25(X), __SIMD_STL_REPEAT__(X)
#define __SIMD_STL_REPEAT_27(X) __SIMD_STL_REPEAT_26(X), __SIMD_STL_REPEAT__(X)
#define __SIMD_STL_REPEAT_28(X) __SIMD_STL_REPEAT_27(X), __SIMD_STL_REPEAT__(X)
#define __SIMD_STL_REPEAT_29(X) __SIMD_STL_REPEAT_28(X), __SIMD_STL_REPEAT__(X)
#define __SIMD_STL_REPEAT_30(X) __SIMD_STL_REPEAT_29(X), __SIMD_STL_REPEAT__(X)
#define __SIMD_STL_REPEAT_31(X) __SIMD_STL_REPEAT_30(X), __SIMD_STL_REPEAT__(X)
#define __SIMD_STL_REPEAT_32(X) __SIMD_STL_REPEAT_31(X), __SIMD_STL_REPEAT__(X)
#define __SIMD_STL_REPEAT_33(X) __SIMD_STL_REPEAT_32(X), __SIMD_STL_REPEAT__(X)
#define __SIMD_STL_REPEAT_34(X) __SIMD_STL_REPEAT_33(X), __SIMD_STL_REPEAT__(X)
#define __SIMD_STL_REPEAT_35(X) __SIMD_STL_REPEAT_34(X), __SIMD_STL_REPEAT__(X)
#define __SIMD_STL_REPEAT_36(X) __SIMD_STL_REPEAT_35(X), __SIMD_STL_REPEAT__(X)
#define __SIMD_STL_REPEAT_37(X) __SIMD_STL_REPEAT_36(X), __SIMD_STL_REPEAT__(X)
#define __SIMD_STL_REPEAT_38(X) __SIMD_STL_REPEAT_37(X), __SIMD_STL_REPEAT__(X)
#define __SIMD_STL_REPEAT_39(X) __SIMD_STL_REPEAT_38(X), __SIMD_STL_REPEAT__(X)
#define __SIMD_STL_REPEAT_40(X) __SIMD_STL_REPEAT_39(X), __SIMD_STL_REPEAT__(X)
#define __SIMD_STL_REPEAT_41(X) __SIMD_STL_REPEAT_40(X), __SIMD_STL_REPEAT__(X)
#define __SIMD_STL_REPEAT_42(X) __SIMD_STL_REPEAT_41(X), __SIMD_STL_REPEAT__(X)
#define __SIMD_STL_REPEAT_43(X) __SIMD_STL_REPEAT_42(X), __SIMD_STL_REPEAT__(X)
#define __SIMD_STL_REPEAT_44(X) __SIMD_STL_REPEAT_43(X), __SIMD_STL_REPEAT__(X)
#define __SIMD_STL_REPEAT_45(X) __SIMD_STL_REPEAT_44(X), __SIMD_STL_REPEAT__(X)
#define __SIMD_STL_REPEAT_46(X) __SIMD_STL_REPEAT_45(X), __SIMD_STL_REPEAT__(X)
#define __SIMD_STL_REPEAT_47(X) __SIMD_STL_REPEAT_46(X), __SIMD_STL_REPEAT__(X)
#define __SIMD_STL_REPEAT_48(X) __SIMD_STL_REPEAT_47(X), __SIMD_STL_REPEAT__(X)
#define __SIMD_STL_REPEAT_49(X) __SIMD_STL_REPEAT_48(X), __SIMD_STL_REPEAT__(X)
#define __SIMD_STL_REPEAT_50(X) __SIMD_STL_REPEAT_49(X), __SIMD_STL_REPEAT__(X)
#define __SIMD_STL_REPEAT_51(X) __SIMD_STL_REPEAT_50(X), __SIMD_STL_REPEAT__(X)
#define __SIMD_STL_REPEAT_52(X) __SIMD_STL_REPEAT_51(X), __SIMD_STL_REPEAT__(X)
#define __SIMD_STL_REPEAT_53(X) __SIMD_STL_REPEAT_52(X), __SIMD_STL_REPEAT__(X)
#define __SIMD_STL_REPEAT_54(X) __SIMD_STL_REPEAT_53(X), __SIMD_STL_REPEAT__(X)
#define __SIMD_STL_REPEAT_55(X) __SIMD_STL_REPEAT_54(X), __SIMD_STL_REPEAT__(X)
#define __SIMD_STL_REPEAT_56(X) __SIMD_STL_REPEAT_55(X), __SIMD_STL_REPEAT__(X)
#define __SIMD_STL_REPEAT_57(X) __SIMD_STL_REPEAT_56(X), __SIMD_STL_REPEAT__(X)
#define __SIMD_STL_REPEAT_58(X) __SIMD_STL_REPEAT_57(X), __SIMD_STL_REPEAT__(X)
#define __SIMD_STL_REPEAT_59(X) __SIMD_STL_REPEAT_58(X), __SIMD_STL_REPEAT__(X)
#define __SIMD_STL_REPEAT_60(X) __SIMD_STL_REPEAT_59(X), __SIMD_STL_REPEAT__(X)
#define __SIMD_STL_REPEAT_61(X) __SIMD_STL_REPEAT_60(X), __SIMD_STL_REPEAT__(X)
#define __SIMD_STL_REPEAT_62(X) __SIMD_STL_REPEAT_61(X), __SIMD_STL_REPEAT__(X)
#define __SIMD_STL_REPEAT_63(X) __SIMD_STL_REPEAT_62(X), __SIMD_STL_REPEAT__(X)
#define __SIMD_STL_REPEAT_64(X) __SIMD_STL_REPEAT_63(X), __SIMD_STL_REPEAT__(X)

#define __SIMD_STL_REPEAT_N(N, X) PP_CAT(__SIMD_STL_REPEAT_, N)(X)