#pragma once 

#include <simd_stl/compatibility/Compatibility.h>

#if defined(simd_stl_processor_x86) && defined(simd_stl_cpp_msvc)

#  if (defined(_M_X64) || _M_IX86_FP >= 2)
#    define __SSE__ 1
#    define __SSE2__ 1
#  endif // (defined(_M_X64) || _M_IX86_FP >= 2)

#  if (defined(_M_AVX) || defined(__AVX__))
#    define __SSE3__                        1

#    define __SSSE3__                       1
#    define __SSE4_1__                      1

#    define __SSE4_2__                      1
#    define __POPCNT__                      1

#    if !defined(__AVX__)
#      define __AVX__                       1
#    endif // !defined(__AVX__)

#  endif // (defined(_M_AVX) || defined(__AVX__))
#  ifdef __AVX2__
// MSVC defines __AVX2__ with /arch:AVX2
#    define __F16C__                        1
#    define __RDRND__                       1

#    define __FMA__                         1

#    define __BMI__                         1
#    define __BMI2__                        1

#    define __MOVBE__                       1
#    define __LZCNT__                       1
#  endif // __AVX2__
# endif // defined(simd_stl_processor_x86) && defined(simd_stl_cpp_msvc)

#if !defined(SIMD_STL_HAS_SSE2_SUPPORT) 
#  if defined(__SSE2__)
#    define SIMD_STL_HAS_SSE2_SUPPORT 1
#  else
#    define SIMD_STL_HAS_SSE2_SUPPORT 0
#  endif // defined(__SSE2__)
#endif // !defined(SIMD_STL_HAS_SSE2_SUPPORT)

#if !defined(SIMD_STL_HAS_SSE3_SUPPORT) 
#  if defined(__SSE3__)
#    define SIMD_STL_HAS_SSE3_SUPPORT 1
#  else
#    define SIMD_STL_HAS_SSE3_SUPPORT 0
#  endif // defined(__SSE3__)
#endif // !defined(SIMD_STL_HAS_SSE3_SUPPORT)

#if !defined(SIMD_STL_HAS_SSSE3_SUPPORT) 
#  if defined(__SSSE3__)
#    define SIMD_STL_HAS_SSSE3_SUPPORT 1
#  else
#    define SIMD_STL_HAS_SSSE3_SUPPORT 0
#  endif // defined(__SSSE3__)
#endif // !defined(SIMD_STL_HAS_SSSE3_SUPPORT)

#if !defined(SIMD_STL_HAS_SSE41_SUPPORT) 
#  if defined (__SSE4_1__)
#    define SIMD_STL_HAS_SSE41_SUPPORT 1
#  else 
#    define SIMD_STL_HAS_SSE41_SUPPORT 0
#  endif // defined(__SSE4_1__)
#endif // !defined(SIMD_STL_HAS_SSE41_SUPPORT)

#if !defined(SIMD_STL_HAS_SSE42_SUPPORT) 
#  if defined (__SSE4_2__)
#    define SIMD_STL_HAS_SSE42_SUPPORT 1
#  else
#    define SIMD_STL_HAS_SSE42_SUPPORT 0
#  endif // defined(__SSE4_2__)
#endif // !defined(SIMD_STL_HAS_SSE42_SUPPORT)

#if !defined(SIMD_STL_HAS_AVX_SUPPORT) 
#  if defined (__AVX__)
#    define SIMD_STL_HAS_AVX_SUPPORT 1
#  else
#    define SIMD_STL_HAS_AVX_SUPPORT 0
#  endif // defined(__AVX__)
#endif // !defined(SIMD_STL_HAS_AVX_SUPPORT)

#if !defined(SIMD_STL_HAS_AVX2_SUPPORT) 
#  if defined (__AVX2__)
#    define SIMD_STL_HAS_AVX2_SUPPORT 1
#  else 
#    define SIMD_STL_HAS_AVX2_SUPPORT 0
#  endif // defined(__AVX2__)
#endif // !defined(SIMD_STL_HAS_AVX2_SUPPORT)

#if !defined(SIMD_STL_HAS_AVX512F_SUPPORT) 
#  if defined (__AVX512F__)
#    define SIMD_STL_HAS_AVX512F_SUPPORT 1
#  else 
#    define SIMD_STL_HAS_AVX512F_SUPPORT 0
#  endif // defined(__AVX512F__)
#endif // !defined(SIMD_STL_HAS_AVX512F_SUPPORT)

#if !defined(SIMD_STL_HAS_AVX512BW_SUPPORT) 
#  if defined (__AVX512BW__)
#    define SIMD_STL_HAS_AVX512BW_SUPPORT 1
#  else 
#    define SIMD_STL_HAS_AVX512BW_SUPPORT 0
#  endif // defined(__AVX512BW__)
#endif // !defined(SIMD_STL_HAS_AVX512BW_SUPPORT)

#if !defined(SIMD_STL_HAS_AVX512CD_SUPPORT) 
#  if defined (__AVX512CD__)
#    define SIMD_STL_HAS_AVX512CD_SUPPORT 1
#  else 
#    define SIMD_STL_HAS_AVX512CD_SUPPORT 0
#  endif // defined(__AVX512CD__)
#endif // !defined(SIMD_STL_HAS_AVX512CD_SUPPORT)

#if !defined(SIMD_STL_HAS_AVX512DQ_SUPPORT) 
#  if defined (__AVX512DQ__)
#    define SIMD_STL_HAS_AVX512DQ_SUPPORT 1
#  else 
#    define SIMD_STL_HAS_AVX512DQ_SUPPORT 0
#  endif // defined(__AVX512DQ__)
#endif // !defined(SIMD_STL_HAS_AVX512DQ_SUPPORT)

#if !defined(SIMD_STL_HAS_AVX512VL_SUPPORT) 
#  if defined (__AVX512VL__)
#    define SIMD_STL_HAS_AVX512VL_SUPPORT 1
#  else 
#    define SIMD_STL_HAS_AVX512VL_SUPPORT 0
#  endif // defined(__AVX512VL__)
#endif // !defined(SIMD_STL_HAS_AVX512VL_SUPPORT)

#if !defined(SIMD_STL_HAS_AVX512VBMI_SUPPORT) 
#  if defined(simd_stl_cpp_msvc)
#    if SIMD_STL_HAS_AVX512F_SUPPORT
#      define SIMD_STL_HAS_AVX512VBMI_SUPPORT 1
#    else 
#      define SIMD_STL_HAS_AVX512VBMI_SUPPORT 0
#    endif // SIMD_STL_HAS_AVX512F_SUPPORT
#  endif // defined(simd_stl_cpp_msvc)
#  else
#    if defined (__AVX512VBMI__)
#      define SIMD_STL_HAS_AVX512VBMI_SUPPORT 1
#    else 
#      define SIMD_STL_HAS_AVX512VBMI_SUPPORT 0
#    endif // defined (__AVX512VBMI__)
#endif // !defined(SIMD_STL_HAS_AVX512VBMI_SUPPORT)

#if !defined(SIMD_STL_HAS_AVX512VBMI2_SUPPORT) 
#  if defined(simd_stl_cpp_msvc)
#    if SIMD_STL_HAS_AVX512F_SUPPORT
#      define SIMD_STL_HAS_AVX512VBMI2_SUPPORT 1
#    else 
#      define SIMD_STL_HAS_AVX512VBMI2_SUPPORT 0
#    endif // SIMD_STL_HAS_AVX512F_SUPPORT
#  endif // defined(simd_stl_cpp_msvc)
#  else
#    if defined (__AVX512VBMI2__)
#      define SIMD_STL_HAS_AVX512VBMI2_SUPPORT 1
#    else 
#      define SIMD_STL_HAS_AVX512VBMI2_SUPPORT 0
#    endif // defined (__AVX512VBMI2__)
#endif // !defined(SIMD_STL_HAS_AVX512VBMI2_SUPPORT)

#if defined(SIMD_STL_FORCE_AVX512F)
#  if !SIMD_STL_HAS_AVX512F_SUPPORT
#    undef SIMD_STL_FORCE_AVX512F
#  endif // !SIMD_STL_HAS_AVX512F_SUPPORT
# endif // defined(SIMD_STL_FORCE_AVX512F)

#if defined(SIMD_STL_FORCE_AVX512BW)
#  if !SIMD_STL_HAS_AVX512BW_SUPPORT
#    undef SIMD_STL_FORCE_AVX512BW
#  endif // !SIMD_STL_HAS_AVX512BW_SUPPORT
# endif // defined(SIMD_STL_FORCE_AVX512BW)

#if defined(SIMD_STL_FORCE_AVX512CD)
#  if !SIMD_STL_HAS_AVX512CD_SUPPORT
#    undef SIMD_STL_FORCE_AVX512CD
#  endif // !SIMD_STL_HAS_AVX512CD_SUPPORT
# endif // defined(SIMD_STL_FORCE_AVX512CD)

#if defined(SIMD_STL_FORCE_AVX512DQ)
#  if !SIMD_STL_HAS_AVX512DQ_SUPPORT
#    undef SIMD_STL_FORCE_AVX512DQ
#  endif // !SIMD_STL_HAS_AVX512DQ_SUPPORT
# endif // defined(SIMD_STL_FORCE_AVX512DQ)

#if defined(SIMD_STL_FORCE_AVX512VL)
#  if !SIMD_STL_HAS_AVX512VL_SUPPORT
#    undef SIMD_STL_FORCE_AVX512VL
#  endif // !SIMD_STL_HAS_AVX512VL_SUPPORT
# endif // defined(SIMD_STL_FORCE_AVX512VL)

#if defined(SIMD_STL_FORCE_AVX512VBMI)
#  if !SIMD_STL_HAS_AVX512VBMI_SUPPORT
#    undef SIMD_STL_FORCE_AVX512VBMI
#  endif // !SIMD_STL_HAS_AVX512VBMI_SUPPORT
# endif // defined(SIMD_STL_FORCE_AVX512VBMI)

#if defined(SIMD_STL_FORCE_AVX512VBMI2)
#  if !SIMD_STL_HAS_AVX512VBMI2_SUPPORT
#    undef SIMD_STL_FORCE_AVX512VBMI2
#  endif // !SIMD_STL_HAS_AVX512VBMI2_SUPPORT
# endif // defined(SIMD_STL_FORCE_AVX512VBMI2)

#if defined(SIMD_STL_FORCE_AVX2)
#  if !SIMD_STL_HAS_AVX2_SUPPORT
#    undef SIMD_STL_FORCE_AVX2
#  endif // !SIMD_STL_HAS_AVX2_SUPPORT
# endif // defined(SIMD_STL_FORCE_AVX2)

#if defined(SIMD_STL_FORCE_AVX)
#  if !SIMD_STL_HAS_AVX_SUPPORT
#    undef SIMD_STL_FORCE_AVX
#  endif // !SIMD_STL_HAS_AVX_SUPPORT
# endif // defined(SIMD_STL_FORCE_AVX)

#if defined(SIMD_STL_FORCE_SSE42)
#  if !SIMD_STL_HAS_SSE42_SUPPORT
#    undef SIMD_STL_FORCE_SSE42
#  endif // !SIMD_STL_HAS_SSE42_SUPPORT
# endif // defined(SIMD_STL_FORCE_SSE42)

#if defined(SIMD_STL_FORCE_SSE41)
#  if !SIMD_STL_HAS_SSE41_SUPPORT
#    undef SIMD_STL_FORCE_SSE41
#  endif // !SIMD_STL_HAS_SSE41_SUPPORT
# endif // defined(SIMD_STL_FORCE_SSE41)

#if defined(SIMD_STL_FORCE_SSSE3)
#  if !SIMD_STL_HAS_SSSE3_SUPPORT
#    undef SIMD_STL_FORCE_SSSE3
#  endif // !SIMD_STL_HAS_SSSE3_SUPPORT
# endif // defined(SIMD_STL_FORCE_SSSE3)

#if defined(SIMD_STL_FORCE_SSE3)
#  if !SIMD_STL_HAS_SSE3_SUPPORT
#    undef SIMD_STL_FORCE_SSE3
#  endif // !SIMD_STL_HAS_SSE3_SUPPORT
# endif // defined(SIMD_STL_FORCE_SSE3)

#if defined(SIMD_STL_FORCE_SSE2)
#  if !SIMD_STL_HAS_SSE2_SUPPORT
#    undef SIMD_STL_FORCE_SSE2
#  endif // !SIMD_STL_HAS_SSE2_SUPPORT
# endif // defined(SIMD_STL_FORCE_SSE2)

#if !defined(SIMD_STL_ISA_FORCE_ENABLED)
#  if defined(SIMD_STL_FORCE_AVX512F) || defined(SIMD_STL_FORCE_AVX512BW) || defined(SIMD_STL_FORCE_AVX512CD) \
	|| defined(SIMD_STL_FORCE_AVX512DQ) || defined(SIMD_STL_FORCE_AVX512VL) || defined(SIMD_STL_FORCE_AVX512VBMI) \
	|| defined(SIMD_STL_FORCE_AVX512VBMI2) || defined(SIMD_STL_FORCE_AVX2) || defined(SIMD_STL_FORCE_AVX) \
	|| defined(SIMD_STL_FORCE_SSE42) || defined(SIMD_STL_FORCE_SSE41) || defined(SIMD_STL_FORCE_SSSE3) \
	|| defined(SIMD_STL_FORCE_SSE3) || defined(SIMD_STL_FORCE_SSE2)
#    define SIMD_STL_ISA_FORCE_ENABLED 1 
#  else 
#    define SIMD_STL_ISA_FORCE_ENABLED 0
#  endif // defined(SIMD_STL_FORCE_AVX512F) || defined(SIMD_STL_FORCE_AVX512BW) || ... || defined(SIMD_STL_FORCE_SSE2).
#endif // defined(SIMD_STL_ISA_FORCE_ENABLED)