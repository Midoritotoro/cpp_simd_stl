#pragma once

#include <simd_stl/compatibility/CompilerDetection.h>

#include <vector>
#include <simd_stl/Types.h>

#include <simd_stl/arch/CpuFeature.h>

#if defined(simd_stl_cpp_msvc) && !defined(_M_ARM64EC)
#  include <intrin.h>
#endif

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


#if defined(simd_stl_processor_x86) && defined(__SSE2__)
#  include <emmintrin.h>
#endif // defined(simd_stl_processor_x86) && defined(__SSE2__)

__SIMD_STL_ARCH_NAMESPACE_BEGIN

using xmmint    = __m128i;
using ymmint    = __m256i;
using zmmint    = __m512i;

using xmmdouble = __m128d;
using ymmdouble = __m256d;
using zmmdouble = __m512d;

using xmmfloat  = __m128;
using ymmfloat  = __m256;
using zmmfloat  = __m512;

__SIMD_STL_ARCH_NAMESPACE_END
