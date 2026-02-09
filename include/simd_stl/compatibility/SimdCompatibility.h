#pragma once

#include <simd_stl/compatibility/CompilerDetection.h>

#include <vector>
#include <simd_stl/Types.h>

#include <simd_stl/arch/CpuFeature.h>

#if defined(simd_stl_cpp_msvc) && !defined(_M_ARM64EC)
#  include <intrin.h>
#endif

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
