#pragma once 

#include <src/simd_stl/numeric/BasicSimdImplementationUnspecialized.h>
#include <src/simd_stl/numeric/xmm/SimdDivisors.h>

__SIMD_STL_NUMERIC_NAMESPACE_BEGIN

template <arch::CpuFeature _SimdGeneration_>
class SimdConvert;

template <>
class SimdConvert<arch::CpuFeature::SSE2> {
public:
};

template <>
class SimdConvert<arch::CpuFeature::SSE3> :
    public SimdConvert<arch::CpuFeature::SSE2>
{};

template <>
class SimdConvert<arch::CpuFeature::SSSE3> :
    public SimdConvert<arch::CpuFeature::SSE3>
{};

template <>
class SimdConvert<arch::CpuFeature::SSE41> :
    public SimdConvert<arch::CpuFeature::SSSE3>
{};

template <>
class SimdConvert<arch::CpuFeature::SSE42> :
    public SimdConvert<arch::CpuFeature::SSE41>
{};

__SIMD_STL_NUMERIC_NAMESPACE_END
