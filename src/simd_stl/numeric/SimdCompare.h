#pragma once 

#include <src/simd_stl/numeric/SimdCast.h>

__SIMD_STL_NUMERIC_NAMESPACE_BEGIN

template <arch::CpuFeature _SimdGeneration_>
class SimdCompare;

template <>
class SimdCompare<arch::CpuFeature::SSE2> {
public:
    template <
        typename _DesiredType_,
        class    _CompareType_,
        typename _VectorType_>
    static simd_stl_always_inline auto compare(
        _VectorType_ left,
        _VectorType_ right) noexcept
    {
        /* if constexpr (std::is_same_v<_CompareType, type_traits::equal_to>) {

         }*/
    }
};

template <>
class SimdCompare<arch::CpuFeature::SSE3> :
    public SimdCompare<arch::CpuFeature::SSE2>
{};

template <>
class SimdCompare<arch::CpuFeature::SSSE3> :
    public SimdCompare<arch::CpuFeature::SSE3>
{};

template <>
class SimdCompare<arch::CpuFeature::SSE41> :
    public SimdCompare<arch::CpuFeature::SSSE3>
{};

template <>
class SimdCompare<arch::CpuFeature::SSE42> :
    public SimdCompare<arch::CpuFeature::SSE41>
{};

__SIMD_STL_NUMERIC_NAMESPACE_END