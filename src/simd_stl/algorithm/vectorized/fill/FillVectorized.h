#pragma once

#include <simd_stl/numeric/BasicSimd.h>
#include <simd_stl/memory/Alignment.h>

#define __SIMD_STL_FILL_CACHE_SIZE_LIMIT 3*1024*1024


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <typename _Type_>
void* _Memset(
    void*       _Destination,
    _Type_      _Value,
    sizetype    _Length) noexcept
{
    _Type_* _Pointer = static_cast<_Type_*>(_Destination);

    while (_Length--)
        *_Pointer++ = _Value;

    return _Destination;
}

template <typename _Type_>
void _MemsetVectorized(
    void*       _Destination,
    _Type_      _Value,
    sizetype    _Count) noexcept
{
    /*if (arch::ProcessorFeatures::AVX512F())
        return _MemsetVectorizedInternal<arch::CpuFeature::AVX512F>(destination, value, bytes);
    else if (arch::ProcessorFeatures::AVX2())
        return _MemsetVectorizedInternal<arch::CpuFeature::AVX2>(destination, value, bytes);
    else if (arch::ProcessorFeatures::AVX())
        return _MemsetVectorizedInternal<arch::CpuFeature::AVX>(destination, value, bytes);
    else *//*if (arch::ProcessorFeatures::SSE41())
        return _MemsetVectorizedInternal<arch::CpuFeature::SSE41>(destination, value, bytes);
    else *//*if (arch::ProcessorFeatures::SSE2())
        return _MemsetVectorizedInternal<arch::CpuFeature::SSE2>(destination, value, bytes);

    return _MemsetVectorizedInternal<arch::CpuFeature::None>(destination, value, bytes);*/
}

__SIMD_STL_ALGORITHM_NAMESPACE_END
