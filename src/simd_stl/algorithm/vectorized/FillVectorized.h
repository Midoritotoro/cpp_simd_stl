#pragma once

#include <simd_stl/numeric/BasicSimd.h>
#include <simd_stl/memory/Alignment.h>

#define __SIMD_STL_FILL_CACHE_SIZE_LIMIT 3*1024*1024


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Type_,
    bool                _Aligned_,
    bool                _Streaming_>
struct _FillVectorized {
    static_assert(_Aligned_ >= _Streaming_, "Streaming loads/stores must be aligned. ");
    using _SimdType_ = numeric::basic_simd<_SimdGeneration_, _Type_>;

    template <sizetype _ElementsCount_>
    simd_stl_always_inline static void Fill(
        _Type_*     destination,
        _Type_      value,
        sizetype    length) noexcept
    {
        if constexpr (
            static_cast<int8>(_SimdGeneration_) == static_cast<int8>(arch::CpuFeature::None) ||
            sizeof(_Type_) * ) {
            while (length--)
                *destination++ = value;

            return;
        }
        else {
            constexpr auto registersCount = _SimdType_::registersCount();

            _SimdType_ vector/* = _SimdType_::template basic_simd<false>()*/;
            vector.fill<_Type_>(value);

            while (length--) {
                if      constexpr (_Streaming_)
                    vector.nonTemporalStore(destination);
                else if constexpr (_Aligned_)
                    vector.storeAligned(destination);
                else
                    vector.storeUnaligned(destination);
            }

            if constexpr (_Streaming_)
                vector.streamingFence();
        }
    }
};

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Type_,
    bool                _Aligned_,
    bool                _Streaming_>
simd_stl_always_inline void _MemsetVectorizedChooser(
    _Type_*     destination,
    _Type_      value,
    sizetype    count) noexcept
{
    sizetype offset = 0;

    const auto bytes = count * sizeof(_Type_);

    while (bytes) {
        if (bytes == sizeof(_Type_)) {
            _FillVectorized<_SimdGeneration_, _Type_, _Aligned_, _Streaming_>::template Fill<1>(destination, value, count);
            offset = bytes & -sizeof(_Type_);
            destination = static_cast<char*>(destination) + offset;
            bytes = 0;
        }
        else if (bytes == (sizeof(_Type_) << 1)) {
            _FillVectorized<_SimdGeneration_, _Type_, _Aligned_, _Streaming_>::template Fill<2>(destination, value, count);
            offset = bytes & -sizeof(_Type_);
            destination = static_cast<char*>(destination) + offset;
            bytes &= sizeof(_Type_) - 1;
        }
        else {

        }
    }
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Type_>
simd_stl_always_inline void _MemsetVectorizedInternal(
    void*       destination,
    _Type_      value,
    sizetype    count) noexcept
{
    using _SimdType_ = type_traits::__deduce_simd_vector_type<_SimdGeneration_, int>;

    if constexpr (std::is_same_v<_SimdType_, void> == false) {
        if (memory::isAligned(destination, sizeof(_SimdType_))) {
            if constexpr (type_traits::is_streaming_supported_v<_SimdGeneration_>)
                if (count > __SIMD_STL_FILL_CACHE_SIZE_LIMIT)
                    return _MemsetVectorizedChooser(static_cast<_Type_*>(destination), value, count);

            return _MemsetVectorizedChooser(static_cast<_Type_*>(destination), value, count);
        }
        else {
            return _MemsetVectorizedChooser(static_cast<_Type_*>(destination), value, count);
        }
    }

    _MemsetVectorizedChooser(static_cast<_Type_*>(destination), value, count);
}

template <typename _Type_>
void _MemsetVectorized(
    void*       destination,
    _Type_      value,
    sizetype    count) noexcept
{
    /*if (arch::ProcessorFeatures::AVX512F())
        return _MemsetVectorizedInternal<arch::CpuFeature::AVX512F>(destination, value, bytes);
    else if (arch::ProcessorFeatures::AVX2())
        return _MemsetVectorizedInternal<arch::CpuFeature::AVX2>(destination, value, bytes);
    else if (arch::ProcessorFeatures::AVX())
        return _MemsetVectorizedInternal<arch::CpuFeature::AVX>(destination, value, bytes);
    else *//*if (arch::ProcessorFeatures::SSE41())
        return _MemsetVectorizedInternal<arch::CpuFeature::SSE41>(destination, value, bytes);
    else */if (arch::ProcessorFeatures::SSE2())
        return _MemsetVectorizedInternal<arch::CpuFeature::SSE2>(destination, value, bytes);

    return _MemsetVectorizedInternal<arch::CpuFeature::None>(destination, value, bytes);
}

__SIMD_STL_ALGORITHM_NAMESPACE_END
