#pragma once

#include <simd_stl/numeric/BasicSimd.h>

#define __SIMD_STL_FILL_CACHE_SIZE_LIMIT 3*1024*1024

#if !defined(__DISPATCH_VECTORIZED_FILL)
#  define __DISPATCH_VECTORIZED_FILL(byteCount, shift) \
    if constexpr (_Streaming_)  {   \
        _VectorizedFillImplementation_::FillStreamAligned<byteCount>(destination, value, bytes >> shift); \
    }\
    else {  \
        _VectorizedFillImplementation_::Fill<byteCount>(destination, value, bytes >> shift); \
    }
#endif // !defined(__DISPATCH_VECTORIZED_FILL)


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Type_,
    bool                _Aligned_>
struct _FillVectorized {
    using _SimdType_ = numeric::basic_simd<arch::CpuFeature::_SimdGeneration_, _Type_>;

    template <sizetype _ByteCount_>
    simd_stl_always_inline static void Fill(
        _Type_*     destination,
        _Type_      value,
        sizetype    length) noexcept
    {
        constexpr auto registersCount = _SimdType_::registersCount();
        constexpr auto repetitions = _ByteCount_ / sizeof(_SimdType_);

        _SimdType_ vector = _SimdType_::basic_simd<false>();
        vector.fill(value);

        // if constexpr (_ByteCount_ > (sizeof(_SimdType_) * registersCount)) // не оптимально
        while (length--) {
            if constexpr (_Aligned_) {
                vector.storeAligned(destination);
            }
            else if constexpr (_Streaming_) {
                vector.nonTemporalStore(destination);
            }
            else {
                vector.storeUnaligned(destination);
            }
        }

        if constexpr (_Streaming_)
            vector.streamingFence();
    }
};

template <
    typename    _Type_,
    bool        _Aligned_>
struct _FillVectorized<arch::CpuFeature::None, _Type_, _Aligned_> {
    simd_stl_always_inline static void Fill(
        _Type_*     destination,
        _Type_      value,
        sizetype    length) noexcept
    {
        while (length--)
            *destination++ = value;
    }
};


template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Type_>
struct _MemsetVectorizedInternal {
    void operator()(
        void*       destination,
        _Type_      value,
        sizetype    bytes) noexcept
    {
        using _SimdType_ = type_traits::__deduce_simd_vector_type<_SimdGeneration_, int>;
        void* returnValue = destination;

        if ((((uintptr)source & (sizeof(_SimdType_) - 1)) == 0) && (((uintptr)destination & (sizeof(_SimdType_) - 1)) == 0))
        {
            if constexpr (static_cast<int8>(_SimdGeneration_) == static_cast<int8>(arch::CpuFeature::SSE41) ||
                static_cast<int8>(_SimdGeneration_) == static_cast<int8>(arch::CpuFeature::AVX2) ||
                static_cast<int8>(_SimdGeneration_) == static_cast<int8>(arch::CpuFeature::AVX512F))
            {
                if (bytes > __SIMD_STL_FILL_CACHE_SIZE_LIMIT) {
                    _MemsetVectorizedChooser<true, true, _SimdGeneration_>()(destination, source, bytes);
                    return returnValue;
                }
            }

            _MemsetVectorizedChooser<true, false, _SimdGeneration_>()(destination, source, bytes);
        }
        else
            _MemsetVectorizedChooser<false, false, _SimdGeneration_>()(destination, source, bytes);

        return returnValue;
    }
};

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Type_>
simd_stl_always_inline void* _MemsetVectorizedInternal(
    void*       destination,
    _Type_      value,
    sizetype    bytes) noexcept
{
   
    return destination;
}

template <typename _Type_>
void* _MemsetVectorized(
    void*       destination,
    _Type_      value,
    sizetype    bytes) noexcept
{
    if (arch::ProcessorFeatures::AVX512F())
        return _MemsetVectorizedInternal<arch::CpuFeature::AVX512F>(destination, value, bytes);
    else if (arch::ProcessorFeatures::AVX2())
        return _MemsetVectorizedInternal<arch::CpuFeature::AVX2>(destination, value, bytes);
    else if (arch::ProcessorFeatures::AVX())
        return _MemsetVectorizedInternal<arch::CpuFeature::AVX>(destination, value, bytes);
    else if (arch::ProcessorFeatures::SSE41())
        return _MemsetVectorizedInternal<arch::CpuFeature::SSE41>(destination, value, bytes);
    else if (arch::ProcessorFeatures::SSE2())
        return _MemsetVectorizedInternal<arch::CpuFeature::SSE2>(destination, value, bytes);

    return _MemsetVectorizedInternal<arch::CpuFeature::None>(destination, value, bytes);
}

__SIMD_STL_ALGORITHM_NAMESPACE_END
