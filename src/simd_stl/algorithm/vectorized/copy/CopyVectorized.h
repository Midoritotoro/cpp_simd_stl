#pragma once

#include <src/simd_stl/algorithm/vectorized/copy/MoveVectorized.h>

#include <simd_stl/memory/Intersects.h>
#include <simd_stl/memory/Alignment.h>

#define __SIMD_STL_COPY_CACHE_SIZE_LIMIT 3*1024*1024

#if !defined(__DISPATCH_VECTORIZED_COPY)
#  define __DISPATCH_VECTORIZED_COPY(byteCount, shift) \
    if constexpr (_Streaming_)  {   \
        _VectorizedCopyImplementation_::CopyStreamAligned<byteCount>(destination, source, bytes >> shift); \
    }\
    else {  \
        _VectorizedCopyImplementation_::Copy<byteCount>(destination, source, bytes >> shift); \
    }
#endif // !defined(__DISPATCH_VECTORIZED_COPY)


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <
    arch::CpuFeature    _SimdGeneration_,
    bool                _Aligned_>
struct _CopyVectorized;

template <>
struct _CopyVectorized<arch::CpuFeature::None, false> {
    template <sizetype    _ElementSize_>
    simd_stl_always_inline static void* Copy(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        AssertUnreachable();
        return nullptr;
    }

    template <>
    simd_stl_always_inline static void* Copy<1>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const uint8* __sourceByte   = static_cast<const uint8*>(source);
        uint8* __destinationByte    = static_cast<uint8*>(destination);

        while (length--)
            *__destinationByte++ = *__sourceByte++;

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Copy<2>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const uint16* __sourceWord  = static_cast<const uint16*>(source);
        uint16* __destinationWord   = static_cast<uint16*>(destination);

        while (length--)
            *__destinationWord++ = *__sourceWord++;

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Copy<4>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const uint32* __sourceDWord  = static_cast<const uint32*>(source);
        uint32* __destinationDWord   = static_cast<uint32*>(destination);

        while (length--)
            *__destinationDWord++ = *__sourceDWord++;

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Copy<8>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const uint64* __sourceQWord = static_cast<const uint64*>(source);
        uint64* __destinationQWord  = static_cast<uint64*>(destination);

        while (length--)
            *__destinationQWord++ = *__sourceQWord++;

        return destination;
    }
};

template <>
struct _CopyVectorized<arch::CpuFeature::SSE2, false>
{
    template <sizetype _ElementSize_>
    simd_stl_always_inline static void* Copy(
        void*       destination,
        const void* source,
        sizetype    length) noexcept
    {
        AssertUnreachable();
        return nullptr;
    }

    template <>
    simd_stl_always_inline static void* Copy<1>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const uint8* __sourceByte   = static_cast<const uint8*>(source);
        uint8* __destinationByte    = static_cast<uint8*>(destination);

        while (length--)
            *__destinationByte++ = *__sourceByte++;

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Copy<2>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const uint16* __sourceWord  = static_cast<const uint16*>(source);
        uint16* __destinationWord   = static_cast<uint16*>(destination);

        while (length--)
            *__destinationWord++ = *__sourceWord++;

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Copy<4>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const uint32* __sourceDWord  = static_cast<const uint32*>(source);
        uint32* __destinationDWord   = static_cast<uint32*>(destination);

        while (length--)
            *__destinationDWord++ = *__sourceDWord++;

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Copy<8>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const uint64* __sourceQWord = static_cast<const uint64*>(source);
        uint64* __destinationQWord  = static_cast<uint64*>(destination);

        while (length--)
            *__destinationQWord++ = *__sourceQWord++;

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Copy<16>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const __m128i* __xmmWordSource  = reinterpret_cast<const __m128i*>(source);
        __m128i* __xmmWordDestination   = reinterpret_cast<__m128i*>(destination);

        while (length--)
            _mm_storeu_si128(__xmmWordDestination++, _mm_loadu_si128(__xmmWordSource++));

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Copy<32>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const __m128i* __xmmWordSource  = reinterpret_cast<const __m128i*>(source);
        __m128i* __xmmWordDestination   = reinterpret_cast<__m128i*>(destination);

        while (length--) {
            __SIMD_STL_REPEAT_N(2, _mm_storeu_si128(__xmmWordDestination++, _mm_loadu_si128(__xmmWordSource++)));
        }

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Copy<64>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const __m128i* __xmmWordSource  = reinterpret_cast<const __m128i*>(source);
        __m128i* __xmmWordDestination   = reinterpret_cast<__m128i*>(destination);

        while (length--) {
            __SIMD_STL_REPEAT_N(4, _mm_storeu_si128(__xmmWordDestination++, _mm_loadu_si128(__xmmWordSource++)));
        }

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Copy<128>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const __m128i* __xmmWordSource  = reinterpret_cast<const __m128i*>(source);
        __m128i* __xmmWordDestination   = reinterpret_cast<__m128i*>(destination);

        while (length--) {
            __SIMD_STL_REPEAT_N(8, _mm_storeu_si128(__xmmWordDestination++, _mm_loadu_si128(__xmmWordSource++)));
        }

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Copy<256>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const __m128i* __xmmWordSource  = reinterpret_cast<const __m128i*>(source);
        __m128i* __xmmWordDestination   = reinterpret_cast<__m128i*>(destination);

        while (length--) {
            __SIMD_STL_REPEAT_N(16, _mm_storeu_si128(__xmmWordDestination++, _mm_loadu_si128(__xmmWordSource++)));
        }

        return destination;
    }
};

template <>
struct _CopyVectorized<arch::CpuFeature::SSE2, true>
{
    template <sizetype _ElementSize_>
    simd_stl_always_inline static void* Copy(
        void*       destination,
        const void* source,
        sizetype    length) noexcept
    {
        AssertUnreachable();
        return nullptr;
    }

    template <>
    simd_stl_always_inline static void* Copy<1>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const uint8* __sourceByte   = static_cast<const uint8*>(source);
        uint8* __destinationByte    = static_cast<uint8*>(destination);

        while (length--)
            *__destinationByte++ = *__sourceByte++;

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Copy<2>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const uint16* __sourceWord  = static_cast<const uint16*>(source);
        uint16* __destinationWord   = static_cast<uint16*>(destination);

        while (length--)
            *__destinationWord++ = *__sourceWord++;

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Copy<4>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const uint32* __sourceDWord  = static_cast<const uint32*>(source);
        uint32* __destinationDWord   = static_cast<uint32*>(destination);

        while (length--)
            *__destinationDWord++ = *__sourceDWord++;

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Copy<8>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const uint64* __sourceQWord = static_cast<const uint64*>(source);
        uint64* __destinationQWord  = static_cast<uint64*>(destination);

        while (length--)
            *__destinationQWord++ = *__sourceQWord++;

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Copy<16>(
        void*       destination,
        const void* source,
        sizetype    length) noexcept
    {
        const __m128i* __xmmWordSource  = reinterpret_cast<const __m128i*>(source);
        __m128i* __xmmWordDestination   = reinterpret_cast<__m128i*>(destination);

        while (length--)
            _mm_store_si128(__xmmWordDestination++, _mm_load_si128(__xmmWordSource++));

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Copy<32>(
        void*       destination,
        const void* source,
        sizetype    length) noexcept
    {
        const __m128i* __xmmWordSource  = reinterpret_cast<const __m128i*>(source);
        __m128i* __xmmWordDestination   = reinterpret_cast<__m128i*>(destination);

        while (length--) {
            __SIMD_STL_REPEAT_N(2, _mm_store_si128(__xmmWordDestination++, _mm_load_si128(__xmmWordSource++)));
        }

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Copy<64>(
        void*       destination,
        const void* source,
        sizetype    length) noexcept
    {
        const __m128i* __xmmWordSource  = reinterpret_cast<const __m128i*>(source);
        __m128i* __xmmWordDestination   = reinterpret_cast<__m128i*>(destination);

        while (length--) {
            __SIMD_STL_REPEAT_N(4, _mm_store_si128(__xmmWordDestination++, _mm_load_si128(__xmmWordSource++)));
        }

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Copy<128>(
        void*       destination,
        const void* source,
        sizetype    length) noexcept
    {
        const __m128i* __xmmWordSource  = reinterpret_cast<const __m128i*>(source);
        __m128i* __xmmWordDestination   = reinterpret_cast<__m128i*>(destination);

        while (length--) {
            __SIMD_STL_REPEAT_N(8, _mm_store_si128(__xmmWordDestination++, _mm_load_si128(__xmmWordSource++)));
        }

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Copy<256>(
        void*       destination,
        const void* source,
        sizetype    length) noexcept
    {
        const __m128i* __xmmWordSource  = reinterpret_cast<const __m128i*>(source);
        __m128i* __xmmWordDestination   = reinterpret_cast<__m128i*>(destination);

        while (length--) {
            __SIMD_STL_REPEAT_N(16, _mm_store_si128(__xmmWordDestination++, _mm_load_si128(__xmmWordSource++)));
        }

        return destination;
    }
    
};

template <>
struct _CopyVectorized<arch::CpuFeature::AVX, false>
{
    template <sizetype _ElementSize_>
    simd_stl_always_inline static void* Copy(
        void*       destination,
        const void* source,
        sizetype    length) noexcept
    {
        AssertUnreachable();
        return nullptr;
    }

    template <>
    simd_stl_always_inline static void* Copy<1>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const uint8* __sourceByte   = static_cast<const uint8*>(source);
        uint8* __destinationByte    = static_cast<uint8*>(destination);

        while (length--)
            *__destinationByte++ = *__sourceByte++;

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Copy<2>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const uint16* __sourceWord  = static_cast<const uint16*>(source);
        uint16* __destinationWord   = static_cast<uint16*>(destination);

        while (length--)
            *__destinationWord++ = *__sourceWord++;

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Copy<4>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const uint32* __sourceDWord  = static_cast<const uint32*>(source);
        uint32* __destinationDWord   = static_cast<uint32*>(destination);

        while (length--)
            *__destinationDWord++ = *__sourceDWord++;

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Copy<8>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const uint64* __sourceQWord = static_cast<const uint64*>(source);
        uint64* __destinationQWord  = static_cast<uint64*>(destination);

        while (length--)
            *__destinationQWord++ = *__sourceQWord++;

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Copy<16>(
        void*       destination,
        const void* source,
        sizetype    length) noexcept
    {
        const __m128i* __xmmWordSource = reinterpret_cast<const __m128i*>(source);
        __m128i* __xmmWordDestination = reinterpret_cast<__m128i*>(destination);

        while (length--)
            _mm_storeu_si128(__xmmWordDestination++, _mm_loadu_si128(__xmmWordSource++));

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Copy<32>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const __m256i* __ymmWordSource  = reinterpret_cast<const __m256i*>(source);
        __m256i* __ymmWordDestination   = reinterpret_cast<__m256i*>(destination);

        while (length--)
            _mm256_storeu_si256(__ymmWordDestination++, _mm256_lddqu_si256(__ymmWordSource++));

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Copy<64>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const __m256i* __ymmWordSource  = reinterpret_cast<const __m256i*>(source);
        __m256i* __ymmWordDestination   = reinterpret_cast<__m256i*>(destination);

        while (length--) {
            __SIMD_STL_REPEAT_N(2, _mm256_storeu_si256(__ymmWordDestination++, _mm256_lddqu_si256(__ymmWordSource++)));
        }

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Copy<128>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const __m256i* __ymmWordSource  = reinterpret_cast<const __m256i*>(source);
        __m256i* __ymmWordDestination   = reinterpret_cast<__m256i*>(destination);

        while (length--) {
            __SIMD_STL_REPEAT_N(4, _mm256_storeu_si256(__ymmWordDestination++, _mm256_lddqu_si256(__ymmWordSource++)));
        }

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Copy<256>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const __m256i* __ymmWordSource  = reinterpret_cast<const __m256i*>(source);
        __m256i* __ymmWordDestination   = reinterpret_cast<__m256i*>(destination);

        while (length--) {
            __SIMD_STL_REPEAT_N(8, _mm256_storeu_si256(__ymmWordDestination++, _mm256_lddqu_si256(__ymmWordSource++)));
        }

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Copy<512>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const __m256i* __ymmWordSource  = reinterpret_cast<const __m256i*>(source);
        __m256i* __ymmWordDestination   = reinterpret_cast<__m256i*>(destination);

        while (length--) {
            __SIMD_STL_REPEAT_N(16, _mm256_storeu_si256(__ymmWordDestination++, _mm256_lddqu_si256(__ymmWordSource++)));
        }

        return destination;
    }
};

template <> 
struct _CopyVectorized<arch::CpuFeature::AVX, true>
{
    template <sizetype _ElementSize_>
    simd_stl_always_inline static void* Copy(
        void*       destination,
        const void* source,
        sizetype    length) noexcept
    {
        AssertUnreachable();
        return nullptr;
    }

    template <>
    simd_stl_always_inline static void* Copy<1>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const uint8* __sourceByte   = static_cast<const uint8*>(source);
        uint8* __destinationByte    = static_cast<uint8*>(destination);

        while (length--)
            *__destinationByte++ = *__sourceByte++;

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Copy<2>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const uint16* __sourceWord  = static_cast<const uint16*>(source);
        uint16* __destinationWord   = static_cast<uint16*>(destination);

        while (length--)
            *__destinationWord++ = *__sourceWord++;

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Copy<4>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const uint32* __sourceDWord  = static_cast<const uint32*>(source);
        uint32* __destinationDWord   = static_cast<uint32*>(destination);

        while (length--)
            *__destinationDWord++ = *__sourceDWord++;

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Copy<8>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const uint64* __sourceQWord = static_cast<const uint64*>(source);
        uint64* __destinationQWord  = static_cast<uint64*>(destination);

        while (length--)
            *__destinationQWord++ = *__sourceQWord++;

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Copy<16>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const __m128i* __xmmWordSource  = reinterpret_cast<const __m128i*>(source);
        __m128i* __xmmWordDestination   = reinterpret_cast<__m128i*>(destination);

        while (length--)
            _mm_store_si128(__xmmWordDestination++, _mm_load_si128(__xmmWordSource++));

        return destination;
    }


    template <>
    simd_stl_always_inline static void* Copy<32>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const __m256i* __ymmWordSource  = reinterpret_cast<const __m256i*>(source);
        __m256i* __ymmWordDestination   = reinterpret_cast<__m256i*>(destination);

        while (length--)
            _mm256_store_si256(__ymmWordDestination++, _mm256_load_si256(__ymmWordSource++));

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Copy<64>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const __m256i* __ymmWordSource  = reinterpret_cast<const __m256i*>(source);
        __m256i* __ymmWordDestination   = reinterpret_cast<__m256i*>(destination);

        while (length--) {
            __SIMD_STL_REPEAT_N(2, _mm256_store_si256(__ymmWordDestination++, _mm256_load_si256(__ymmWordSource++)));
        }

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Copy<128>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const __m256i* __ymmWordSource  = reinterpret_cast<const __m256i*>(source);
        __m256i* __ymmWordDestination   = reinterpret_cast<__m256i*>(destination);

        while (length--) {
            __SIMD_STL_REPEAT_N(4, _mm256_store_si256(__ymmWordDestination++, _mm256_load_si256(__ymmWordSource++)));
        }

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Copy<256>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const __m256i* __ymmWordSource  = reinterpret_cast<const __m256i*>(source);
        __m256i* __ymmWordDestination   = reinterpret_cast<__m256i*>(destination);

        while (length--) {
            __SIMD_STL_REPEAT_N(8, _mm256_store_si256(__ymmWordDestination++, _mm256_load_si256(__ymmWordSource++)));
        }

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Copy<512>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const __m256i* __ymmWordSource  = reinterpret_cast<const __m256i*>(source);
        __m256i* __ymmWordDestination   = reinterpret_cast<__m256i*>(destination);

        while (length--) {
            __SIMD_STL_REPEAT_N(16, _mm256_store_si256(__ymmWordDestination++, _mm256_load_si256(__ymmWordSource++)));
        }

        return destination;
    }
};

template <>
struct _CopyVectorized<arch::CpuFeature::AVX512F, false>
{
    template <sizetype _ElementSize_>
    simd_stl_always_inline static void* Copy(
        void*       destination,
        const void* source,
        sizetype    length) noexcept
    {
        AssertUnreachable();
        return nullptr;
    }

    template <>
    simd_stl_always_inline static void* Copy<1>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const uint8* __sourceByte   = static_cast<const uint8*>(source);
        uint8* __destinationByte    = static_cast<uint8*>(destination);

        while (length--)
            *__destinationByte++ = *__sourceByte++;

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Copy<2>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const uint16* __sourceWord  = static_cast<const uint16*>(source);
        uint16* __destinationWord   = static_cast<uint16*>(destination);

        while (length--)
            *__destinationWord++ = *__sourceWord++;

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Copy<4>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const uint32* __sourceDWord  = static_cast<const uint32*>(source);
        uint32* __destinationDWord   = static_cast<uint32*>(destination);

        while (length--)
            *__destinationDWord++ = *__sourceDWord++;

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Copy<8>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const uint64* __sourceQWord = static_cast<const uint64*>(source);
        uint64* __destinationQWord  = static_cast<uint64*>(destination);

        while (length--)
            *__destinationQWord++ = *__sourceQWord++;

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Copy<16>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const __m128i* __xmmWordSource  = reinterpret_cast<const __m128i*>(source);
        __m128i* __xmmWordDestination   = reinterpret_cast<__m128i*>(destination);

        while (length--)
            _mm_storeu_si128(__xmmWordDestination++, _mm_loadu_si128(__xmmWordSource++));

        return destination;
    }


    template <>
    simd_stl_always_inline static void* Copy<32>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const __m256i* __ymmWordSource  = reinterpret_cast<const __m256i*>(source);
        __m256i* __ymmWordDestination   = reinterpret_cast<__m256i*>(destination);

        while (length--)
            _mm256_storeu_si256(__ymmWordDestination++, _mm256_lddqu_si256(__ymmWordSource++));

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Copy<64>(
        void*       destination,
        const void* source,
        sizetype    length) noexcept
    {
        const __m512i* __zmmWordSource = reinterpret_cast<const __m512i*>(source);
        __m512i* __zmmWordDestination = reinterpret_cast<__m512i*>(destination);

        while (length--)
            _mm512_storeu_si512(__zmmWordDestination++, _mm512_loadu_si512(__zmmWordSource++));

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Copy<128>(
        void*       destination,
        const void* source,
        sizetype    length) noexcept
    {
        const __m512i* __zmmWordSource = reinterpret_cast<const __m512i*>(source);
        __m512i* __zmmWordDestination = reinterpret_cast<__m512i*>(destination);

        while (length--) {
            __SIMD_STL_REPEAT_N(2, _mm512_storeu_si512(__zmmWordDestination++, _mm512_loadu_si512(__zmmWordSource++)));
        }

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Copy<256>(
        void*       destination,
        const void* source,
        sizetype    length) noexcept
    {
        const __m512i* __zmmWordSource = reinterpret_cast<const __m512i*>(source);
        __m512i* __zmmWordDestination = reinterpret_cast<__m512i*>(destination);

        while (length--) {
            __SIMD_STL_REPEAT_N(4, _mm512_storeu_si512(__zmmWordDestination++, _mm512_loadu_si512(__zmmWordSource++)));
        }

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Copy<512>(
        void*       destination,
        const void* source,
        sizetype    length) noexcept
    {
        const __m512i* __zmmWordSource = reinterpret_cast<const __m512i*>(source);
        __m512i* __zmmWordDestination = reinterpret_cast<__m512i*>(destination);

        while (length--) {
            __SIMD_STL_REPEAT_N(8, _mm512_storeu_si512(__zmmWordDestination++, _mm512_loadu_si512(__zmmWordSource++)));
        }

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Copy<1024>(
        void*       destination,
        const void* source,
        sizetype    length) noexcept
    {
        const __m512i* __zmmWordSource = reinterpret_cast<const __m512i*>(source);
        __m512i* __zmmWordDestination = reinterpret_cast<__m512i*>(destination);

        while (length--) {
            __SIMD_STL_REPEAT_N(16, _mm512_storeu_si512(__zmmWordDestination++, _mm512_loadu_si512(__zmmWordSource++)));
        }

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Copy<2048>(
        void*       destination,
        const void* source,
        sizetype    length) noexcept
    {
        const __m512i* __zmmWordSource = reinterpret_cast<const __m512i*>(source);
        __m512i* __zmmWordDestination = reinterpret_cast<__m512i*>(destination);

        while (length--) {
            __SIMD_STL_REPEAT_N(32, _mm512_storeu_si512(__zmmWordDestination++, _mm512_loadu_si512(__zmmWordSource++)));
        }

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Copy<4096>(
        void*       destination,
        const void* source,
        sizetype    length) noexcept
    {
        const __m512i* __zmmWordSource = reinterpret_cast<const __m512i*>(source);
        __m512i* __zmmWordDestination = reinterpret_cast<__m512i*>(destination);

        while (length--) {
            __SIMD_STL_REPEAT_N(64, _mm512_storeu_si512(__zmmWordDestination++, _mm512_loadu_si512(__zmmWordSource++)));
        }

        return destination;
    }
};

template <>
struct _CopyVectorized<arch::CpuFeature::AVX512F, true>
{
    template <sizetype _ElementSize_>
    simd_stl_always_inline static void* Copy(
        void*       destination,
        const void* source,
        sizetype    length) noexcept
    {
        AssertUnreachable();
        return nullptr;
    }

    template <>
    simd_stl_always_inline static void* Copy<1>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const uint8* __sourceByte   = static_cast<const uint8*>(source);
        uint8* __destinationByte    = static_cast<uint8*>(destination);

        while (length--)
            *__destinationByte++ = *__sourceByte++;

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Copy<2>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const uint16* __sourceWord  = static_cast<const uint16*>(source);
        uint16* __destinationWord   = static_cast<uint16*>(destination);

        while (length--)
            *__destinationWord++ = *__sourceWord++;

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Copy<4>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const uint32* __sourceDWord  = static_cast<const uint32*>(source);
        uint32* __destinationDWord   = static_cast<uint32*>(destination);

        while (length--)
            *__destinationDWord++ = *__sourceDWord++;

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Copy<8>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const uint64* __sourceQWord = static_cast<const uint64*>(source);
        uint64* __destinationQWord  = static_cast<uint64*>(destination);

        while (length--)
            *__destinationQWord++ = *__sourceQWord++;

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Copy<16>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const __m128i* __xmmWordSource  = reinterpret_cast<const __m128i*>(source);
        __m128i* __xmmWordDestination   = reinterpret_cast<__m128i*>(destination);

        while (length--)
            _mm_store_si128(__xmmWordDestination++, _mm_load_si128(__xmmWordSource++));

        return destination;
    }


    template <>
    simd_stl_always_inline static void* Copy<32>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const __m256i* __ymmWordSource  = reinterpret_cast<const __m256i*>(source);
        __m256i* __ymmWordDestination   = reinterpret_cast<__m256i*>(destination);

        while (length--)
            _mm256_store_si256(__ymmWordDestination++, _mm256_load_si256(__ymmWordSource++));

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Copy<64>(
        void*       destination,
        const void* source,
        sizetype    length) noexcept
    {
        const __m512i* __zmmWordSource  = reinterpret_cast<const __m512i*>(source);
        __m512i* __zmmWordDestination   = reinterpret_cast<__m512i*>(destination);

        while (length--)
            _mm512_store_si512(__zmmWordDestination++, _mm512_load_si512(__zmmWordSource++));

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Copy<128>(
        void*       destination,
        const void* source,
        sizetype    length) noexcept
    {
        const __m512i* __zmmWordSource  = reinterpret_cast<const __m512i*>(source);
        __m512i* __zmmWordDestination   = reinterpret_cast<__m512i*>(destination);

        while (length--) {
            __SIMD_STL_REPEAT_N(2, _mm512_store_si512(__zmmWordDestination++, _mm512_load_si512(__zmmWordSource++)));
        }

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Copy<256>(
        void*       destination,
        const void* source,
        sizetype    length) noexcept
    {
        const __m512i* __zmmWordSource  = reinterpret_cast<const __m512i*>(source);
        __m512i* __zmmWordDestination   = reinterpret_cast<__m512i*>(destination);

        while (length--) {
            __SIMD_STL_REPEAT_N(4, _mm512_store_si512(__zmmWordDestination++, _mm512_load_si512(__zmmWordSource++)));
        }

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Copy<512>(
        void*       destination,
        const void* source,
        sizetype    length) noexcept
    {
        const __m512i* __zmmWordSource  = reinterpret_cast<const __m512i*>(source);
        __m512i* __zmmWordDestination   = reinterpret_cast<__m512i*>(destination);

        while (length--) {
            __SIMD_STL_REPEAT_N(8, _mm512_store_si512(__zmmWordDestination++, _mm512_load_si512(__zmmWordSource++)));
        }

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Copy<1024>(
        void*       destination,
        const void* source,
        sizetype    length) noexcept
    {
        const __m512i* __zmmWordSource  = reinterpret_cast<const __m512i*>(source);
        __m512i* __zmmWordDestination   = reinterpret_cast<__m512i*>(destination);

        while (length--) {
            __SIMD_STL_REPEAT_N(16, _mm512_store_si512(__zmmWordDestination++, _mm512_load_si512(__zmmWordSource++)));
        }

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Copy<2048>(
        void*       destination,
        const void* source,
        sizetype    length) noexcept
    {
        const __m512i* __zmmWordSource  = reinterpret_cast<const __m512i*>(source);
        __m512i* __zmmWordDestination   = reinterpret_cast<__m512i*>(destination);

        while (length--) {
            __SIMD_STL_REPEAT_N(32, _mm512_store_si512(__zmmWordDestination++, _mm512_load_si512(__zmmWordSource++)));
        }

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Copy<4096>(
        void*       destination,
        const void* source,
        sizetype    length) noexcept
    {
        const __m512i* __zmmWordSource  = reinterpret_cast<const __m512i*>(source);
        __m512i* __zmmWordDestination   = reinterpret_cast<__m512i*>(destination);

        while (length--) {
            __SIMD_STL_REPEAT_N(64, _mm512_store_si512(__zmmWordDestination++, _mm512_load_si512(__zmmWordSource++)));
        }

        return destination;
    }

    // ==============================================================================
    //                              ALIGNED, STREAMING
    // ==============================================================================

    template <sizetype _ElementSize_>
    simd_stl_always_inline static void* CopyStreamAligned(
        void*       destination,
        const void* source,
        sizetype    length) noexcept
    {
        AssertUnreachable();
        return nullptr;
    }

    template <>
    simd_stl_always_inline static void* CopyStreamAligned<64>(
        void*       destination,
        const void* source,
        sizetype    length) noexcept
    {
        const __m512i* __zmmWordSource  = reinterpret_cast<const __m512i*>(source);
        __m512i* __zmmWordDestination   = reinterpret_cast<__m512i*>(destination);

        while (length--)
            _mm512_stream_si512(__zmmWordDestination++, _mm512_stream_load_si512(__zmmWordSource++));

        _mm_sfence();

        return destination;
    }

    template <>
    simd_stl_always_inline static void* CopyStreamAligned<128>(
        void*       destination,
        const void* source,
        sizetype    length) noexcept
    {
        const __m512i* __zmmWordSource  = reinterpret_cast<const __m512i*>(source);
        __m512i* __zmmWordDestination   = reinterpret_cast<__m512i*>(destination);

        while (length--) {
            __SIMD_STL_REPEAT_N(2, _mm512_stream_si512(__zmmWordDestination++, _mm512_stream_load_si512(__zmmWordSource++)));
        }

        _mm_sfence();

        return destination;
    }

    template <>
    simd_stl_always_inline static void* CopyStreamAligned<256>(
        void*       destination,
        const void* source,
        sizetype    length) noexcept
    {
        const __m512i* __zmmWordSource  = reinterpret_cast<const __m512i*>(source);
        __m512i* __zmmWordDestination   = reinterpret_cast<__m512i*>(destination);

        while (length--) {
            __SIMD_STL_REPEAT_N(4, _mm512_stream_si512(__zmmWordDestination++, _mm512_stream_load_si512(__zmmWordSource++)));
        }

        _mm_sfence();

        return destination;
    }

    template <>
    simd_stl_always_inline static void* CopyStreamAligned<512>(
        void*       destination,
        const void* source,
        sizetype    length) noexcept
    {
        const __m512i* __zmmWordSource  = reinterpret_cast<const __m512i*>(source);
        __m512i* __zmmWordDestination   = reinterpret_cast<__m512i*>(destination);

        while (length--) {
            __SIMD_STL_REPEAT_N(8, _mm512_stream_si512(__zmmWordDestination++, _mm512_stream_load_si512(__zmmWordSource++)));
        }

        _mm_sfence();

        return destination;
    }

    template <>
    simd_stl_always_inline static void* CopyStreamAligned<1024>(
        void*       destination,
        const void* source,
        sizetype    length) noexcept
    {
        const __m512i* __zmmWordSource  = reinterpret_cast<const __m512i*>(source);
        __m512i* __zmmWordDestination   = reinterpret_cast<__m512i*>(destination);

        while (length--) {
            __SIMD_STL_REPEAT_N(16, _mm512_stream_si512(__zmmWordDestination++, _mm512_stream_load_si512(__zmmWordSource++)));
        }

        _mm_sfence();

        return destination;
    }

    template <>
    simd_stl_always_inline static void* CopyStreamAligned<2048>(
        void*       destination,
        const void* source,
        sizetype    length) noexcept
    {
        const __m512i* __zmmWordSource  = reinterpret_cast<const __m512i*>(source);
        __m512i* __zmmWordDestination   = reinterpret_cast<__m512i*>(destination);

        while (length--) {
            __SIMD_STL_REPEAT_N(32, _mm512_stream_si512(__zmmWordDestination++, _mm512_stream_load_si512(__zmmWordSource++)));
        }

        _mm_sfence();

        return destination;
    }

    template <>
    simd_stl_always_inline static void* CopyStreamAligned<4096>(
        void*       destination,
        const void* source,
        sizetype    length) noexcept
    {
        const __m512i* __zmmWordSource  = reinterpret_cast<const __m512i*>(source);
        __m512i* __zmmWordDestination   = reinterpret_cast<__m512i*>(destination);

        while (length--) {
            __SIMD_STL_REPEAT_N(64, _mm512_stream_si512(__zmmWordDestination++, _mm512_stream_load_si512(__zmmWordSource++)));
        }

        _mm_sfence();

        return destination;
    }
};

template <>
struct _CopyVectorized<arch::CpuFeature::AVX2, true>:
    _CopyVectorized<arch::CpuFeature::AVX, true>
{
    // ==============================================================================
    //                              ALIGNED, STREAMING
    // ==============================================================================


    template <sizetype _ElementSize_>
    simd_stl_always_inline static void* CopyStreamAligned(
        void*       destination,
        const void* source,
        sizetype    length) noexcept
    {
        AssertUnreachable();
        return nullptr;
    }

    template <>
    simd_stl_always_inline static void* CopyStreamAligned<32>(
        void*       destination,
        const void* source,
        sizetype    length) noexcept
    {
        const __m256i* __ymmWordSource  = reinterpret_cast<const __m256i*>(source);
        __m256i* __ymmWordDestination   = reinterpret_cast<__m256i*>(destination);

        while (length--)
            _mm256_stream_si256(__ymmWordDestination++, _mm256_stream_load_si256(__ymmWordSource++));

        _mm_sfence();

        return destination;
    }

    template <>
    simd_stl_always_inline static void* CopyStreamAligned<64>(
        void*       destination,
        const void* source,
        sizetype    length) noexcept
    {
        const __m256i* __ymmWordSource  = reinterpret_cast<const __m256i*>(source);
        __m256i* __ymmWordDestination   = reinterpret_cast<__m256i*>(destination);

        while (length--) {
            __SIMD_STL_REPEAT_N(2, _mm256_stream_si256(__ymmWordDestination++, _mm256_stream_load_si256(__ymmWordSource++)));
        }

        _mm_sfence();

        return destination;
    }

    template <>
    simd_stl_always_inline static void* CopyStreamAligned<128>(
        void*       destination,
        const void* source,
        sizetype    length) noexcept
    {
        const __m256i* __ymmWordSource  = reinterpret_cast<const __m256i*>(source);
        __m256i* __ymmWordDestination   = reinterpret_cast<__m256i*>(destination);

        while (length--) {
            __SIMD_STL_REPEAT_N(4, _mm256_stream_si256(__ymmWordDestination++, _mm256_stream_load_si256(__ymmWordSource++)));
        }

        _mm_sfence();

        return destination;
    }

    template <>
    simd_stl_always_inline static void* CopyStreamAligned<256>(
        void*       destination,
        const void* source,
        sizetype    length) noexcept
    {
        const __m256i* __ymmWordSource  = reinterpret_cast<const __m256i*>(source);
        __m256i* __ymmWordDestination   = reinterpret_cast<__m256i*>(destination);

        while (length--) {
            __SIMD_STL_REPEAT_N(8, _mm256_stream_si256(__ymmWordDestination++, _mm256_stream_load_si256(__ymmWordSource++)));
        }

        _mm_sfence();

        return destination;
    }

    template <>
    simd_stl_always_inline static void* CopyStreamAligned<512>(
        void*       destination,
        const void* source,
        sizetype    length) noexcept
    {
        const __m256i* __ymmWordSource  = reinterpret_cast<const __m256i*>(source);
        __m256i* __ymmWordDestination   = reinterpret_cast<__m256i*>(destination);

        while (length--) {
            __SIMD_STL_REPEAT_N(16, _mm256_stream_si256(__ymmWordDestination++, _mm256_stream_load_si256(__ymmWordSource++)));
        }

        _mm_sfence();

        return destination;
    }
};

template <>
struct _CopyVectorized<arch::CpuFeature::AVX2, false>:
    _CopyVectorized<arch::CpuFeature::AVX, false>
{};

template <>
struct _CopyVectorized<arch::CpuFeature::SSE41, false>:
    _CopyVectorized<arch::CpuFeature::SSE2, false>
{};

template <>
struct _CopyVectorized<arch::CpuFeature::SSE41, true>:
    _CopyVectorized<arch::CpuFeature::SSE2, true>
{
        // ==============================================================================
    //                              ALIGNED, STREAMING
    // ==============================================================================

    template <sizetype _ElementSize_>
    simd_stl_always_inline static void* CopyStreamAligned(
        void*       destination,
        const void* source,
        sizetype    length) noexcept
    {
        AssertUnreachable();
        return nullptr;
    }

    template <>
    simd_stl_always_inline static void* CopyStreamAligned<16>(
        void*       destination,
        const void* source,
        sizetype    length) noexcept
    {
        const __m128i* __xmmWordSource  = reinterpret_cast<const __m128i*>(source);
        __m128i* __xmmWordDestination   = reinterpret_cast<__m128i*>(destination);

        while (length--)
            _mm_stream_si128(__xmmWordDestination++, _mm_stream_load_si128(__xmmWordSource++));

        _mm_sfence();

        return destination;
    }

    template <>
    simd_stl_always_inline static void* CopyStreamAligned<32>(
        void*       destination,
        const void* source,
        sizetype    length) noexcept
    {
        const __m128i* __xmmWordSource  = reinterpret_cast<const __m128i*>(source);
        __m128i* __xmmWordDestination   = reinterpret_cast<__m128i*>(destination);

        while (length--) {
            __SIMD_STL_REPEAT_N(2, _mm_stream_si128(__xmmWordDestination++, _mm_stream_load_si128(__xmmWordSource++)));
        }

        _mm_sfence();

        return destination;
    }

    template <>
    simd_stl_always_inline static void* CopyStreamAligned<64>(
        void*       destination,
        const void* source,
        sizetype    length) noexcept
    {
        const __m128i* __xmmWordSource  = reinterpret_cast<const __m128i*>(source);
        __m128i* __xmmWordDestination   = reinterpret_cast<__m128i*>(destination);

        while (length--) {
            __SIMD_STL_REPEAT_N(4, _mm_stream_si128(__xmmWordDestination++, _mm_stream_load_si128(__xmmWordSource++)));
        }

        _mm_sfence();

        return destination;
    }

    template <>
    simd_stl_always_inline static void* CopyStreamAligned<128>(
        void*       destination,
        const void* source,
        sizetype    length) noexcept
    {
        const __m128i* __xmmWordSource  = reinterpret_cast<const __m128i*>(source);
        __m128i* __xmmWordDestination   = reinterpret_cast<__m128i*>(destination);

        while (length--) {
            __SIMD_STL_REPEAT_N(8, _mm_stream_si128(__xmmWordDestination++, _mm_stream_load_si128(__xmmWordSource++)));
        }

        _mm_sfence();

        return destination;
    }

    template <>
    simd_stl_always_inline static void* CopyStreamAligned<256>(
        void*       destination,
        const void* source,
        sizetype    length) noexcept
    {
        const __m128i* __xmmWordSource  = reinterpret_cast<const __m128i*>(source);
        __m128i* __xmmWordDestination   = reinterpret_cast<__m128i*>(destination);

        while (length--) {
            __SIMD_STL_REPEAT_N(16, _mm_stream_si128(__xmmWordDestination++, _mm_stream_load_si128(__xmmWordSource++)));
        }

        _mm_sfence();

        return destination;
    }
};

template <
    bool                _Aligned_,
    bool                _Streaming_,
    arch::CpuFeature    _SimdGeneration_>
struct _MemcpyVectorizedChooser {};


template <
    bool _Aligned_,
    bool _Streaming_>
struct _MemcpyVectorizedChooser<_Aligned_, _Streaming_, arch::CpuFeature::SSE2> {
    static_assert(!_Streaming_, "Streaming not supported for SSE2. ");

    simd_stl_always_inline void operator()(
        void*       destination, 
        const void* source,
        sizetype    bytes) noexcept 
    {
        using _VectorizedCopyImplementation_ = _CopyVectorized<arch::CpuFeature::SSE2, _Aligned_>;
        sizetype offset = 0;

        while (bytes)
        {
            if (bytes < 2)
            {
                _VectorizedCopyImplementation_::Copy<1>(destination, source, bytes);
                offset = bytes & -1;
                destination = static_cast<char*>(destination) + offset; 
                source = static_cast<const char*>(source) + offset; 
                bytes = 0;
            }
            else if (bytes < 4)
            {
                _VectorizedCopyImplementation_::Copy<2>(destination, source, bytes >> 1);
                __RECALCULATE_REMAINING(2)
            }
            else if (bytes < 8)
            {
                _VectorizedCopyImplementation_::Copy<4>(destination, source, bytes >> 2);
                __RECALCULATE_REMAINING(4)
            }
            else if (bytes < 16)
            {
                _VectorizedCopyImplementation_::Copy<8>(destination, source, bytes >> 3);
                __RECALCULATE_REMAINING(8)
            }
            else if (bytes < 32)
            {
                _VectorizedCopyImplementation_::Copy<16>(destination, source, bytes >> 4);
                __RECALCULATE_REMAINING(16)
            }
            else if (bytes < 64)
            {
                _VectorizedCopyImplementation_::Copy<32>(destination, source, bytes >> 5);
                __RECALCULATE_REMAINING(32)
            }
            else if (bytes < 128)
            {
                _VectorizedCopyImplementation_::Copy<64>(destination, source, bytes >> 6);
                __RECALCULATE_REMAINING(64)
            }
            else if (bytes < 256)
            {
                _VectorizedCopyImplementation_::Copy<128>(destination, source, bytes >> 7);
                __RECALCULATE_REMAINING(128)
            }
            else
            {
                _VectorizedCopyImplementation_::Copy<256>(destination, source, bytes >> 8);
                __RECALCULATE_REMAINING(256)
            }
        }
    }
};

template <
    bool _Aligned_,
    bool _Streaming_>
struct _MemcpyVectorizedChooser<_Aligned_, _Streaming_, arch::CpuFeature::SSE41> {
    static_assert(_Aligned_ >= _Streaming_, "Streaming loads/stores must be aligned. ");

    simd_stl_always_inline void operator()(
        void*       destination, 
        const void* source,
        sizetype    bytes) noexcept 
    {
        using _VectorizedCopyImplementation_ = _CopyVectorized<arch::CpuFeature::SSE41, _Aligned_>;
        sizetype offset = 0;

        while (bytes)
        {
            if (bytes < 2)
            {
                __DISPATCH_VECTORIZED_COPY(1, 0)
                offset = bytes & -1;
                destination = static_cast<char*>(destination) + offset;
                source = static_cast<const char*>(source) + offset;
                bytes = 0;
            }
            else if (bytes < 4)
            {
                __DISPATCH_VECTORIZED_COPY(2, 1)
                __RECALCULATE_REMAINING(2)
            }
            else if (bytes < 8)
            {
                __DISPATCH_VECTORIZED_COPY(4, 2)
                __RECALCULATE_REMAINING(4)
            }
            else if (bytes < 16)
            {
                __DISPATCH_VECTORIZED_COPY(8, 3)
                __RECALCULATE_REMAINING(8)
            }
            else if (bytes < 32)
            {
                __DISPATCH_VECTORIZED_COPY(16, 4)
                __RECALCULATE_REMAINING(16)
            }
            else if (bytes < 64)
            {
                __DISPATCH_VECTORIZED_COPY(32, 5)
                __RECALCULATE_REMAINING(32)
            }
            else if (bytes < 128)
            {
                __DISPATCH_VECTORIZED_COPY(64, 6)
                __RECALCULATE_REMAINING(64)
            }
            else if (bytes < 256)
            {
                __DISPATCH_VECTORIZED_COPY(128, 7)
                __RECALCULATE_REMAINING(128)
            }
            else
            {
                __DISPATCH_VECTORIZED_COPY(256, 8)
                __RECALCULATE_REMAINING(256)
            }
        }
    }
};

template <
    bool _Aligned_,
    bool _Streaming_>
struct _MemcpyVectorizedChooser<_Aligned_, _Streaming_, arch::CpuFeature::AVX> {
    static_assert(!_Streaming_, "Streaming not supported for AVX. ");

    simd_stl_always_inline void operator()(
        void*       destination, 
        const void* source,
        sizetype    bytes) noexcept 
    {
        using _VectorizedCopyImplementation_ = _CopyVectorized<arch::CpuFeature::AVX, _Aligned_>;
        sizetype offset = 0;

        while (bytes)
        {
            if (bytes < 2)
            {
                _VectorizedCopyImplementation_::Copy<1>(destination, source, bytes);
                offset = bytes & -1;
                destination = static_cast<char*>(destination) + offset;
                source = static_cast<const char*>(source) + offset;
                bytes = 0;
            }
            else if (bytes < 4)
            {
                _VectorizedCopyImplementation_::Copy<2>(destination, source, bytes >> 1);
                __RECALCULATE_REMAINING(2)
            }
            else if (bytes < 8)
            {
                _VectorizedCopyImplementation_::Copy<4>(destination, source, bytes >> 2);
                __RECALCULATE_REMAINING(4)
            }
            else if (bytes < 16)
            {
                _VectorizedCopyImplementation_::Copy<8>(destination, source, bytes >> 3);
                __RECALCULATE_REMAINING(8)
            }
            else if (bytes < 32)
            {
                _VectorizedCopyImplementation_::Copy<16>(destination, source, bytes >> 4);
                __RECALCULATE_REMAINING(16)
            }
            else if (bytes < 64)
            {
                _VectorizedCopyImplementation_::Copy<32>(destination, source, bytes >> 5);
                __RECALCULATE_REMAINING(32)
            }
            else if (bytes < 128)
            {
                _VectorizedCopyImplementation_::Copy<64>(destination, source, bytes >> 6);
                __RECALCULATE_REMAINING(64)
            }
            else if (bytes < 256)
            {
                _VectorizedCopyImplementation_::Copy<128>(destination, source, bytes >> 7);
                __RECALCULATE_REMAINING(128)
            }
            else if (bytes < 512)
            {
                _VectorizedCopyImplementation_::Copy<256>(destination, source, bytes >> 8);
                __RECALCULATE_REMAINING(256)
            }
            else
            {
                _VectorizedCopyImplementation_::Copy<512>(destination, source, bytes >> 9);
                __RECALCULATE_REMAINING(512)
            }
        }
    }
};

template <
    bool _Aligned_,
    bool _Streaming_>
struct _MemcpyVectorizedChooser<_Aligned_, _Streaming_, arch::CpuFeature::AVX2> {
    static_assert(_Aligned_ >= _Streaming_, "Streaming loads/stores must be aligned. ");

    simd_stl_always_inline void operator()(
        void*       destination, 
        const void* source,
        sizetype    bytes) noexcept 
    {
        using _VectorizedCopyImplementation_ = _CopyVectorized<arch::CpuFeature::AVX2, _Aligned_>;
        sizetype offset = 0;

        while (bytes)
        {
            if (bytes < 2)
            {
                __DISPATCH_VECTORIZED_COPY(1, 0)
                offset = bytes & -1;
                destination = static_cast<char*>(destination) + offset;
                source = static_cast<const char*>(source) + offset;
                bytes = 0;
            }
            else if (bytes < 4)
            {
                __DISPATCH_VECTORIZED_COPY(2, 1)
                __RECALCULATE_REMAINING(2)
            }
            else if (bytes < 8)
            {
                __DISPATCH_VECTORIZED_COPY(4, 2)
                __RECALCULATE_REMAINING(4)
            }
            else if (bytes < 16)
            {
                __DISPATCH_VECTORIZED_COPY(8, 3)
                __RECALCULATE_REMAINING(8)
            }
            else if (bytes < 32)
            {
                __DISPATCH_VECTORIZED_COPY(16, 4)
                __RECALCULATE_REMAINING(16)
            }
            else if (bytes < 64)
            {
                __DISPATCH_VECTORIZED_COPY(32, 5)
                __RECALCULATE_REMAINING(32)
            }
            else if (bytes < 128)
            {
                __DISPATCH_VECTORIZED_COPY(64, 6)
                __RECALCULATE_REMAINING(64)
            }
            else if (bytes < 256)
            {
                __DISPATCH_VECTORIZED_COPY(128, 7)
                __RECALCULATE_REMAINING(128)
            }
            else if (bytes < 512)
            {
                __DISPATCH_VECTORIZED_COPY(256, 8)
                __RECALCULATE_REMAINING(256)
            }
            else
            {
                __DISPATCH_VECTORIZED_COPY(512, 9)
                __RECALCULATE_REMAINING(512)
            }
        }
    }
};

template <
    bool _Aligned_,
    bool _Streaming_>
struct _MemcpyVectorizedChooser<_Aligned_, _Streaming_, arch::CpuFeature::AVX512F> {
    static_assert(_Aligned_ >= _Streaming_, "Streaming loads/stores must be aligned. ");

    simd_stl_always_inline void operator()(
        void*       destination, 
        const void* source,
        sizetype    bytes) noexcept 
    {
        using _VectorizedCopyImplementation_ = _CopyVectorized<arch::CpuFeature::AVX512F, _Aligned_>;
        sizetype offset = 0;

        while (bytes)
        {
            if (bytes < 2)
            {
                __DISPATCH_VECTORIZED_COPY(1, 0)
                offset = bytes & -1;
                destination = static_cast<char*>(destination) + offset; 
                source = static_cast<const char*>(source) + offset; 
                bytes = 0;
            }
            else if (bytes < 4)
            {
                __DISPATCH_VECTORIZED_COPY(2, 1)
                __RECALCULATE_REMAINING(2)
            }
            else if (bytes < 8)
            {
                __DISPATCH_VECTORIZED_COPY(4, 2)
                __RECALCULATE_REMAINING(4)
            }
            else if (bytes < 16)
            {
                __DISPATCH_VECTORIZED_COPY(8, 3)
                __RECALCULATE_REMAINING(8)
            }
            else if (bytes < 32)
            {
                __DISPATCH_VECTORIZED_COPY(16, 4)
                __RECALCULATE_REMAINING(16)
            }
            else if (bytes < 64)
            {
                __DISPATCH_VECTORIZED_COPY(32, 5)
                __RECALCULATE_REMAINING(32)
            }
            else if (bytes < 128)
            {
                __DISPATCH_VECTORIZED_COPY(64, 6)
                __RECALCULATE_REMAINING(64)
            }
            else if (bytes < 256)
            {
                __DISPATCH_VECTORIZED_COPY(128, 7)
                __RECALCULATE_REMAINING(128)
            }
            else if (bytes < 512)
            {
                __DISPATCH_VECTORIZED_COPY(256, 8)
                __RECALCULATE_REMAINING(256)
            }
            else if (bytes < 1024)
            {
                __DISPATCH_VECTORIZED_COPY(512, 9)
                __RECALCULATE_REMAINING(512)
            }
            else if (bytes < 2048)
            {
                __DISPATCH_VECTORIZED_COPY(1024, 10)
                __RECALCULATE_REMAINING(1024)
            }
            else if (bytes < 4096)
            {
                __DISPATCH_VECTORIZED_COPY(2048, 11)
                __RECALCULATE_REMAINING(2048)
            }
            else
            {
                __DISPATCH_VECTORIZED_COPY(4096, 12)
                __RECALCULATE_REMAINING(4096)
            }
        }
    }
};

template <
    bool _Aligned_,
    bool _Streaming_>
struct _MemcpyVectorizedChooser<_Aligned_, _Streaming_, arch::CpuFeature::None> {
    static_assert(!_Streaming_, "Streaming not supported. ");

    simd_stl_always_inline void operator()(
        void* destination,
        const void* source,
        sizetype    bytes) noexcept
    {
        using _VectorizedCopyImplementation_ = _CopyVectorized<arch::CpuFeature::None, _Aligned_>;
        sizetype offset = 0;

        while (bytes)
        {
            if (bytes < 2)
            {
                __DISPATCH_VECTORIZED_COPY(1, 0)
                offset = bytes & -1;
                destination = static_cast<char*>(destination) + offset; 
                source = static_cast<const char*>(source) + offset; 
                bytes = 0;
            }
            else if (bytes < 4)
            {
                __DISPATCH_VECTORIZED_COPY(2, 1)
                __RECALCULATE_REMAINING(2)
            }
            else if (bytes < 8)
            {
                __DISPATCH_VECTORIZED_COPY(4, 2)
                __RECALCULATE_REMAINING(4)
            }
            else {
                __DISPATCH_VECTORIZED_COPY(8, 3)
                __RECALCULATE_REMAINING(8)
            }
        }
    }
};

template <arch::CpuFeature _SimdGeneration_>
simd_stl_always_inline void* _MemcpyVectorizedInternal(
    void*       destination,
    const void* source,
    sizetype    bytes) noexcept
{
    using _SimdType_ = type_traits::__deduce_simd_vector_type<_SimdGeneration_, int>;

    if (memory::intersects(static_cast<char*>(destination), static_cast<char*>(destination) + bytes, static_cast<const char*>(source)))
        return _MemmoveVectorizedInternal<_SimdGeneration_>(destination, source, bytes);

    void* returnValue = destination;

    if (memory::isAligned(destination, sizeof(_SimdType_)) && memory::isAligned(source, sizeof(_SimdType_)))
    {
        if constexpr (type_traits::is_streaming_supported_v<_SimdGeneration_>) {
            if (bytes > __SIMD_STL_COPY_CACHE_SIZE_LIMIT) {
                _MemcpyVectorizedChooser<true, true, _SimdGeneration_>()(destination, source, bytes);
                return returnValue;
            }
        }
        
        _MemcpyVectorizedChooser<true, false, _SimdGeneration_>()(destination, source, bytes);
    }
    else
    {
        sizetype alignedBytes = (sizeof(_SimdType_)) - ((uintptr)destination & (sizeof(_SimdType_) - 1));

        if (bytes > alignedBytes) {
            void* destinationWithOffset     = static_cast<char*>(destination) + alignedBytes;
            const void* sourceWithOffset    = static_cast<const char*>(source) + alignedBytes;

            _MemcpyVectorizedChooser<false, false, _SimdGeneration_>()(destination, source, alignedBytes);
            _MemcpyVectorizedChooser<false, false, _SimdGeneration_>()(destinationWithOffset, sourceWithOffset, bytes - alignedBytes);
        }
        else
            _MemcpyVectorizedChooser<false, false, _SimdGeneration_>()(destination, source, bytes);
    }

    return returnValue;
}

template <>
simd_stl_always_inline void* _MemcpyVectorizedInternal<arch::CpuFeature::None>(
    void*       destination,
    const void* source,
    sizetype    bytes) noexcept
{
    _MemcpyVectorizedChooser<false, false, arch::CpuFeature::None>()(destination, source, bytes);
    return destination;
}

void* _MemcpyVectorized(
    void*       destination,
    const void* source,
    sizetype    bytes) noexcept
{
    if (arch::ProcessorFeatures::AVX512F())
        return _MemcpyVectorizedInternal<arch::CpuFeature::AVX512F>(destination, source, bytes);
    else if (arch::ProcessorFeatures::AVX2())
        return _MemcpyVectorizedInternal<arch::CpuFeature::AVX2>(destination, source, bytes);
    else if (arch::ProcessorFeatures::AVX())
        return _MemcpyVectorizedInternal<arch::CpuFeature::AVX>(destination, source, bytes);
    else if (arch::ProcessorFeatures::SSE41())
        return _MemcpyVectorizedInternal<arch::CpuFeature::SSE41>(destination, source, bytes);
    else if (arch::ProcessorFeatures::SSE2())
        return _MemcpyVectorizedInternal<arch::CpuFeature::SSE2>(destination, source, bytes);

    return _MemcpyVectorizedInternal<arch::CpuFeature::None>(destination, source, bytes);
}

__SIMD_STL_ALGORITHM_NAMESPACE_END

