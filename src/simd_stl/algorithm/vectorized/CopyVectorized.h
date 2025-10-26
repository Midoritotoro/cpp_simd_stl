#pragma once

#include <src/simd_stl/algorithm/AdvanceBytes.h>
#include <simd_stl/compatibility/SimdCompatibility.h>

#include <simd_stl/compatibility/FunctionAttributes.h>
#include <simd_stl/arch/ProcessorFeatures.h>

#define __SIMD_STL_COPY_CACHE_SIZE_LIMIT 3*1024*1024

__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <arch::CpuFeature _SimdGeneration_> 
struct _CopyVectorized;

template <>
struct _CopyVectorized<arch::CpuFeature::None> {
    template <sizetype    _ElementSize_>
    static void* CopyUnaligned(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        AssertUnreachable();
        return nullptr;
    }

    template <>
    static void* CopyUnaligned<1>(
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
    static void* CopyUnaligned<2>(
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
    static void* CopyUnaligned<4>(
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
    static void* CopyUnaligned<8>(
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
struct _CopyVectorized<arch::CpuFeature::SSE2> 
{
    // ==============================================================================
    //                                 ALIGNED
    // ==============================================================================

    template <sizetype _ElementSize_>
    static void* CopyAligned(
        void*       destination,
        const void* source,
        sizetype    length) noexcept
    {
        AssertUnreachable();
        return nullptr;
    }

    template <>
    static void* CopyAligned<16>(
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
    static void* CopyAligned<32>(
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
    static void* CopyAligned<64>(
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
    static void* CopyAligned<128>(
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
    static void* CopyAligned<256>(
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
    
    // ==============================================================================
    //                                UNALIGNED
    // ==============================================================================

    template <sizetype _ElementSize_>
    static void* CopyUnaligned(
        void*       destination,
        const void* source,
        sizetype    length) noexcept
    {
        AssertUnreachable();
        return nullptr;
    }

    template <>
    static void* CopyUnaligned<16>(
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
    static void* CopyUnaligned<32>(
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
    static void* CopyUnaligned<64>(
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
    static void* CopyUnaligned<128>(
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
    static void* CopyUnaligned<256>(
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
struct _CopyVectorized<arch::CpuFeature::AVX>
{
    // ==============================================================================
    //                                  ALIGNED
    // ==============================================================================
    
    template <sizetype _ElementSize_>
    static void* CopyAligned(
        void*       destination,
        const void* source,
        sizetype    length) noexcept
    {
        AssertUnreachable();
        return nullptr;
    }


    template <>
    static void* CopyAligned<32>(
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
    static void* CopyAligned<64>(
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
    static void* CopyAligned<128>(
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
    static void* CopyAligned<256>(
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
    static void* CopyAligned<512>(
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

    // ==============================================================================
    //                                  UNALIGNED
    // ==============================================================================

    template <sizetype _ElementSize_>
    static void* CopyUnaligned(
        void*       destination,
        const void* source,
        sizetype    length) noexcept
    {
        AssertUnreachable();
        return nullptr;
    }


    template <>
    static void* CopyUnaligned<32>(
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
    static void* CopyUnaligned<64>(
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
    static void* CopyUnaligned<128>(
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
    static void* CopyUnaligned<256>(
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
    static void* CopyUnaligned<512>(
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
struct _CopyVectorized<arch::CpuFeature::AVX512F>
{
    // ==============================================================================
    //                                  ALIGNED
    // ==============================================================================

    template <sizetype _ElementSize_>
    static void* CopyAligned(
        void*       destination,
        const void* source,
        sizetype    length) noexcept
    {
        AssertUnreachable();
        return nullptr;
    }


    template <>
    static void* CopyAligned<64>(
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
    static void* CopyAligned<128>(
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
    static void* CopyAligned<256>(
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
    static void* CopyAligned<512>(
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
    static void* CopyAligned<1024>(
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
    static void* CopyAligned<2048>(
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
    static void* CopyAligned<4096>(
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
    //                                  UNALIGNED
    // ==============================================================================

    template <sizetype _ElementSize_>
    static void* CopyUnaligned(
        void*       destination,
        const void* source,
        sizetype    length) noexcept
    {
        AssertUnreachable();
        return nullptr;
    }


    template <>
    static void* CopyUnaligned<64>(
        void*       destination,
        const void* source,
        sizetype    length) noexcept
    {
        const __m512i* __zmmWordSource  = reinterpret_cast<const __m512i*>(source);
        __m512i* __zmmWordDestination   = reinterpret_cast<__m512i*>(destination);

        while (length--)
            _mm512_storeu_si512(__zmmWordDestination++, _mm512_loadu_si512(__zmmWordSource++));

        return destination;
    }

    template <>
    static void* CopyUnaligned<128>(
        void*       destination,
        const void* source,
        sizetype    length) noexcept
    {
        const __m512i* __zmmWordSource   = reinterpret_cast<const __m512i*>(source);
        __m512i* __zmmWordDestination   = reinterpret_cast<__m512i*>(destination);

        while (length--) {
            __SIMD_STL_REPEAT_N(2, _mm512_storeu_si512(__zmmWordDestination++, _mm512_loadu_si512(__zmmWordSource++)));
        }

        return destination;
    }

    template <>
    static void* CopyUnaligned<256>(
        void*       destination,
        const void* source,
        sizetype    length) noexcept
    {
        const __m512i* __zmmWordSource   = reinterpret_cast<const __m512i*>(source);
        __m512i* __zmmWordDestination   = reinterpret_cast<__m512i*>(destination);

        while (length--) {
            __SIMD_STL_REPEAT_N(4, _mm512_storeu_si512(__zmmWordDestination++, _mm512_loadu_si512(__zmmWordSource++)));
        }

        return destination;
    }

    template <>
    static void* CopyUnaligned<512>(
        void*       destination,
        const void* source,
        sizetype    length) noexcept
    {
        const __m512i* __zmmWordSource  = reinterpret_cast<const __m512i*>(source);
        __m512i* __zmmWordDestination   = reinterpret_cast<__m512i*>(destination);

        while (length--) {
            __SIMD_STL_REPEAT_N(8, _mm512_storeu_si512(__zmmWordDestination++, _mm512_loadu_si512(__zmmWordSource++)));
        }

        return destination;
    }

    template <>
    static void* CopyUnaligned<1024>(
        void*       destination,
        const void* source,
        sizetype    length) noexcept
    {
        const __m512i* __zmmWordSource  = reinterpret_cast<const __m512i*>(source);
        __m512i* __zmmWordDestination   = reinterpret_cast<__m512i*>(destination);

        while (length--) {
            __SIMD_STL_REPEAT_N(16, _mm512_storeu_si512(__zmmWordDestination++, _mm512_loadu_si512(__zmmWordSource++)));
        }

        return destination;
    }

    template <>
    static void* CopyUnaligned<2048>(
        void*       destination,
        const void* source,
        sizetype    length) noexcept
    {
        const __m512i* __zmmWordSource  = reinterpret_cast<const __m512i*>(source);
        __m512i* __zmmWordDestination   = reinterpret_cast<__m512i*>(destination);

        while (length--) {
            __SIMD_STL_REPEAT_N(32, _mm512_storeu_si512(__zmmWordDestination++, _mm512_loadu_si512(__zmmWordSource++)));
        }

        return destination;
    }

    template <>
    static void* CopyUnaligned<4096>(
        void*       destination,
        const void* source,
        sizetype    length) noexcept
    {
        const __m512i* __zmmWordSource  = reinterpret_cast<const __m512i*>(source);
        __m512i* __zmmWordDestination   = reinterpret_cast<__m512i*>(destination);

        while (length--) {
            __SIMD_STL_REPEAT_N(64, _mm512_storeu_si512(__zmmWordDestination++, _mm512_loadu_si512(__zmmWordSource++)));
        }

        return destination;
    }

    // ==============================================================================
    //                              ALIGNED, STREAMING
    // ==============================================================================

    template <sizetype _ElementSize_>
    static void* CopyStreamAligned(
        void*       destination,
        const void* source,
        sizetype    length) noexcept
    {
        AssertUnreachable();
        return nullptr;
    }

    template <>
    static void* CopyStreamAligned<64>(
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
    static void* CopyStreamAligned<128>(
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
    static void* CopyStreamAligned<256>(
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
    static void* CopyStreamAligned<512>(
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
    static void* CopyStreamAligned<1024>(
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
    static void* CopyStreamAligned<2048>(
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
    static void* CopyStreamAligned<4096>(
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
struct _CopyVectorized<arch::CpuFeature::AVX2>:
    _CopyVectorized<arch::CpuFeature::AVX>
{
    // ==============================================================================
    //                              ALIGNED, STREAMING
    // ==============================================================================

    template <sizetype _ElementSize_>
    static void* CopyStreamAligned(
        void*       destination,
        const void* source,
        sizetype    length) noexcept
    {
        AssertUnreachable();
        return nullptr;
    }

    template <>
    static void* CopyStreamAligned<32>(
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
    static void* CopyStreamAligned<64>(
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
    static void* CopyStreamAligned<128>(
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
    static void* CopyStreamAligned<256>(
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
    static void* CopyStreamAligned<512>(
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


void* AVX_memcpy(void* dest, void* src, size_t numbytes)
{
    void* returnval = dest;

    if ((char*)src == (char*)dest)
    {
        return returnval;
    }

    if (
        (
            (
                (char*)dest > (char*)src
                )
            &&
            (
                (char*)dest < ((char*)src + numbytes)
                )
            )
        ||
        (
            (
                (char*)src > (char*)dest
                )
            &&
            (
                (char*)src < ((char*)dest + numbytes)
                )
            )
        ) // Why didn't you just use memmove directly???
    {
        returnval = AVX_memmove(dest, src, numbytes);
        return returnval;
    }

    if (
        (((uintptr)src & BYTE_ALIGNMENT) == 0)
        &&
        (((uintptr)dest & BYTE_ALIGNMENT) == 0)
        )
    {
        if (numbytes > __SIMD_STL_COPY_CACHE_SIZE_LIMIT)
        {
            memcpy_large_as(dest, src, numbytes);
        }
        else
        {
            memcpy_large_a(dest, src, numbytes);
        }
    }
    else
    {
        size_t numbytes_to_align = (BYTE_ALIGNMENT + 1) - ((uintptr_t)dest & BYTE_ALIGNMENT);

        if (numbytes > numbytes_to_align)
        {
            void* destoffset = (char*)dest + numbytes_to_align;
            void* srcoffset = (char*)src + numbytes_to_align;

            memcpy_large(dest, src, numbytes_to_align);
            memcpy_large(destoffset, srcoffset, numbytes - numbytes_to_align);
        }
        else
        {
            memcpy_large(dest, src, numbytes);
        }
    }

    return returnval;
}


__SIMD_STL_ALGORITHM_NAMESPACE_END
