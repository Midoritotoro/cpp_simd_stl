#pragma once

#include <src/simd_stl/algorithm/AdvanceBytes.h>
#include <simd_stl/compatibility/SimdCompatibility.h>

#include <simd_stl/compatibility/FunctionAttributes.h>
#include <simd_stl/arch/ProcessorFeatures.h>

#include <src/simd_stl/type_traits/SimdTypeCheck.h>
#include <simd_stl/numeric/BasicSimd.h>


#define __SIMD_STL_COPY_CACHE_SIZE_LIMIT 3*1024*1024

#if !defined(__DISPATCH_VECTORIZED_MOVE)
#  define __DISPATCH_VECTORIZED_MOVE(byteCount, shift) \
    if constexpr (_Streaming_)  {   \
        _VectorizedMoveImplementation_::MoveStreamAligned<byteCount>(destination, source, bytes >> shift); \
    }\
    else {  \
        _VectorizedMoveImplementation_::Move<byteCount>(destination, source, bytes >> shift); \
    }
#endif // !defined(__DISPATCH_VECTORIZED_MOVE)

#if !defined(__DISPATCH_VECTORIZED_REVERSED_MOVE)
#  define __DISPATCH_VECTORIZED_REVERSED_MOVE(byteCount, shift) \
    if constexpr (_Streaming_)  {   \
        _VectorizedMoveImplementation_::MoveStreamAligned<byteCount>(nextDestination, nextSource, 1); \
    }\
    else {  \
        _VectorizedMoveImplementation_::Move<byteCount>(nextDestination, nextSource, 1); \
    }
#endif // !defined(__DISPATCH_VECTORIZED_REVERSED_MOVE)

#if !defined(__RECALCULATE_REMAINING)
#  define __RECALCULATE_REMAINING(byteCount)       \
    offset = bytes & -byteCount;                            \
    destination = static_cast<char*>(destination) + offset; \
    source = static_cast<const char*>(source) + offset;     \
    bytes &= (byteCount - 1);
#endif // !defined(__RECALCULATE_REMAINING)

#if !defined(__REVERSED_MEMMOVE_CALL_WITH_DISPATCH)
#  define __REVERSED_MEMMOVE_CALL_WITH_DISPATCH(byteCount)                \
    offset = bytes & ((byteCount * 2) - 1);              \
    nextDestination = (char *)nextDestination - offset;     \
    nextSource = (char *)nextSource - offset;               \
    __DISPATCH_VECTORIZED_REVERSED_MOVE(byteCount, 0);      \
    bytes &= -(byteCount * 2);
#endif // !defined(__REVERSED_MEMMOVE_CALL_WITH_DISPATCH)

#if !defined(__REVERSED_MEMMOVE_CALL)
#  define __REVERSED_MEMMOVE_CALL(byteCount)                \
    offset = bytes & ((byteCount * 2) - 1);              \
    nextDestination = (char *)nextDestination - offset;     \
    nextSource = (char *)nextSource - offset;               \
    _VectorizedMoveImplementation_::Move<byteCount>(nextDestination, nextSource, 1); \
    bytes &= -(byteCount * 2);
#endif // !defined(__REVERSED_MEMMOVE_CALL)

__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <
    arch::CpuFeature    _SimdGeneration_,
    bool                _Aligned_>
struct _MoveVectorized;

template <bool _Aligned_>
struct _MoveVectorized<arch::CpuFeature::None, _Aligned_> {
    template <sizetype    _ElementSize_>
    simd_stl_always_inline static void* Move(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        AssertUnreachable();
        return nullptr;
    }

    template <>
    simd_stl_always_inline static void* Move<1>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const uint8* __sourceByte   = static_cast<const uint8*>(source);
        uint8* __destinationByte    = static_cast<uint8*>(destination);

        const uint8* nextSource = __sourceByte + length;
        uint8* nextDestination  = __destinationByte + length;

        if (__destinationByte < __sourceByte)
            while (__destinationByte != nextDestination)
                *__destinationByte++ = *__sourceByte++;
        else
            while (nextDestination != __destinationByte)
                *--nextDestination = *--nextSource;

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Move<2>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const uint16* __sourceWord   = static_cast<const uint16*>(source);
        uint16* __destinationWord    = static_cast<uint16*>(destination);

        const uint16* nextSource = __sourceWord + length;
        uint16* nextDestination  = __destinationWord + length;

        if (__destinationWord < __sourceWord)
            while (__destinationWord != nextDestination)
                *__destinationWord++ = *__sourceWord++;
        else
            while (nextDestination != __destinationWord)
                *--nextDestination = *--nextSource;

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Move<4>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const uint32* __sourceDWord   = static_cast<const uint32*>(source);
        uint32* __destinationDWord    = static_cast<uint32*>(destination);

        const uint32* nextSource = __sourceDWord + length;
        uint32* nextDestination  = __destinationDWord + length;

        if (__destinationDWord < __sourceDWord)
            while (__destinationDWord != nextDestination)
                *__destinationDWord++ = *__sourceDWord++;
        else
            while (nextDestination != __destinationDWord)
                *--nextDestination = *--nextSource;

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Move<8>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const uint64* __sourceQWord   = static_cast<const uint64*>(source);
        uint64* __destinationQWord    = static_cast<uint64*>(destination);

        const uint64* nextSource = __sourceQWord + length;
        uint64* nextDestination  = __destinationQWord + length;

        if (__destinationQWord < __sourceQWord)
            while (__destinationQWord != nextDestination)
                *__destinationQWord++ = *__sourceQWord++;
        else
            while (nextDestination != __destinationQWord)
                *--nextDestination = *--nextSource;

        return destination;
    }
};

template <>
struct _MoveVectorized<arch::CpuFeature::SSE2, false>:
    _MoveVectorized<arch::CpuFeature::None, false> 
{
    template <sizetype    _ElementSize_>
    simd_stl_always_inline static void* Move(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        AssertUnreachable();
        return nullptr;
    }

    template <>
    simd_stl_always_inline static void* Move<1>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const uint8* __sourceByte   = static_cast<const uint8*>(source);
        uint8* __destinationByte    = static_cast<uint8*>(destination);

        const uint8* nextSource = __sourceByte + length;
        uint8* nextDestination  = __destinationByte + length;

        if (__destinationByte < __sourceByte)
            while (__destinationByte != nextDestination)
                *__destinationByte++ = *__sourceByte++;
        else
            while (nextDestination != __destinationByte)
                *--nextDestination = *--nextSource;

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Move<2>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const uint16* __sourceWord   = static_cast<const uint16*>(source);
        uint16* __destinationWord    = static_cast<uint16*>(destination);

        const uint16* nextSource = __sourceWord + length;
        uint16* nextDestination  = __destinationWord + length;

        if (__destinationWord < __sourceWord)
            while (__destinationWord != nextDestination)
                *__destinationWord++ = *__sourceWord++;
        else
            while (nextDestination != __destinationWord)
                *--nextDestination = *--nextSource;

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Move<4>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const uint32* __sourceDWord   = static_cast<const uint32*>(source);
        uint32* __destinationDWord    = static_cast<uint32*>(destination);

        const uint32* nextSource = __sourceDWord + length;
        uint32* nextDestination  = __destinationDWord + length;

        if (__destinationDWord < __sourceDWord)
            while (__destinationDWord != nextDestination)
                *__destinationDWord++ = *__sourceDWord++;
        else
            while (nextDestination != __destinationDWord)
                *--nextDestination = *--nextSource;

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Move<8>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const uint64* __sourceQWord   = static_cast<const uint64*>(source);
        uint64* __destinationQWord    = static_cast<uint64*>(destination);

        const uint64* nextSource = __sourceQWord + length;
        uint64* nextDestination  = __destinationQWord + length;

        if (__destinationQWord < __sourceQWord)
            while (__destinationQWord != nextDestination)
                *__destinationQWord++ = *__sourceQWord++;
        else
            while (nextDestination != __destinationQWord)
                *--nextDestination = *--nextSource;

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Move<16>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const __m128i* __xmmWordSource  = reinterpret_cast<const __m128i*>(source);
        __m128i* __xmmWordDestination   = reinterpret_cast<__m128i*>(destination);

        const __m128i* nextSource   = __xmmWordSource + length;
        __m128i* nextDestination    = __xmmWordDestination + length;

        if (__xmmWordDestination < __xmmWordSource)
            while (__xmmWordDestination != nextDestination)
                _mm_storeu_si128(__xmmWordDestination++, _mm_loadu_si128(__xmmWordSource++));
        else
            while (nextDestination != __xmmWordDestination)
                _mm_storeu_si128(--nextDestination, _mm_loadu_si128(--nextSource));

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Move<32>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const __m128i* __xmmWordSource  = reinterpret_cast<const __m128i*>(source);
        __m128i* __xmmWordDestination   = reinterpret_cast<__m128i*>(destination);

        const __m128i* nextSource   = __xmmWordSource + (length << 1);
        __m128i* nextDestination    = __xmmWordDestination + (length << 1);

        if (__xmmWordDestination < __xmmWordSource) {
            while (__xmmWordDestination != nextDestination) {
                __SIMD_STL_REPEAT_N(2, _mm_storeu_si128(__xmmWordDestination++, _mm_loadu_si128(__xmmWordSource++)));
            }
        }
        else {
            while (nextDestination != __xmmWordDestination) {
                __SIMD_STL_REPEAT_N(2, _mm_storeu_si128(--nextDestination, _mm_loadu_si128(--nextSource)));
            }
        }

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Move<64>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const __m128i* __xmmWordSource  = reinterpret_cast<const __m128i*>(source);
        __m128i* __xmmWordDestination   = reinterpret_cast<__m128i*>(destination);

        const __m128i* nextSource   = __xmmWordSource + (length << 2);
        __m128i* nextDestination    = __xmmWordDestination + (length << 2);

        if (__xmmWordDestination < __xmmWordSource) {
            while (__xmmWordDestination != nextDestination) {
                __SIMD_STL_REPEAT_N(4, _mm_storeu_si128(__xmmWordDestination++, _mm_loadu_si128(__xmmWordSource++)));
            }
        }
        else {
            while (nextDestination != __xmmWordDestination) {
                __SIMD_STL_REPEAT_N(4, _mm_storeu_si128(--nextDestination, _mm_loadu_si128(--nextSource)));
            }
        }

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Move<128>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const __m128i* __xmmWordSource  = reinterpret_cast<const __m128i*>(source);
        __m128i* __xmmWordDestination   = reinterpret_cast<__m128i*>(destination);

        const __m128i* nextSource   = __xmmWordSource + (length << 3);
        __m128i* nextDestination    = __xmmWordDestination + (length << 3);

        if (__xmmWordDestination < __xmmWordSource) {
            while (__xmmWordDestination != nextDestination) {
                __SIMD_STL_REPEAT_N(8, _mm_storeu_si128(__xmmWordDestination++, _mm_loadu_si128(__xmmWordSource++)));
            }
        }
        else {
            while (nextDestination != __xmmWordDestination) {
                __SIMD_STL_REPEAT_N(8, _mm_storeu_si128(--nextDestination, _mm_loadu_si128(--nextSource)));
            }
        }

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Move<256>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const __m128i* __xmmWordSource  = reinterpret_cast<const __m128i*>(source);
        __m128i* __xmmWordDestination   = reinterpret_cast<__m128i*>(destination);

        const __m128i* nextSource   = __xmmWordSource + (length << 4);
        __m128i* nextDestination    = __xmmWordDestination + (length << 4);

        if (__xmmWordDestination < __xmmWordSource) {
            while (__xmmWordDestination != nextDestination) {
                __SIMD_STL_REPEAT_N(16, _mm_storeu_si128(__xmmWordDestination++, _mm_loadu_si128(__xmmWordSource++)));
            }
        }
        else {
            while (nextDestination != __xmmWordDestination) {
                __SIMD_STL_REPEAT_N(16, _mm_storeu_si128(--nextDestination, _mm_loadu_si128(--nextSource)));
            }
        }

        return destination;
    }
};

template <>
struct _MoveVectorized<arch::CpuFeature::SSE2, true>:
    _MoveVectorized<arch::CpuFeature::None, true> 
{
    template <sizetype    _ElementSize_>
    simd_stl_always_inline static void* Move(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        AssertUnreachable();
        return nullptr;
    }

        template <>
    simd_stl_always_inline static void* Move<1>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const uint8* __sourceByte   = static_cast<const uint8*>(source);
        uint8* __destinationByte    = static_cast<uint8*>(destination);

        const uint8* nextSource = __sourceByte + length;
        uint8* nextDestination  = __destinationByte + length;

        if (__destinationByte < __sourceByte)
            while (__destinationByte != nextDestination)
                *__destinationByte++ = *__sourceByte++;
        else
            while (nextDestination != __destinationByte)
                *--nextDestination = *--nextSource;

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Move<2>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const uint16* __sourceWord   = static_cast<const uint16*>(source);
        uint16* __destinationWord    = static_cast<uint16*>(destination);

        const uint16* nextSource = __sourceWord + length;
        uint16* nextDestination  = __destinationWord + length;

        if (__destinationWord < __sourceWord)
            while (__destinationWord != nextDestination)
                *__destinationWord++ = *__sourceWord++;
        else
            while (nextDestination != __destinationWord)
                *--nextDestination = *--nextSource;

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Move<4>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const uint32* __sourceDWord   = static_cast<const uint32*>(source);
        uint32* __destinationDWord    = static_cast<uint32*>(destination);

        const uint32* nextSource = __sourceDWord + length;
        uint32* nextDestination  = __destinationDWord + length;

        if (__destinationDWord < __sourceDWord)
            while (__destinationDWord != nextDestination)
                *__destinationDWord++ = *__sourceDWord++;
        else
            while (nextDestination != __destinationDWord)
                *--nextDestination = *--nextSource;

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Move<8>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const uint64* __sourceQWord   = static_cast<const uint64*>(source);
        uint64* __destinationQWord    = static_cast<uint64*>(destination);

        const uint64* nextSource = __sourceQWord + length;
        uint64* nextDestination  = __destinationQWord + length;

        if (__destinationQWord < __sourceQWord)
            while (__destinationQWord != nextDestination)
                *__destinationQWord++ = *__sourceQWord++;
        else
            while (nextDestination != __destinationQWord)
                *--nextDestination = *--nextSource;

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Move<16>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const __m128i* __xmmWordSource  = reinterpret_cast<const __m128i*>(source);
        __m128i* __xmmWordDestination   = reinterpret_cast<__m128i*>(destination);

        const __m128i* nextSource   = __xmmWordSource + length;
        __m128i* nextDestination    = __xmmWordDestination + length;

        if (__xmmWordDestination < __xmmWordSource)
            while (__xmmWordDestination != nextDestination)
                _mm_store_si128(__xmmWordDestination++, _mm_load_si128(__xmmWordSource++));
        else
            while (nextDestination != __xmmWordDestination)
                _mm_store_si128(--nextDestination, _mm_load_si128(--nextSource));

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Move<32>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const __m128i* __xmmWordSource  = reinterpret_cast<const __m128i*>(source);
        __m128i* __xmmWordDestination   = reinterpret_cast<__m128i*>(destination);

        const __m128i* nextSource   = __xmmWordSource + (length << 1);
        __m128i* nextDestination    = __xmmWordDestination + (length << 1);

        if (__xmmWordDestination < __xmmWordSource) {
            while (__xmmWordDestination != nextDestination) {
                __SIMD_STL_REPEAT_N(2, _mm_store_si128(__xmmWordDestination++, _mm_load_si128(__xmmWordSource++)));
            }
        }
        else {
            while (nextDestination != __xmmWordDestination) {
                __SIMD_STL_REPEAT_N(2, _mm_store_si128(--nextDestination, _mm_load_si128(--nextSource)));
            }
        }

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Move<64>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const __m128i* __xmmWordSource  = reinterpret_cast<const __m128i*>(source);
        __m128i* __xmmWordDestination   = reinterpret_cast<__m128i*>(destination);

        const __m128i* nextSource   = __xmmWordSource + (length << 2);
        __m128i* nextDestination    = __xmmWordDestination + (length << 2);

        if (__xmmWordDestination < __xmmWordSource) {
            while (__xmmWordDestination != nextDestination) {
                __SIMD_STL_REPEAT_N(4, _mm_store_si128(__xmmWordDestination++, _mm_load_si128(__xmmWordSource++)));
            }
        }
        else {
            while (nextDestination != __xmmWordDestination) {
                __SIMD_STL_REPEAT_N(4, _mm_store_si128(--nextDestination, _mm_load_si128(--nextSource)));
            }
        }

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Move<128>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const __m128i* __xmmWordSource  = reinterpret_cast<const __m128i*>(source);
        __m128i* __xmmWordDestination   = reinterpret_cast<__m128i*>(destination);

        const __m128i* nextSource   = __xmmWordSource + (length << 3);
        __m128i* nextDestination    = __xmmWordDestination + (length << 3);

        if (__xmmWordDestination < __xmmWordSource) {
            while (__xmmWordDestination != nextDestination) {
                __SIMD_STL_REPEAT_N(8, _mm_store_si128(__xmmWordDestination++, _mm_load_si128(__xmmWordSource++)));
            }
        }
        else {
            while (nextDestination != __xmmWordDestination) {
                __SIMD_STL_REPEAT_N(8, _mm_store_si128(--nextDestination, _mm_load_si128(--nextSource)));
            }
        }

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Move<256>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const __m128i* __xmmWordSource  = reinterpret_cast<const __m128i*>(source);
        __m128i* __xmmWordDestination   = reinterpret_cast<__m128i*>(destination);

        const __m128i* nextSource   = __xmmWordSource + (length << 4);
        __m128i* nextDestination    = __xmmWordDestination + (length << 4);

        if (__xmmWordDestination < __xmmWordSource) {
            while (__xmmWordDestination != nextDestination) {
                __SIMD_STL_REPEAT_N(16, _mm_store_si128(__xmmWordDestination++, _mm_load_si128(__xmmWordSource++)));
            }
        }
        else {
            while (nextDestination != __xmmWordDestination) {
                __SIMD_STL_REPEAT_N(16, _mm_store_si128(--nextDestination, _mm_load_si128(--nextSource)));
            }
        }

        return destination;
    }
};

template <>
struct _MoveVectorized<arch::CpuFeature::AVX, true>:
    _MoveVectorized<arch::CpuFeature::None, true> 
{
    template <sizetype    _ElementSize_>
    simd_stl_always_inline static void* Move(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        AssertUnreachable();
        return nullptr;
    }

        template <>
    simd_stl_always_inline static void* Move<1>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const uint8* __sourceByte   = static_cast<const uint8*>(source);
        uint8* __destinationByte    = static_cast<uint8*>(destination);

        const uint8* nextSource = __sourceByte + length;
        uint8* nextDestination  = __destinationByte + length;

        if (__destinationByte < __sourceByte)
            while (__destinationByte != nextDestination)
                *__destinationByte++ = *__sourceByte++;
        else
            while (nextDestination != __destinationByte)
                *--nextDestination = *--nextSource;

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Move<2>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const uint16* __sourceWord   = static_cast<const uint16*>(source);
        uint16* __destinationWord    = static_cast<uint16*>(destination);

        const uint16* nextSource = __sourceWord + length;
        uint16* nextDestination  = __destinationWord + length;

        if (__destinationWord < __sourceWord)
            while (__destinationWord != nextDestination)
                *__destinationWord++ = *__sourceWord++;
        else
            while (nextDestination != __destinationWord)
                *--nextDestination = *--nextSource;

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Move<4>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const uint32* __sourceDWord   = static_cast<const uint32*>(source);
        uint32* __destinationDWord    = static_cast<uint32*>(destination);

        const uint32* nextSource = __sourceDWord + length;
        uint32* nextDestination  = __destinationDWord + length;

        if (__destinationDWord < __sourceDWord)
            while (__destinationDWord != nextDestination)
                *__destinationDWord++ = *__sourceDWord++;
        else
            while (nextDestination != __destinationDWord)
                *--nextDestination = *--nextSource;

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Move<8>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const uint64* __sourceQWord   = static_cast<const uint64*>(source);
        uint64* __destinationQWord    = static_cast<uint64*>(destination);

        const uint64* nextSource = __sourceQWord + length;
        uint64* nextDestination  = __destinationQWord + length;

        if (__destinationQWord < __sourceQWord)
            while (__destinationQWord != nextDestination)
                *__destinationQWord++ = *__sourceQWord++;
        else
            while (nextDestination != __destinationQWord)
                *--nextDestination = *--nextSource;

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Move<16>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const __m128i* __xmmWordSource  = reinterpret_cast<const __m128i*>(source);
        __m128i* __xmmWordDestination   = reinterpret_cast<__m128i*>(destination);

        const __m128i* nextSource   = __xmmWordSource + length;
        __m128i* nextDestination    = __xmmWordDestination + length;

        if (__xmmWordDestination < __xmmWordSource)
            while (__xmmWordDestination != nextDestination)
                _mm_store_si128(__xmmWordDestination++, _mm_load_si128(__xmmWordSource++));
        else
            while (nextDestination != __xmmWordDestination)
                _mm_store_si128(--nextDestination, _mm_load_si128(--nextSource));

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Move<32>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const __m256i* __ymmWordSource  = reinterpret_cast<const __m256i*>(source);
        __m256i* __ymmWordDestination   = reinterpret_cast<__m256i*>(destination);

        const __m256i* nextSource   = __ymmWordSource + length;
        __m256i* nextDestination    = __ymmWordDestination + length;

        if (__ymmWordDestination < __ymmWordSource)
            while (__ymmWordDestination != nextDestination)
                _mm256_store_si256(__ymmWordDestination++, _mm256_load_si256(__ymmWordSource++));
        else
            while (nextDestination != __ymmWordDestination)
                _mm256_store_si256(--nextDestination, _mm256_load_si256(--nextSource));

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Move<64>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const __m256i* __ymmWordSource  = reinterpret_cast<const __m256i*>(source);
        __m256i* __ymmWordDestination   = reinterpret_cast<__m256i*>(destination);

        const __m256i* nextSource   = __ymmWordSource + (length << 1);
        __m256i* nextDestination    = __ymmWordDestination + (length << 1);

        if (__ymmWordDestination < __ymmWordSource) {
            while (__ymmWordDestination != nextDestination) {
                __SIMD_STL_REPEAT_N(2, _mm256_store_si256(__ymmWordDestination++, _mm256_load_si256(__ymmWordSource++)));
            }
        }
        else {
            while (nextDestination != __ymmWordDestination) {
                __SIMD_STL_REPEAT_N(2, _mm256_store_si256(--nextDestination, _mm256_load_si256(--nextSource)));
            }
        }

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Move<128>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const __m256i* __ymmWordSource  = reinterpret_cast<const __m256i*>(source);
        __m256i* __ymmWordDestination   = reinterpret_cast<__m256i*>(destination);

        const __m256i* nextSource   = __ymmWordSource + (length << 2);
        __m256i* nextDestination    = __ymmWordDestination + (length << 2);

        if (__ymmWordDestination < __ymmWordSource) {
            while (__ymmWordDestination != nextDestination) {
                __SIMD_STL_REPEAT_N(4, _mm256_store_si256(__ymmWordDestination++, _mm256_load_si256(__ymmWordSource++)));
            }
        }
        else {
            while (nextDestination != __ymmWordDestination) {
                __SIMD_STL_REPEAT_N(4, _mm256_store_si256(--nextDestination, _mm256_load_si256(--nextSource)));
            }
        }

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Move<256>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const __m256i* __ymmWordSource  = reinterpret_cast<const __m256i*>(source);
        __m256i* __ymmWordDestination   = reinterpret_cast<__m256i*>(destination);

        const __m256i* nextSource   = __ymmWordSource + (length << 3);
        __m256i* nextDestination    = __ymmWordDestination + (length << 3);

        if (__ymmWordDestination < __ymmWordSource) {
            while (__ymmWordDestination != nextDestination) {
                __SIMD_STL_REPEAT_N(8, _mm256_store_si256(__ymmWordDestination++, _mm256_load_si256(__ymmWordSource++)));
            }
        }
        else {
            while (nextDestination != __ymmWordDestination) {
                __SIMD_STL_REPEAT_N(8, _mm256_store_si256(--nextDestination, _mm256_load_si256(--nextSource)));
            }
        }

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Move<512>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const __m256i* __ymmWordSource  = reinterpret_cast<const __m256i*>(source);
        __m256i* __ymmWordDestination   = reinterpret_cast<__m256i*>(destination);

        const __m256i* nextSource   = __ymmWordSource + (length << 4);
        __m256i* nextDestination    = __ymmWordDestination + (length << 4);

        if (__ymmWordDestination < __ymmWordSource) {
            while (__ymmWordDestination != nextDestination) {
                __SIMD_STL_REPEAT_N(16, _mm256_store_si256(__ymmWordDestination++, _mm256_load_si256(__ymmWordSource++)));
            }
        }
        else {
            while (nextDestination != __ymmWordDestination) {
                __SIMD_STL_REPEAT_N(16, _mm256_store_si256(--nextDestination, _mm256_load_si256(--nextSource)));
            }
        }

        return destination;
    }
};

template <>
struct _MoveVectorized<arch::CpuFeature::AVX, false>:
    _MoveVectorized<arch::CpuFeature::None, true> 
{
    template <sizetype    _ElementSize_>
    simd_stl_always_inline static void* Move(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        AssertUnreachable();
        return nullptr;
    }

    template <>
    simd_stl_always_inline static void* Move<1>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const uint8* __sourceByte   = static_cast<const uint8*>(source);
        uint8* __destinationByte    = static_cast<uint8*>(destination);

        const uint8* nextSource = __sourceByte + length;
        uint8* nextDestination  = __destinationByte + length;

        if (__destinationByte < __sourceByte)
            while (__destinationByte != nextDestination)
                *__destinationByte++ = *__sourceByte++;
        else
            while (nextDestination != __destinationByte)
                *--nextDestination = *--nextSource;

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Move<2>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const uint16* __sourceWord   = static_cast<const uint16*>(source);
        uint16* __destinationWord    = static_cast<uint16*>(destination);

        const uint16* nextSource = __sourceWord + length;
        uint16* nextDestination  = __destinationWord + length;

        if (__destinationWord < __sourceWord)
            while (__destinationWord != nextDestination)
                *__destinationWord++ = *__sourceWord++;
        else
            while (nextDestination != __destinationWord)
                *--nextDestination = *--nextSource;

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Move<4>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const uint32* __sourceDWord   = static_cast<const uint32*>(source);
        uint32* __destinationDWord    = static_cast<uint32*>(destination);

        const uint32* nextSource = __sourceDWord + length;
        uint32* nextDestination  = __destinationDWord + length;

        if (__destinationDWord < __sourceDWord)
            while (__destinationDWord != nextDestination)
                *__destinationDWord++ = *__sourceDWord++;
        else
            while (nextDestination != __destinationDWord)
                *--nextDestination = *--nextSource;

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Move<8>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const uint64* __sourceQWord   = static_cast<const uint64*>(source);
        uint64* __destinationQWord    = static_cast<uint64*>(destination);

        const uint64* nextSource = __sourceQWord + length;
        uint64* nextDestination  = __destinationQWord + length;

        if (__destinationQWord < __sourceQWord)
            while (__destinationQWord != nextDestination)
                *__destinationQWord++ = *__sourceQWord++;
        else
            while (nextDestination != __destinationQWord)
                *--nextDestination = *--nextSource;

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Move<16>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const __m128i* __xmmWordSource  = reinterpret_cast<const __m128i*>(source);
        __m128i* __xmmWordDestination   = reinterpret_cast<__m128i*>(destination);

        const __m128i* nextSource   = __xmmWordSource + length;
        __m128i* nextDestination    = __xmmWordDestination + length;

        if (__xmmWordDestination < __xmmWordSource)
            while (__xmmWordDestination != nextDestination)
                _mm_storeu_si128(__xmmWordDestination++, _mm_loadu_si128(__xmmWordSource++));
        else
            while (nextDestination != __xmmWordDestination)
                _mm_storeu_si128(--nextDestination, _mm_loadu_si128(--nextSource));

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Move<32>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const __m256i* __ymmWordSource  = reinterpret_cast<const __m256i*>(source);
        __m256i* __ymmWordDestination   = reinterpret_cast<__m256i*>(destination);

        const __m256i* nextSource   = __ymmWordSource + length;
        __m256i* nextDestination    = __ymmWordDestination + length;

        if (__ymmWordDestination < __ymmWordSource)
            while (__ymmWordDestination != nextDestination)
                _mm256_storeu_si256(__ymmWordDestination++, _mm256_lddqu_si256(__ymmWordSource++));
        else
            while (nextDestination != __ymmWordDestination)
                _mm256_storeu_si256(--nextDestination, _mm256_lddqu_si256(--nextSource));

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Move<64>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const __m256i* __ymmWordSource  = reinterpret_cast<const __m256i*>(source);
        __m256i* __ymmWordDestination   = reinterpret_cast<__m256i*>(destination);

        const __m256i* nextSource   = __ymmWordSource + (length << 1);
        __m256i* nextDestination    = __ymmWordDestination + (length << 1);

        if (__ymmWordDestination < __ymmWordSource) {
            while (__ymmWordDestination != nextDestination) {
                __SIMD_STL_REPEAT_N(2, _mm256_storeu_si256(__ymmWordDestination++, _mm256_lddqu_si256(__ymmWordSource++)));
            }
        }
        else {
            while (nextDestination != __ymmWordDestination) {
                __SIMD_STL_REPEAT_N(2, _mm256_storeu_si256(--nextDestination, _mm256_lddqu_si256(--nextSource)));
            }
        }

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Move<128>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const __m256i* __ymmWordSource  = reinterpret_cast<const __m256i*>(source);
        __m256i* __ymmWordDestination   = reinterpret_cast<__m256i*>(destination);

        const __m256i* nextSource   = __ymmWordSource + (length << 2);
        __m256i* nextDestination    = __ymmWordDestination + (length << 2);

        if (__ymmWordDestination < __ymmWordSource) {
            while (__ymmWordDestination != nextDestination) {
                __SIMD_STL_REPEAT_N(4, _mm256_storeu_si256(__ymmWordDestination++, _mm256_lddqu_si256(__ymmWordSource++)));
            }
        }
        else {
            while (nextDestination != __ymmWordDestination) {
                __SIMD_STL_REPEAT_N(4, _mm256_storeu_si256(--nextDestination, _mm256_lddqu_si256(--nextSource)));
            }
        }

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Move<256>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const __m256i* __ymmWordSource  = reinterpret_cast<const __m256i*>(source);
        __m256i* __ymmWordDestination   = reinterpret_cast<__m256i*>(destination);

        const __m256i* nextSource   = __ymmWordSource + (length << 3);
        __m256i* nextDestination    = __ymmWordDestination + (length << 3);

        if (__ymmWordDestination < __ymmWordSource) {
            while (__ymmWordDestination != nextDestination) {
                __SIMD_STL_REPEAT_N(8, _mm256_storeu_si256(__ymmWordDestination++, _mm256_lddqu_si256(__ymmWordSource++)));
            }
        }
        else {
            while (nextDestination != __ymmWordDestination) {
                __SIMD_STL_REPEAT_N(8, _mm256_storeu_si256(--nextDestination, _mm256_lddqu_si256(--nextSource)));
            }
        }

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Move<512>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const __m256i* __ymmWordSource  = reinterpret_cast<const __m256i*>(source);
        __m256i* __ymmWordDestination   = reinterpret_cast<__m256i*>(destination);

        const __m256i* nextSource   = __ymmWordSource + (length << 4);
        __m256i* nextDestination    = __ymmWordDestination + (length << 4);

        if (__ymmWordDestination < __ymmWordSource) {
            while (__ymmWordDestination != nextDestination) {
                __SIMD_STL_REPEAT_N(16, _mm256_storeu_si256(__ymmWordDestination++, _mm256_lddqu_si256(__ymmWordSource++)));
            }
        }
        else {
            while (nextDestination != __ymmWordDestination) {
                __SIMD_STL_REPEAT_N(16, _mm256_storeu_si256(--nextDestination, _mm256_lddqu_si256(--nextSource)));
            }
        }

        return destination;
    }
};

template <>
struct _MoveVectorized<arch::CpuFeature::AVX512F, false> :
    _MoveVectorized<arch::CpuFeature::None, false>
{
    template <sizetype    _ElementSize_>
    simd_stl_always_inline static void* Move(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        AssertUnreachable();
        return nullptr;
    }

        template <>
    simd_stl_always_inline static void* Move<1>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const uint8* __sourceByte   = static_cast<const uint8*>(source);
        uint8* __destinationByte    = static_cast<uint8*>(destination);

        const uint8* nextSource = __sourceByte + length;
        uint8* nextDestination  = __destinationByte + length;

        if (__destinationByte < __sourceByte)
            while (__destinationByte != nextDestination)
                *__destinationByte++ = *__sourceByte++;
        else
            while (nextDestination != __destinationByte)
                *--nextDestination = *--nextSource;

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Move<2>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const uint16* __sourceWord   = static_cast<const uint16*>(source);
        uint16* __destinationWord    = static_cast<uint16*>(destination);

        const uint16* nextSource = __sourceWord + length;
        uint16* nextDestination  = __destinationWord + length;

        if (__destinationWord < __sourceWord)
            while (__destinationWord != nextDestination)
                *__destinationWord++ = *__sourceWord++;
        else
            while (nextDestination != __destinationWord)
                *--nextDestination = *--nextSource;

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Move<4>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const uint32* __sourceDWord   = static_cast<const uint32*>(source);
        uint32* __destinationDWord    = static_cast<uint32*>(destination);

        const uint32* nextSource = __sourceDWord + length;
        uint32* nextDestination  = __destinationDWord + length;

        if (__destinationDWord < __sourceDWord)
            while (__destinationDWord != nextDestination)
                *__destinationDWord++ = *__sourceDWord++;
        else
            while (nextDestination != __destinationDWord)
                *--nextDestination = *--nextSource;

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Move<8>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const uint64* __sourceQWord   = static_cast<const uint64*>(source);
        uint64* __destinationQWord    = static_cast<uint64*>(destination);

        const uint64* nextSource = __sourceQWord + length;
        uint64* nextDestination  = __destinationQWord + length;

        if (__destinationQWord < __sourceQWord)
            while (__destinationQWord != nextDestination)
                *__destinationQWord++ = *__sourceQWord++;
        else
            while (nextDestination != __destinationQWord)
                *--nextDestination = *--nextSource;

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Move<16>(
        void* destination,
        const void* source,
        sizetype    length) noexcept
    {
        const __m128i* __xmmWordSource = reinterpret_cast<const __m128i*>(source);
        __m128i* __xmmWordDestination = reinterpret_cast<__m128i*>(destination);

        const __m128i* nextSource = __xmmWordSource + length;
        __m128i* nextDestination = __xmmWordDestination + length;

        if (__xmmWordDestination < __xmmWordSource)
            while (__xmmWordDestination != nextDestination)
                _mm_storeu_si128(__xmmWordDestination++, _mm_loadu_si128(__xmmWordSource++));
        else
            while (nextDestination != __xmmWordDestination)
                _mm_storeu_si128(--nextDestination, _mm_loadu_si128(--nextSource));

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Move<32>(
        void* destination,
        const void* source,
        sizetype    length) noexcept
    {
        const __m256i* __ymmWordSource = reinterpret_cast<const __m256i*>(source);
        __m256i* __ymmWordDestination = reinterpret_cast<__m256i*>(destination);

        const __m256i* nextSource = __ymmWordSource + length;
        __m256i* nextDestination = __ymmWordDestination + length;

        if (__ymmWordDestination < __ymmWordSource)
            while (__ymmWordDestination != nextDestination)
                _mm256_storeu_si256(__ymmWordDestination++, _mm256_lddqu_si256(__ymmWordSource++));
        else
            while (nextDestination != __ymmWordDestination)
                _mm256_storeu_si256(--nextDestination, _mm256_lddqu_si256(--nextSource));

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Move<64>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const __m512i* __zmmWordSource  = reinterpret_cast<const __m512i*>(source);
        __m512i* __zmmWordDestination   = reinterpret_cast<__m512i*>(destination);

        const __m512i* nextSource   = __zmmWordSource + length;
        __m512i* nextDestination    = __zmmWordDestination + length;

        if (__zmmWordDestination < __zmmWordSource)
            while (__zmmWordDestination != nextDestination)
                _mm512_storeu_si512(__zmmWordDestination++, _mm512_loadu_si512(__zmmWordSource++));
        else
            while (nextDestination != __zmmWordDestination)
                _mm512_storeu_si512(--nextDestination, _mm512_loadu_si512(--nextSource));

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Move<128>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const __m512i* __zmmWordSource  = reinterpret_cast<const __m512i*>(source);
        __m512i* __zmmWordDestination   = reinterpret_cast<__m512i*>(destination);

        const __m512i* nextSource   = __zmmWordSource + (length << 1);
        __m512i* nextDestination    = __zmmWordDestination + (length << 1);

        if (__zmmWordDestination < __zmmWordSource) {
            while (__zmmWordDestination != nextDestination) {
                __SIMD_STL_REPEAT_N(2, _mm512_storeu_si512(__zmmWordDestination++, _mm512_loadu_si512(__zmmWordSource++)));
            }
        }
        else {
            while (nextDestination != __zmmWordDestination) {
                __SIMD_STL_REPEAT_N(2, _mm512_storeu_si512(--nextDestination, _mm512_loadu_si512(--nextSource)));
            }
        }

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Move<256>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const __m512i* __zmmWordSource  = reinterpret_cast<const __m512i*>(source);
        __m512i* __zmmWordDestination   = reinterpret_cast<__m512i*>(destination);

        const __m512i* nextSource   = __zmmWordSource + (length << 2);
        __m512i* nextDestination    = __zmmWordDestination + (length << 2);

        if (__zmmWordDestination < __zmmWordSource) {
            while (__zmmWordDestination != nextDestination) {
                __SIMD_STL_REPEAT_N(4, _mm512_storeu_si512(__zmmWordDestination++, _mm512_loadu_si512(__zmmWordSource++)));
            }
        }
        else {
            while (nextDestination != __zmmWordDestination) {
                __SIMD_STL_REPEAT_N(4, _mm512_storeu_si512(--nextDestination, _mm512_loadu_si512(--nextSource)));
            }
        }

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Move<512>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const __m512i* __zmmWordSource  = reinterpret_cast<const __m512i*>(source);
        __m512i* __zmmWordDestination   = reinterpret_cast<__m512i*>(destination);

        const __m512i* nextSource   = __zmmWordSource + (length << 3);
        __m512i* nextDestination    = __zmmWordDestination + (length << 3);

        if (__zmmWordDestination < __zmmWordSource) {
            while (__zmmWordDestination != nextDestination) {
                __SIMD_STL_REPEAT_N(8, _mm512_storeu_si512(__zmmWordDestination++, _mm512_loadu_si512(__zmmWordSource++)));
            }
        }
        else {
            while (nextDestination != __zmmWordDestination) {
                __SIMD_STL_REPEAT_N(8, _mm512_storeu_si512(--nextDestination, _mm512_loadu_si512(--nextSource)));
            }
        }

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Move<1024>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const __m512i* __zmmWordSource  = reinterpret_cast<const __m512i*>(source);
        __m512i* __zmmWordDestination   = reinterpret_cast<__m512i*>(destination);

        const __m512i* nextSource   = __zmmWordSource + (length << 4);
        __m512i* nextDestination    = __zmmWordDestination + (length << 4);

        if (__zmmWordDestination < __zmmWordSource) {
            while (__zmmWordDestination != nextDestination) {
                __SIMD_STL_REPEAT_N(16, _mm512_storeu_si512(__zmmWordDestination++, _mm512_loadu_si512(__zmmWordSource++)));
            }
        }
        else {
            while (nextDestination != __zmmWordDestination) {
                __SIMD_STL_REPEAT_N(16, _mm512_storeu_si512(--nextDestination, _mm512_loadu_si512(--nextSource)));
            }
        }

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Move<2048>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const __m512i* __zmmWordSource  = reinterpret_cast<const __m512i*>(source);
        __m512i* __zmmWordDestination   = reinterpret_cast<__m512i*>(destination);

        const __m512i* nextSource   = __zmmWordSource + (length << 5);
        __m512i* nextDestination    = __zmmWordDestination + (length << 5);

        if (__zmmWordDestination < __zmmWordSource) {
            while (__zmmWordDestination != nextDestination) {
                __SIMD_STL_REPEAT_N(32, _mm512_storeu_si512(__zmmWordDestination++, _mm512_loadu_si512(__zmmWordSource++)));
            }
        }
        else {
            while (nextDestination != __zmmWordDestination) {
                __SIMD_STL_REPEAT_N(32, _mm512_storeu_si512(--nextDestination, _mm512_loadu_si512(--nextSource)));
            }
        }

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Move<4096>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const __m512i* __zmmWordSource  = reinterpret_cast<const __m512i*>(source);
        __m512i* __zmmWordDestination   = reinterpret_cast<__m512i*>(destination);

        const __m512i* nextSource   = __zmmWordSource + (length << 6);
        __m512i* nextDestination    = __zmmWordDestination + (length << 6);

        if (__zmmWordDestination < __zmmWordSource) {
            while (__zmmWordDestination != nextDestination) {
                __SIMD_STL_REPEAT_N(64, _mm512_storeu_si512(__zmmWordDestination++, _mm512_loadu_si512(__zmmWordSource++)));
            }
        }
        else {
            while (nextDestination != __zmmWordDestination) {
                __SIMD_STL_REPEAT_N(64, _mm512_storeu_si512(--nextDestination, _mm512_loadu_si512(--nextSource)));
            }
        }

        return destination;
    }
};

template <>
struct _MoveVectorized<arch::CpuFeature::AVX512F, true> :
    _MoveVectorized<arch::CpuFeature::None, false>
{
    template <sizetype    _ElementSize_>
    simd_stl_always_inline static void* Move(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        AssertUnreachable();
        return nullptr;
    }

        template <>
    simd_stl_always_inline static void* Move<1>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const uint8* __sourceByte   = static_cast<const uint8*>(source);
        uint8* __destinationByte    = static_cast<uint8*>(destination);

        const uint8* nextSource = __sourceByte + length;
        uint8* nextDestination  = __destinationByte + length;

        if (__destinationByte < __sourceByte)
            while (__destinationByte != nextDestination)
                *__destinationByte++ = *__sourceByte++;
        else
            while (nextDestination != __destinationByte)
                *--nextDestination = *--nextSource;

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Move<2>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const uint16* __sourceWord   = static_cast<const uint16*>(source);
        uint16* __destinationWord    = static_cast<uint16*>(destination);

        const uint16* nextSource = __sourceWord + length;
        uint16* nextDestination  = __destinationWord + length;

        if (__destinationWord < __sourceWord)
            while (__destinationWord != nextDestination)
                *__destinationWord++ = *__sourceWord++;
        else
            while (nextDestination != __destinationWord)
                *--nextDestination = *--nextSource;

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Move<4>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const uint32* __sourceDWord   = static_cast<const uint32*>(source);
        uint32* __destinationDWord    = static_cast<uint32*>(destination);

        const uint32* nextSource = __sourceDWord + length;
        uint32* nextDestination  = __destinationDWord + length;

        if (__destinationDWord < __sourceDWord)
            while (__destinationDWord != nextDestination)
                *__destinationDWord++ = *__sourceDWord++;
        else
            while (nextDestination != __destinationDWord)
                *--nextDestination = *--nextSource;

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Move<8>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const uint64* __sourceQWord   = static_cast<const uint64*>(source);
        uint64* __destinationQWord    = static_cast<uint64*>(destination);

        const uint64* nextSource = __sourceQWord + length;
        uint64* nextDestination  = __destinationQWord + length;

        if (__destinationQWord < __sourceQWord)
            while (__destinationQWord != nextDestination)
                *__destinationQWord++ = *__sourceQWord++;
        else
            while (nextDestination != __destinationQWord)
                *--nextDestination = *--nextSource;

        return destination;
    }

    
    template <>
    simd_stl_always_inline static void* Move<16>(
        void*       destination,
        const void* source,
        sizetype    length) noexcept
    {
        const __m128i* __xmmWordSource  = reinterpret_cast<const __m128i*>(source);
        __m128i* __xmmWordDestination   = reinterpret_cast<__m128i*>(destination);

        const __m128i* nextSource   = __xmmWordSource + length;
        __m128i* nextDestination    = __xmmWordDestination + length;

        if (__xmmWordDestination < __xmmWordSource)
            while (__xmmWordDestination != nextDestination)
                _mm_store_si128(__xmmWordDestination++, _mm_load_si128(__xmmWordSource++));
        else
            while (nextDestination != __xmmWordDestination)
                _mm_store_si128(--nextDestination, _mm_load_si128(--nextSource));

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Move<32>(
        void*       destination,
        const void* source,
        sizetype    length) noexcept
    {
        const __m256i* __ymmWordSource  = reinterpret_cast<const __m256i*>(source);
        __m256i* __ymmWordDestination   = reinterpret_cast<__m256i*>(destination);

        const __m256i* nextSource   = __ymmWordSource + length;
        __m256i* nextDestination    = __ymmWordDestination + length;

        if (__ymmWordDestination < __ymmWordSource)
            while (__ymmWordDestination != nextDestination)
                _mm256_store_si256(__ymmWordDestination++, _mm256_load_si256(__ymmWordSource++));
        else
            while (nextDestination != __ymmWordDestination)
                _mm256_store_si256(--nextDestination, _mm256_load_si256(--nextSource));

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Move<64>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const __m512i* __zmmWordSource  = reinterpret_cast<const __m512i*>(source);
        __m512i* __zmmWordDestination   = reinterpret_cast<__m512i*>(destination);

        const __m512i* nextSource   = __zmmWordSource + length;
        __m512i* nextDestination    = __zmmWordDestination + length;

        if (__zmmWordDestination < __zmmWordSource)
            while (__zmmWordDestination != nextDestination)
                _mm512_store_si512(__zmmWordDestination++, _mm512_load_si512(__zmmWordSource++));
        else
            while (nextDestination != __zmmWordDestination)
                _mm512_store_si512(--nextDestination, _mm512_load_si512(--nextSource));

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Move<128>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const __m512i* __zmmWordSource  = reinterpret_cast<const __m512i*>(source);
        __m512i* __zmmWordDestination   = reinterpret_cast<__m512i*>(destination);

        const __m512i* nextSource   = __zmmWordSource + (length << 1);
        __m512i* nextDestination    = __zmmWordDestination + (length << 1);

        if (__zmmWordDestination < __zmmWordSource) {
            while (__zmmWordDestination != nextDestination) {
                __SIMD_STL_REPEAT_N(2, _mm512_store_si512(__zmmWordDestination++, _mm512_load_si512(__zmmWordSource++)));
            }
        }
        else {
            while (nextDestination != __zmmWordDestination) {
                __SIMD_STL_REPEAT_N(2, _mm512_store_si512(--nextDestination, _mm512_load_si512(--nextSource)));
            }
        }

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Move<256>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const __m512i* __zmmWordSource  = reinterpret_cast<const __m512i*>(source);
        __m512i* __zmmWordDestination   = reinterpret_cast<__m512i*>(destination);

        const __m512i* nextSource   = __zmmWordSource + (length << 2);
        __m512i* nextDestination    = __zmmWordDestination + (length << 2);

        if (__zmmWordDestination < __zmmWordSource) {
            while (__zmmWordDestination != nextDestination) {
                __SIMD_STL_REPEAT_N(4, _mm512_store_si512(__zmmWordDestination++, _mm512_load_si512(__zmmWordSource++)));
            }
        }
        else {
            while (nextDestination != __zmmWordDestination) {
                __SIMD_STL_REPEAT_N(4, _mm512_store_si512(--nextDestination, _mm512_load_si512(--nextSource)));
            }
        }

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Move<512>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const __m512i* __zmmWordSource  = reinterpret_cast<const __m512i*>(source);
        __m512i* __zmmWordDestination   = reinterpret_cast<__m512i*>(destination);

        const __m512i* nextSource   = __zmmWordSource + (length << 3);
        __m512i* nextDestination    = __zmmWordDestination + (length << 3);

        if (__zmmWordDestination < __zmmWordSource) {
            while (__zmmWordDestination != nextDestination) {
                __SIMD_STL_REPEAT_N(8, _mm512_store_si512(__zmmWordDestination++, _mm512_load_si512(__zmmWordSource++)));
            }
        }
        else {
            while (nextDestination != __zmmWordDestination) {
                __SIMD_STL_REPEAT_N(8, _mm512_store_si512(--nextDestination, _mm512_load_si512(--nextSource)));
            }
        }

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Move<1024>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const __m512i* __zmmWordSource  = reinterpret_cast<const __m512i*>(source);
        __m512i* __zmmWordDestination   = reinterpret_cast<__m512i*>(destination);

        const __m512i* nextSource   = __zmmWordSource + (length << 4);
        __m512i* nextDestination    = __zmmWordDestination + (length << 4);

        if (__zmmWordDestination < __zmmWordSource) {
            while (__zmmWordDestination != nextDestination) {
                __SIMD_STL_REPEAT_N(16, _mm512_store_si512(__zmmWordDestination++, _mm512_load_si512(__zmmWordSource++)));
            }
        }
        else {
            while (nextDestination != __zmmWordDestination) {
                __SIMD_STL_REPEAT_N(16, _mm512_store_si512(--nextDestination, _mm512_load_si512(--nextSource)));
            }
        }

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Move<2048>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const __m512i* __zmmWordSource  = reinterpret_cast<const __m512i*>(source);
        __m512i* __zmmWordDestination   = reinterpret_cast<__m512i*>(destination);

        const __m512i* nextSource   = __zmmWordSource + (length << 5);
        __m512i* nextDestination    = __zmmWordDestination + (length << 5);

        if (__zmmWordDestination < __zmmWordSource) {
            while (__zmmWordDestination != nextDestination) {
                __SIMD_STL_REPEAT_N(32, _mm512_store_si512(__zmmWordDestination++, _mm512_load_si512(__zmmWordSource++)));
            }
        }
        else {
            while (nextDestination != __zmmWordDestination) {
                __SIMD_STL_REPEAT_N(32, _mm512_store_si512(--nextDestination, _mm512_load_si512(--nextSource)));
            }
        }

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Move<4096>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const __m512i* __zmmWordSource  = reinterpret_cast<const __m512i*>(source);
        __m512i* __zmmWordDestination   = reinterpret_cast<__m512i*>(destination);

        const __m512i* nextSource   = __zmmWordSource + (length << 6);
        __m512i* nextDestination    = __zmmWordDestination + (length << 6);

        if (__zmmWordDestination < __zmmWordSource) {
            while (__zmmWordDestination != nextDestination) {
                __SIMD_STL_REPEAT_N(64, _mm512_store_si512(__zmmWordDestination++, _mm512_load_si512(__zmmWordSource++)));
            }
        }
        else {
            while (nextDestination != __zmmWordDestination) {
                __SIMD_STL_REPEAT_N(64, _mm512_store_si512(--nextDestination, _mm512_load_si512(--nextSource)));
            }
        }

        return destination;
    }

    template <sizetype    _ElementSize_>
    simd_stl_always_inline static void* MoveStreamAligned(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        AssertUnreachable();
        return nullptr;
    }

    template <>
    simd_stl_always_inline static void* MoveStreamAligned<64>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const __m512i* __zmmWordSource  = reinterpret_cast<const __m512i*>(source);
        __m512i* __zmmWordDestination   = reinterpret_cast<__m512i*>(destination);

        const __m512i* nextSource   = __zmmWordSource + length;
        __m512i* nextDestination    = __zmmWordDestination + length;

        if (__zmmWordDestination < __zmmWordSource)
            while (__zmmWordDestination != nextDestination)
                _mm512_stream_si512(__zmmWordDestination++, _mm512_stream_load_si512(__zmmWordSource++));
        else
            while (nextDestination != __zmmWordDestination)
                _mm512_stream_si512(--nextDestination, _mm512_stream_load_si512(--nextSource));

        _mm_sfence();

        return destination;
    }

    template <>
    simd_stl_always_inline static void* MoveStreamAligned<128>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const __m512i* __zmmWordSource  = reinterpret_cast<const __m512i*>(source);
        __m512i* __zmmWordDestination   = reinterpret_cast<__m512i*>(destination);

        const __m512i* nextSource   = __zmmWordSource + (length << 1);
        __m512i* nextDestination    = __zmmWordDestination + (length << 1);

        if (__zmmWordDestination < __zmmWordSource) {
            while (__zmmWordDestination != nextDestination) {
                __SIMD_STL_REPEAT_N(2, _mm512_stream_si512(__zmmWordDestination++, _mm512_stream_load_si512(__zmmWordSource++)));
            }
        }
        else {
            while (nextDestination != __zmmWordDestination) {
                __SIMD_STL_REPEAT_N(2, _mm512_stream_si512(--nextDestination, _mm512_stream_load_si512(--nextSource)));
            }
        }

        _mm_sfence();

        return destination;
    }

    template <>
    simd_stl_always_inline static void* MoveStreamAligned<256>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const __m512i* __zmmWordSource  = reinterpret_cast<const __m512i*>(source);
        __m512i* __zmmWordDestination   = reinterpret_cast<__m512i*>(destination);

        const __m512i* nextSource   = __zmmWordSource + (length << 2);
        __m512i* nextDestination    = __zmmWordDestination + (length << 2);

        if (__zmmWordDestination < __zmmWordSource) {
            while (__zmmWordDestination != nextDestination) {
                __SIMD_STL_REPEAT_N(4, _mm512_stream_si512(__zmmWordDestination++, _mm512_stream_load_si512(__zmmWordSource++)));
            }
        }
        else {
            while (nextDestination != __zmmWordDestination) {
                __SIMD_STL_REPEAT_N(4, _mm512_stream_si512(--nextDestination, _mm512_stream_load_si512(--nextSource)));
            }
        }

        _mm_sfence();

        return destination;
    }

    template <>
    simd_stl_always_inline static void* MoveStreamAligned<512>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const __m512i* __zmmWordSource  = reinterpret_cast<const __m512i*>(source);
        __m512i* __zmmWordDestination   = reinterpret_cast<__m512i*>(destination);

        const __m512i* nextSource   = __zmmWordSource + (length << 3);
        __m512i* nextDestination    = __zmmWordDestination + (length << 3);

        if (__zmmWordDestination < __zmmWordSource) {
            while (__zmmWordDestination != nextDestination) {
                __SIMD_STL_REPEAT_N(8, _mm512_stream_si512(__zmmWordDestination++, _mm512_stream_load_si512(__zmmWordSource++)));
            }
        }
        else {
            while (nextDestination != __zmmWordDestination) {
                __SIMD_STL_REPEAT_N(8, _mm512_stream_si512(--nextDestination, _mm512_stream_load_si512(--nextSource)));
            }
        }

        _mm_sfence();

        return destination;
    }

    template <>
    simd_stl_always_inline static void* MoveStreamAligned<1024>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const __m512i* __zmmWordSource  = reinterpret_cast<const __m512i*>(source);
        __m512i* __zmmWordDestination   = reinterpret_cast<__m512i*>(destination);

        const __m512i* nextSource   = __zmmWordSource + (length << 4);
        __m512i* nextDestination    = __zmmWordDestination + (length << 4);

        if (__zmmWordDestination < __zmmWordSource) {
            while (__zmmWordDestination != nextDestination) {
                __SIMD_STL_REPEAT_N(16, _mm512_stream_si512(__zmmWordDestination++, _mm512_stream_load_si512(__zmmWordSource++)));
            }
        }
        else {
            while (nextDestination != __zmmWordDestination) {
                __SIMD_STL_REPEAT_N(16, _mm512_stream_si512(--nextDestination, _mm512_stream_load_si512(--nextSource)));
            }
        }

        _mm_sfence();

        return destination;
    }

    template <>
    simd_stl_always_inline static void* MoveStreamAligned<2048>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const __m512i* __zmmWordSource  = reinterpret_cast<const __m512i*>(source);
        __m512i* __zmmWordDestination   = reinterpret_cast<__m512i*>(destination);

        const __m512i* nextSource   = __zmmWordSource + (length << 5);
        __m512i* nextDestination    = __zmmWordDestination + (length << 5);

        if (__zmmWordDestination < __zmmWordSource) {
            while (__zmmWordDestination != nextDestination) {
                __SIMD_STL_REPEAT_N(32, _mm512_stream_si512(__zmmWordDestination++, _mm512_stream_load_si512(__zmmWordSource++)));
            }
        }
        else {
            while (nextDestination != __zmmWordDestination) {
                __SIMD_STL_REPEAT_N(32, _mm512_stream_si512(--nextDestination, _mm512_stream_load_si512(--nextSource)));
            }
        }

        _mm_sfence();

        return destination;
    }

    template <>
    simd_stl_always_inline static void* MoveStreamAligned<4096>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const __m512i* __zmmWordSource  = reinterpret_cast<const __m512i*>(source);
        __m512i* __zmmWordDestination   = reinterpret_cast<__m512i*>(destination);

        const __m512i* nextSource   = __zmmWordSource + (length << 6);
        __m512i* nextDestination    = __zmmWordDestination + (length << 6);

        if (__zmmWordDestination < __zmmWordSource) {
            while (__zmmWordDestination != nextDestination) {
                __SIMD_STL_REPEAT_N(64, _mm512_stream_si512(__zmmWordDestination++, _mm512_stream_load_si512(__zmmWordSource++)));
            }
        }
        else {
            while (nextDestination != __zmmWordDestination) {
                __SIMD_STL_REPEAT_N(64, _mm512_stream_si512(--nextDestination, _mm512_stream_load_si512(--nextSource)));
            }
        }

        _mm_sfence();

        return destination;
    }
};

template <>
struct _MoveVectorized<arch::CpuFeature::SSE41, true>:
    _MoveVectorized<arch::CpuFeature::None, false> 
{
    template <sizetype    _ElementSize_>
    simd_stl_always_inline static void* Move(
        void*       destination,
        const void* source,
        sizetype    length) noexcept
    {
        AssertUnreachable();
        return nullptr;
    }

    template <>
    simd_stl_always_inline static void* Move<1>(
        void*       destination,
        const void* source,
        sizetype    length) noexcept
    {
        const uint8* __sourceByte = static_cast<const uint8*>(source);
        uint8* __destinationByte = static_cast<uint8*>(destination);

        const uint8* nextSource = __sourceByte + length;
        uint8* nextDestination = __destinationByte + length;

        if (__destinationByte < __sourceByte)
            while (__destinationByte != nextDestination)
                *__destinationByte++ = *__sourceByte++;
        else
            while (nextDestination != __destinationByte)
                *--nextDestination = *--nextSource;

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Move<2>(
        void*       destination,
        const void* source,
        sizetype    length) noexcept
    {
        const uint16* __sourceWord = static_cast<const uint16*>(source);
        uint16* __destinationWord = static_cast<uint16*>(destination);

        const uint16* nextSource = __sourceWord + length;
        uint16* nextDestination = __destinationWord + length;

        if (__destinationWord < __sourceWord)
            while (__destinationWord != nextDestination)
                *__destinationWord++ = *__sourceWord++;
        else
            while (nextDestination != __destinationWord)
                *--nextDestination = *--nextSource;

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Move<4>(
        void*       destination,
        const void* source,
        sizetype    length) noexcept
    {
        const uint32* __sourceDWord = static_cast<const uint32*>(source);
        uint32* __destinationDWord = static_cast<uint32*>(destination);

        const uint32* nextSource = __sourceDWord + length;
        uint32* nextDestination = __destinationDWord + length;

        if (__destinationDWord < __sourceDWord)
            while (__destinationDWord != nextDestination)
                *__destinationDWord++ = *__sourceDWord++;
        else
            while (nextDestination != __destinationDWord)
                *--nextDestination = *--nextSource;

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Move<8>(
        void*       destination,
        const void* source,
        sizetype    length) noexcept
    {
        const uint64* __sourceQWord = static_cast<const uint64*>(source);
        uint64* __destinationQWord = static_cast<uint64*>(destination);

        const uint64* nextSource = __sourceQWord + length;
        uint64* nextDestination = __destinationQWord + length;

        if (__destinationQWord < __sourceQWord)
            while (__destinationQWord != nextDestination)
                *__destinationQWord++ = *__sourceQWord++;
        else
            while (nextDestination != __destinationQWord)
                *--nextDestination = *--nextSource;

        return destination;
    }

    template <sizetype    _ElementSize_>
    simd_stl_always_inline static void* MoveStreamAligned(
        void* destination,
        const void* source,
        sizetype    length) noexcept
    {
        AssertUnreachable();
        return nullptr;
    }

    template <>
    simd_stl_always_inline static void* MoveStreamAligned<16>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const __m128i* __xmmWordSource  = reinterpret_cast<const __m128i*>(source);
        __m128i* __xmmWordDestination   = reinterpret_cast<__m128i*>(destination);

        const __m128i* nextSource   = __xmmWordSource + length;
        __m128i* nextDestination    = __xmmWordDestination + length;

        if (__xmmWordDestination < __xmmWordSource)
            while (__xmmWordDestination != nextDestination)
                _mm_stream_si128(__xmmWordDestination++, _mm_stream_load_si128(__xmmWordSource++));
        else
            while (nextDestination != __xmmWordDestination)
                _mm_stream_si128(--nextDestination, _mm_stream_load_si128(--nextSource));

        _mm_sfence();

        return destination;
    }

    template <>
    simd_stl_always_inline static void* MoveStreamAligned<32>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const __m128i* __xmmWordSource  = reinterpret_cast<const __m128i*>(source);
        __m128i* __xmmWordDestination   = reinterpret_cast<__m128i*>(destination);

        const __m128i* nextSource   = __xmmWordSource + (length << 1);
        __m128i* nextDestination    = __xmmWordDestination + (length << 1);

        if (__xmmWordDestination < __xmmWordSource) {
            while (__xmmWordDestination != nextDestination) {
                __SIMD_STL_REPEAT_N(2, _mm_stream_si128(__xmmWordDestination++, _mm_stream_load_si128(__xmmWordSource++)));
            }
        }
        else {
            while (nextDestination != __xmmWordDestination) {
                __SIMD_STL_REPEAT_N(2, _mm_stream_si128(--nextDestination, _mm_stream_load_si128(--nextSource)));
            }
        }

        _mm_sfence();

        return destination;
    }

    template <>
    simd_stl_always_inline static void* MoveStreamAligned<64>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const __m128i* __xmmWordSource  = reinterpret_cast<const __m128i*>(source);
        __m128i* __xmmWordDestination   = reinterpret_cast<__m128i*>(destination);

        const __m128i* nextSource   = __xmmWordSource + (length << 2);
        __m128i* nextDestination    = __xmmWordDestination + (length << 2);

        if (__xmmWordDestination < __xmmWordSource) {
            while (__xmmWordDestination != nextDestination) {
                __SIMD_STL_REPEAT_N(4, _mm_stream_si128(__xmmWordDestination++, _mm_stream_load_si128(__xmmWordSource++)));
            }
        }
        else {
            while (nextDestination != __xmmWordDestination) {
                __SIMD_STL_REPEAT_N(4, _mm_stream_si128(--nextDestination, _mm_stream_load_si128(--nextSource)));
            }
        }

        _mm_sfence();

        return destination;
    }

    template <>
    simd_stl_always_inline static void* MoveStreamAligned<128>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const __m128i* __xmmWordSource  = reinterpret_cast<const __m128i*>(source);
        __m128i* __xmmWordDestination   = reinterpret_cast<__m128i*>(destination);

        const __m128i* nextSource   = __xmmWordSource + (length << 3);
        __m128i* nextDestination    = __xmmWordDestination + (length << 3);

        if (__xmmWordDestination < __xmmWordSource) {
            while (__xmmWordDestination != nextDestination) {
                __SIMD_STL_REPEAT_N(8, _mm_stream_si128(__xmmWordDestination++, _mm_stream_load_si128(__xmmWordSource++)));
            }
        }
        else {
            while (nextDestination != __xmmWordDestination) {
                __SIMD_STL_REPEAT_N(8, _mm_stream_si128(--nextDestination, _mm_stream_load_si128(--nextSource)));
            }
        }

        _mm_sfence();

        return destination;
    }

    template <>
    simd_stl_always_inline static void* MoveStreamAligned<256>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const __m128i* __xmmWordSource  = reinterpret_cast<const __m128i*>(source);
        __m128i* __xmmWordDestination   = reinterpret_cast<__m128i*>(destination);

        const __m128i* nextSource   = __xmmWordSource + (length << 4);
        __m128i* nextDestination    = __xmmWordDestination + (length << 4);

        if (__xmmWordDestination < __xmmWordSource) {
            while (__xmmWordDestination != nextDestination) {
                __SIMD_STL_REPEAT_N(16, _mm_stream_si128(__xmmWordDestination++, _mm_stream_load_si128(__xmmWordSource++)));
            }
        }
        else {
            while (nextDestination != __xmmWordDestination) {
                __SIMD_STL_REPEAT_N(16, _mm_stream_si128(--nextDestination, _mm_stream_load_si128(--nextSource)));
            }
        }

        _mm_sfence();

        return destination;
    }
};

template <>
struct _MoveVectorized<arch::CpuFeature::SSE41, false> :
    _MoveVectorized<arch::CpuFeature::SSE2, false>
{};

template <>
struct _MoveVectorized<arch::CpuFeature::AVX2, false> :
    _MoveVectorized<arch::CpuFeature::AVX, false>
{};

template <>
struct _MoveVectorized<arch::CpuFeature::AVX2, true>:
    _MoveVectorized<arch::CpuFeature::None, true> 
{
    template <sizetype    _ElementSize_>
    simd_stl_always_inline static void* Move(
        void*       destination,
        const void* source,
        sizetype    length) noexcept
    {
        AssertUnreachable();
        return nullptr;
    }

    template <>
    simd_stl_always_inline static void* Move<1>(
        void*       destination,
        const void* source,
        sizetype    length) noexcept
    {
        const uint8* __sourceByte = static_cast<const uint8*>(source);
        uint8* __destinationByte = static_cast<uint8*>(destination);

        const uint8* nextSource = __sourceByte + length;
        uint8* nextDestination = __destinationByte + length;

        if (__destinationByte < __sourceByte)
            while (__destinationByte != nextDestination)
                *__destinationByte++ = *__sourceByte++;
        else
            while (nextDestination != __destinationByte)
                *--nextDestination = *--nextSource;

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Move<2>(
        void*       destination,
        const void* source,
        sizetype    length) noexcept
    {
        const uint16* __sourceWord = static_cast<const uint16*>(source);
        uint16* __destinationWord = static_cast<uint16*>(destination);

        const uint16* nextSource = __sourceWord + length;
        uint16* nextDestination = __destinationWord + length;

        if (__destinationWord < __sourceWord)
            while (__destinationWord != nextDestination)
                *__destinationWord++ = *__sourceWord++;
        else
            while (nextDestination != __destinationWord)
                *--nextDestination = *--nextSource;

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Move<4>(
        void*       destination,
        const void* source,
        sizetype    length) noexcept
    {
        const uint32* __sourceDWord = static_cast<const uint32*>(source);
        uint32* __destinationDWord = static_cast<uint32*>(destination);

        const uint32* nextSource = __sourceDWord + length;
        uint32* nextDestination = __destinationDWord + length;

        if (__destinationDWord < __sourceDWord)
            while (__destinationDWord != nextDestination)
                *__destinationDWord++ = *__sourceDWord++;
        else
            while (nextDestination != __destinationDWord)
                *--nextDestination = *--nextSource;

        return destination;
    }

    template <>
    simd_stl_always_inline static void* Move<8>(
        void*       destination,
        const void* source,
        sizetype    length) noexcept
    {
        const uint64* __sourceQWord = static_cast<const uint64*>(source);
        uint64* __destinationQWord = static_cast<uint64*>(destination);

        const uint64* nextSource = __sourceQWord + length;
        uint64* nextDestination = __destinationQWord + length;

        if (__destinationQWord < __sourceQWord)
            while (__destinationQWord != nextDestination)
                *__destinationQWord++ = *__sourceQWord++;
        else
            while (nextDestination != __destinationQWord)
                *--nextDestination = *--nextSource;

        return destination;
    }

    template <sizetype    _ElementSize_>
    simd_stl_always_inline static void* MoveStreamAligned(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        AssertUnreachable();
        return nullptr;
    }

    template <>
    simd_stl_always_inline static void* MoveStreamAligned<16>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const __m128i* __xmmWordSource  = reinterpret_cast<const __m128i*>(source);
        __m128i* __xmmWordDestination   = reinterpret_cast<__m128i*>(destination);

        const __m128i* nextSource   = __xmmWordSource + length;
        __m128i* nextDestination    = __xmmWordDestination + length;

        if (__xmmWordDestination < __xmmWordSource)
            while (__xmmWordDestination != nextDestination)
                _mm_stream_si128(__xmmWordDestination++, _mm_stream_load_si128(__xmmWordSource++));
        else
            while (nextDestination != __xmmWordDestination)
                _mm_stream_si128(--nextDestination, _mm_stream_load_si128(--nextSource));

        _mm_sfence();

        return destination;
    }

    template <>
    simd_stl_always_inline static void* MoveStreamAligned<32>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const __m256i* __ymmWordSource  = reinterpret_cast<const __m256i*>(source);
        __m256i* __ymmWordDestination   = reinterpret_cast<__m256i*>(destination);

        const __m256i* nextSource   = __ymmWordSource + length;
        __m256i* nextDestination    = __ymmWordDestination + length;

        if (__ymmWordDestination < __ymmWordSource)
            while (__ymmWordDestination != nextDestination)
                _mm256_stream_si256(__ymmWordDestination++, _mm256_stream_load_si256(__ymmWordSource++));
        else
            while (nextDestination != __ymmWordDestination)
                _mm256_stream_si256(--nextDestination, _mm256_stream_load_si256(--nextSource));

        _mm_sfence();

        return destination;
    }

    template <>
    simd_stl_always_inline static void* MoveStreamAligned<64>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const __m256i* __ymmWordSource  = reinterpret_cast<const __m256i*>(source);
        __m256i* __ymmWordDestination   = reinterpret_cast<__m256i*>(destination);

        const __m256i* nextSource   = __ymmWordSource + (length << 1);
        __m256i* nextDestination    = __ymmWordDestination + (length << 1);

        if (__ymmWordDestination < __ymmWordSource) {
            while (__ymmWordDestination != nextDestination) {
                __SIMD_STL_REPEAT_N(2, _mm256_stream_si256(__ymmWordDestination++, _mm256_stream_load_si256(__ymmWordSource++)));
            }
        }
        else {
            while (nextDestination != __ymmWordDestination) {
                __SIMD_STL_REPEAT_N(2, _mm256_stream_si256(--nextDestination, _mm256_stream_load_si256(--nextSource)));
            }
        }

        _mm_sfence();

        return destination;
    }

    template <>
    simd_stl_always_inline static void* MoveStreamAligned<128>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const __m256i* __ymmWordSource  = reinterpret_cast<const __m256i*>(source);
        __m256i* __ymmWordDestination   = reinterpret_cast<__m256i*>(destination);

        const __m256i* nextSource   = __ymmWordSource + (length << 2);
        __m256i* nextDestination    = __ymmWordDestination + (length << 2);

        if (__ymmWordDestination < __ymmWordSource) {
            while (__ymmWordDestination != nextDestination) {
                __SIMD_STL_REPEAT_N(4, _mm256_stream_si256(__ymmWordDestination++, _mm256_stream_load_si256(__ymmWordSource++)));
            }
        }
        else {
            while (nextDestination != __ymmWordDestination) {
                __SIMD_STL_REPEAT_N(4, _mm256_stream_si256(--nextDestination, _mm256_stream_load_si256(--nextSource)));
            }
        }

        _mm_sfence();

        return destination;
    }

    template <>
    simd_stl_always_inline static void* MoveStreamAligned<256>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const __m256i* __ymmWordSource  = reinterpret_cast<const __m256i*>(source);
        __m256i* __ymmWordDestination   = reinterpret_cast<__m256i*>(destination);

        const __m256i* nextSource   = __ymmWordSource + (length << 3);
        __m256i* nextDestination    = __ymmWordDestination + (length << 3);

        if (__ymmWordDestination < __ymmWordSource) {
            while (__ymmWordDestination != nextDestination) {
                __SIMD_STL_REPEAT_N(8, _mm256_stream_si256(__ymmWordDestination++, _mm256_stream_load_si256(__ymmWordSource++)));
            }
        }
        else {
            while (nextDestination != __ymmWordDestination) {
                __SIMD_STL_REPEAT_N(8, _mm256_stream_si256(--nextDestination, _mm256_stream_load_si256(--nextSource)));
            }
        }

        _mm_sfence();

        return destination;
    }

    template <>
    simd_stl_always_inline static void* MoveStreamAligned<512>(
        void*       destination,
        const void* source, 
        sizetype    length) noexcept
    {
        const __m256i* __ymmWordSource  = reinterpret_cast<const __m256i*>(source);
        __m256i* __ymmWordDestination   = reinterpret_cast<__m256i*>(destination);

        const __m256i* nextSource   = __ymmWordSource + (length << 4);
        __m256i* nextDestination    = __ymmWordDestination + (length << 4);

        if (__ymmWordDestination < __ymmWordSource) {
            while (__ymmWordDestination != nextDestination) {
                __SIMD_STL_REPEAT_N(16, _mm256_stream_si256(__ymmWordDestination++, _mm256_stream_load_si256(__ymmWordSource++)));
            }
        }
        else {
            while (nextDestination != __ymmWordDestination) {
                __SIMD_STL_REPEAT_N(16, _mm256_stream_si256(--nextDestination, _mm256_stream_load_si256(--nextSource)));
            }
        }

        _mm_sfence();

        return destination;
    }
};

template <
    bool                _Aligned_,
    bool                _Streaming_,
    arch::CpuFeature    _SimdGeneration_>
struct _MemmoveVectorizedChooser;

template <
    bool _Aligned_,
    bool _Streaming_>
struct _MemmoveVectorizedChooser<_Aligned_, _Streaming_, arch::CpuFeature::None> {
    static_assert(!_Streaming_, "Streaming not supported for SSE2. ");

    simd_stl_always_inline void operator()(
        void* destination,
        const void* source,
        sizetype    bytes) noexcept
    {
        using _VectorizedMoveImplementation_ = _MoveVectorized<arch::CpuFeature::SSE2, _Aligned_>;
        sizetype offset = 0;

        while (bytes)
        {
            if (bytes < 2)
            {
                _VectorizedMoveImplementation_::Move<1>(destination, source, bytes);
                offset = bytes & -1;
                destination = static_cast<char*>(destination) + offset;
                source = static_cast<const char*>(source) + offset;
                bytes = 0;
            }
            else if (bytes < 4)
            {
                _VectorizedMoveImplementation_::Move<2>(destination, source, bytes >> 1);
                __RECALCULATE_REMAINING(2)
            }
            else if (bytes < 8)
            {
                _VectorizedMoveImplementation_::Move<4>(destination, source, bytes >> 2);
                __RECALCULATE_REMAINING(4)
            }
            else
            {
                _VectorizedMoveImplementation_::Move<8>(destination, source, bytes >> 3);
                __RECALCULATE_REMAINING(8)
            }
        }
    }
};

template <
    bool _Aligned_,
    bool _Streaming_>
struct _MemmoveVectorizedChooser<_Aligned_, _Streaming_, arch::CpuFeature::SSE2> {
    static_assert(!_Streaming_, "Streaming not supported for SSE2. ");

    simd_stl_always_inline void operator()(
        void*       destination, 
        const void* source,
        sizetype    bytes) noexcept 
    {
        using _VectorizedMoveImplementation_ = _MoveVectorized<arch::CpuFeature::SSE2, _Aligned_>;
        sizetype offset = 0;

        while (bytes)
        {
            if (bytes < 2)
            {
                _VectorizedMoveImplementation_::Move<1>(destination, source, bytes);
                offset = bytes & -1;
                destination = static_cast<char*>(destination) + offset; 
                source = static_cast<const char*>(source) + offset; 
                bytes = 0;
            }
            else if (bytes < 4)
            {
                _VectorizedMoveImplementation_::Move<2>(destination, source, bytes >> 1);
                __RECALCULATE_REMAINING(2)
            }
            else if (bytes < 8)
            {
                _VectorizedMoveImplementation_::Move<4>(destination, source, bytes >> 2);
                __RECALCULATE_REMAINING(4)
            }
            else if (bytes < 16)
            {
                _VectorizedMoveImplementation_::Move<8>(destination, source, bytes >> 3);
                __RECALCULATE_REMAINING(8)
            }
            else if (bytes < 32)
            {
                _VectorizedMoveImplementation_::Move<16>(destination, source, bytes >> 4);
                __RECALCULATE_REMAINING(16)
            }
            else if (bytes < 64)
            {
                _VectorizedMoveImplementation_::Move<32>(destination, source, bytes >> 5);
                __RECALCULATE_REMAINING(32)
            }
            else if (bytes < 128)
            {
                _VectorizedMoveImplementation_::Move<64>(destination, source, bytes >> 6);
                __RECALCULATE_REMAINING(64)
            }
            else if (bytes < 256)
            {
                _VectorizedMoveImplementation_::Move<128>(destination, source, bytes >> 7);
                __RECALCULATE_REMAINING(128)
            }
            else
            {
                _VectorizedMoveImplementation_::Move<256>(destination, source, bytes >> 8);
                __RECALCULATE_REMAINING(256)
            }
        }
    }
};

template <
    bool _Aligned_,
    bool _Streaming_>
struct _MemmoveVectorizedChooser<_Aligned_, _Streaming_, arch::CpuFeature::SSE41> {
    static_assert(_Aligned_ >= _Streaming_, "Streaming loads/stores must be aligned. ");

    simd_stl_always_inline void operator()(
        void*       destination, 
        const void* source,
        sizetype    bytes) noexcept 
    {
        using _VectorizedMoveImplementation_ = _MoveVectorized<arch::CpuFeature::SSE41, _Aligned_>;
        sizetype offset = 0;

        while (bytes)
        {
            if (bytes < 2)
            {
                __DISPATCH_VECTORIZED_MOVE(1, 0)
                offset = bytes & -1;
                destination = static_cast<char*>(destination) + offset;
                source = static_cast<const char*>(source) + offset;
                bytes = 0;
            }
            else if (bytes < 4)
            {
                __DISPATCH_VECTORIZED_MOVE(2, 1)
                __RECALCULATE_REMAINING(2)
            }
            else if (bytes < 8)
            {
                __DISPATCH_VECTORIZED_MOVE(4, 2)
                __RECALCULATE_REMAINING(4)
            }
            else if (bytes < 16)
            {
                __DISPATCH_VECTORIZED_MOVE(8, 3)
                __RECALCULATE_REMAINING(8)
            }
            else if (bytes < 32)
            {
                __DISPATCH_VECTORIZED_MOVE(16, 4)
                __RECALCULATE_REMAINING(16)
            }
            else if (bytes < 64)
            {
                __DISPATCH_VECTORIZED_MOVE(32, 5)
                __RECALCULATE_REMAINING(32)
            }
            else if (bytes < 128)
            {
                __DISPATCH_VECTORIZED_MOVE(64, 6)
                __RECALCULATE_REMAINING(64)
            }
            else if (bytes < 256)
            {
                __DISPATCH_VECTORIZED_MOVE(128, 7)
                __RECALCULATE_REMAINING(128)
            }
            else
            {
                __DISPATCH_VECTORIZED_MOVE(256, 8)
                __RECALCULATE_REMAINING(256)
            }
        }
    }
};

template <
    bool _Aligned_,
    bool _Streaming_>
struct _MemmoveVectorizedChooser<_Aligned_, _Streaming_, arch::CpuFeature::AVX> {
    static_assert(!_Streaming_, "Streaming not supported for AVX. ");

    simd_stl_always_inline void operator()(
        void*       destination, 
        const void* source,
        sizetype    bytes) noexcept 
    {
        using _VectorizedMoveImplementation_ = _MoveVectorized<arch::CpuFeature::AVX, _Aligned_>;
        sizetype offset = 0;

        while (bytes)
        {
            if (bytes < 2)
            {
                _VectorizedMoveImplementation_::Move<1>(destination, source, bytes);
                offset = bytes & -1;
                destination = static_cast<char*>(destination) + offset;
                source = static_cast<const char*>(source) + offset;
                bytes = 0;
            }
            else if (bytes < 4)
            {
                _VectorizedMoveImplementation_::Move<2>(destination, source, bytes >> 1);
                __RECALCULATE_REMAINING(2)
            }
            else if (bytes < 8)
            {
                _VectorizedMoveImplementation_::Move<4>(destination, source, bytes >> 2);
                __RECALCULATE_REMAINING(4)
            }
            else if (bytes < 16)
            {
                _VectorizedMoveImplementation_::Move<8>(destination, source, bytes >> 3);
                __RECALCULATE_REMAINING(8)
            }
            else if (bytes < 32)
            {
                _VectorizedMoveImplementation_::Move<16>(destination, source, bytes >> 4);
                __RECALCULATE_REMAINING(16)
            }
            else if (bytes < 64)
            {
                _VectorizedMoveImplementation_::Move<32>(destination, source, bytes >> 5);
                __RECALCULATE_REMAINING(32)
            }
            else if (bytes < 128)
            {
                _VectorizedMoveImplementation_::Move<64>(destination, source, bytes >> 6);
                __RECALCULATE_REMAINING(64)
            }
            else if (bytes < 256)
            {
                _VectorizedMoveImplementation_::Move<128>(destination, source, bytes >> 7);
                __RECALCULATE_REMAINING(128)
            }
            else if (bytes < 512)
            {
                _VectorizedMoveImplementation_::Move<256>(destination, source, bytes >> 8);
                __RECALCULATE_REMAINING(256)
            }
            else
            {
                _VectorizedMoveImplementation_::Move<512>(destination, source, bytes >> 9);
                __RECALCULATE_REMAINING(512)
            }
        }
    }
};

template <
    bool _Aligned_,
    bool _Streaming_>
struct _MemmoveVectorizedChooser<_Aligned_, _Streaming_, arch::CpuFeature::AVX2> {
    static_assert(_Aligned_ >= _Streaming_, "Streaming loads/stores must be aligned. ");

    simd_stl_always_inline void operator()(
        void*       destination, 
        const void* source,
        sizetype    bytes) noexcept 
    {
        using _VectorizedMoveImplementation_ = _MoveVectorized<arch::CpuFeature::AVX2, _Aligned_>;
        sizetype offset = 0;

        while (bytes)
        {
            if (bytes < 2)
            {
                __DISPATCH_VECTORIZED_MOVE(1, 0);
                offset = bytes & -1;
                destination = static_cast<char*>(destination) + offset;
                source = static_cast<const char*>(source) + offset;
                bytes = 0;
            }
            else if (bytes < 4)
            {
                __DISPATCH_VECTORIZED_MOVE(2, 1)
                __RECALCULATE_REMAINING(2)
            }
            else if (bytes < 8)
            {
                __DISPATCH_VECTORIZED_MOVE(4, 2)
                __RECALCULATE_REMAINING(4)
            }
            else if (bytes < 16)
            {
                __DISPATCH_VECTORIZED_MOVE(8, 3)
                __RECALCULATE_REMAINING(8)
            }
            else if (bytes < 32)
            {
                __DISPATCH_VECTORIZED_MOVE(16, 4)
                __RECALCULATE_REMAINING(16)
            }
            else if (bytes < 64)
            {
                __DISPATCH_VECTORIZED_MOVE(32, 5)
                __RECALCULATE_REMAINING(32)
            }
            else if (bytes < 128)
            {
                __DISPATCH_VECTORIZED_MOVE(64, 6)
                __RECALCULATE_REMAINING(64)
            }
            else if (bytes < 256)
            {
                __DISPATCH_VECTORIZED_MOVE(128, 7)
                __RECALCULATE_REMAINING(128)
            }
            else if (bytes < 512)
            {
                __DISPATCH_VECTORIZED_MOVE(256, 8)
                __RECALCULATE_REMAINING(256)
            }
            else
            {
                __DISPATCH_VECTORIZED_MOVE(512, 9)
                __RECALCULATE_REMAINING(512);
            }
        }
    }
};


template <
    bool _Aligned_,
    bool _Streaming_>
struct _MemmoveVectorizedChooser<_Aligned_, _Streaming_, arch::CpuFeature::AVX512F> {
    static_assert(_Aligned_ >= _Streaming_, "Streaming loads/stores must be aligned. ");

    simd_stl_always_inline void operator()(
        void*       destination, 
        const void* source,
        sizetype    bytes) noexcept 
    {
        using _VectorizedMoveImplementation_ = _MoveVectorized<arch::CpuFeature::AVX512F, _Aligned_>;
        sizetype offset = 0;

        while (bytes)
        {
            if (bytes < 2)
            {
                __DISPATCH_VECTORIZED_MOVE(1, 0);
                offset = bytes & -1;
                destination = static_cast<char*>(destination) + offset;
                source = static_cast<const char*>(source) + offset;
                bytes = 0;
            }
            else if (bytes < 4)
            {
                __DISPATCH_VECTORIZED_MOVE(2, 1)
                __RECALCULATE_REMAINING(2)
            }
            else if (bytes < 8)
            {
                __DISPATCH_VECTORIZED_MOVE(4, 2)
                __RECALCULATE_REMAINING(4)
            }
            else if (bytes < 16)
            {
                __DISPATCH_VECTORIZED_MOVE(8, 3)
                __RECALCULATE_REMAINING(8)
            }
            else if (bytes < 32)
            {
                __DISPATCH_VECTORIZED_MOVE(16, 4)
                __RECALCULATE_REMAINING(16)
            }
            else if (bytes < 64)
            {
                __DISPATCH_VECTORIZED_MOVE(32, 5)
                __RECALCULATE_REMAINING(32)
            }
            else if (bytes < 128)
            {
                __DISPATCH_VECTORIZED_MOVE(64, 6)
                __RECALCULATE_REMAINING(64)
            }
            else if (bytes < 256)
            {
                __DISPATCH_VECTORIZED_MOVE(128, 7)
                __RECALCULATE_REMAINING(128)
            }
            else if (bytes < 512)
            {
                __DISPATCH_VECTORIZED_MOVE(256, 8)
                __RECALCULATE_REMAINING(256)
            }
            else if (bytes < 1024)
            {
                __DISPATCH_VECTORIZED_MOVE(512, 9)
                __RECALCULATE_REMAINING(512)
            }
            else if (bytes < 2048)
            {
                __DISPATCH_VECTORIZED_MOVE(1024, 10)
                __RECALCULATE_REMAINING(1024)
            }
            else if (bytes < 4096)
            {
                __DISPATCH_VECTORIZED_MOVE(2048, 11)
                __RECALCULATE_REMAINING(2048)
            }
            else
            {
                __DISPATCH_VECTORIZED_MOVE(4096, 12)
                __RECALCULATE_REMAINING(4096);
            }
        }
    }
};

template <
    bool                _Aligned_,
    bool                _Streaming_,
    arch::CpuFeature    _SimdGeneration_>
struct _MemmoveVectorizedReversedChooser {};


template <
    bool _Aligned_,
    bool _Streaming_>
struct _MemmoveVectorizedReversedChooser<_Aligned_, _Streaming_, arch::CpuFeature::None> {
    static_assert(!_Streaming_, "Streaming not supported. ");

    simd_stl_always_inline void operator()(
        void* destination,
        const void* source,
        sizetype    bytes) noexcept
    {
        using _VectorizedMoveImplementation_ = _MoveVectorized<arch::CpuFeature::None, _Aligned_>;
        sizetype offset = 0;

        void* nextDestination   = static_cast<char*>(destination) + bytes;
        const void* nextSource  = static_cast<const char*>(source) + bytes;

        while (bytes)
        {
            if (bytes & 1) {
                __REVERSED_MEMMOVE_CALL(1);
            }
            else if (bytes & 2) {
                __REVERSED_MEMMOVE_CALL(2);
            }
            else if (bytes & 4) {
                __REVERSED_MEMMOVE_CALL(4);
            }
            else {
                offset = bytes;

                nextDestination = (char*)nextDestination - offset;
                nextSource      = (char*)nextSource - offset;

                _VectorizedMoveImplementation_::Move<8>(nextDestination, nextSource, bytes >> 3); 
                bytes = 0;
            }
        }
    }
};

template <
    bool _Aligned_,
    bool _Streaming_>
struct _MemmoveVectorizedReversedChooser<_Aligned_, _Streaming_, arch::CpuFeature::SSE2> {
    static_assert(!_Streaming_, "Streaming not supported for SSE2. ");

    simd_stl_always_inline void operator()(
        void*       destination, 
        const void* source,
        sizetype    bytes) noexcept 
    {
        using _VectorizedMoveImplementation_ = _MoveVectorized<arch::CpuFeature::SSE2, _Aligned_>;
        sizetype offset = 0;

        void* nextDestination       = static_cast<char*>(destination) + bytes;
        const void* nextSource      = static_cast<const char*>(source) + bytes;


        while (bytes)
        {
            if (bytes & 1) {
                __REVERSED_MEMMOVE_CALL(1);
            }
            else if (bytes & 2) {
                __REVERSED_MEMMOVE_CALL(2);
            }
            else if (bytes & 4) {
                __REVERSED_MEMMOVE_CALL(4);
            }
            else if (bytes & 8) {
                __REVERSED_MEMMOVE_CALL(8);
            }
            else if (bytes & 16) {
                __REVERSED_MEMMOVE_CALL(16);
            }
            else if (bytes & 32) {
                __REVERSED_MEMMOVE_CALL(32);
            }
            else if (bytes & 64) {
                __REVERSED_MEMMOVE_CALL(64);
            }
            else if (bytes & 128) {
                __REVERSED_MEMMOVE_CALL(128);
            }
            else
            {
                offset = bytes; 

                nextDestination = (char*)nextDestination - offset;
                nextSource      = (char*)nextSource - offset;

                _VectorizedMoveImplementation_::Move<256>(nextDestination, nextSource, bytes >> 8);
                bytes = 0;
            }
        }
    }
};


template <
    bool _Aligned_,
    bool _Streaming_>
struct _MemmoveVectorizedReversedChooser<_Aligned_, _Streaming_, arch::CpuFeature::SSE41> {
    static_assert(_Aligned_ >= _Streaming_, "Streaming loads/stores must be aligned. ");

    simd_stl_always_inline void operator()(
        void*       destination, 
        const void* source,
        sizetype    bytes) noexcept 
    {
        using _VectorizedMoveImplementation_ = _MoveVectorized<arch::CpuFeature::SSE41, _Aligned_>;
        sizetype offset = 0;

        void* nextDestination   = static_cast<char*>(destination) + bytes;
        const void* nextSource  = static_cast<const char*>(source) + bytes;


        while (bytes)
        {
            if (bytes & 1) {
                __REVERSED_MEMMOVE_CALL(1);
            }
            else if (bytes & 2) {
                __REVERSED_MEMMOVE_CALL(2);
            }
            else if (bytes & 4) {
                __REVERSED_MEMMOVE_CALL(4);
            }
            else if (bytes & 8) {
                __REVERSED_MEMMOVE_CALL(8);
            }
            else if (bytes & 16) {
                __REVERSED_MEMMOVE_CALL_WITH_DISPATCH(16);
            }
            else if (bytes & 32) {
                __REVERSED_MEMMOVE_CALL_WITH_DISPATCH(32);
            }
            else if (bytes & 64) {
                __REVERSED_MEMMOVE_CALL_WITH_DISPATCH(64);
            }
            else if (bytes & 128) {
                __REVERSED_MEMMOVE_CALL_WITH_DISPATCH(128);
            }
            else {
                offset = bytes; 

                nextDestination = (char*)nextDestination - offset; 
                nextSource      = (char*)nextSource - offset;
                
                if constexpr (_Streaming_)
                    _VectorizedMoveImplementation_::MoveStreamAligned<256>(nextDestination, nextSource, bytes >> 8);
                else
                    _VectorizedMoveImplementation_::Move<256>(nextDestination, nextSource, bytes >> 8);              
                
                bytes = 0;
            }
        }
    }
};


template <
    bool _Aligned_,
    bool _Streaming_>
struct _MemmoveVectorizedReversedChooser<_Aligned_, _Streaming_, arch::CpuFeature::AVX> {
    static_assert(!_Streaming_, "Streaming not supported for AVX. ");

    simd_stl_always_inline void operator()(
        void*       destination, 
        const void* source,
        sizetype    bytes) noexcept 
    {
        using _VectorizedMoveImplementation_ = _MoveVectorized<arch::CpuFeature::AVX, _Aligned_>;
        sizetype offset = 0;

        void* nextDestination   = static_cast<char*>(destination) + bytes;
        const void* nextSource  = static_cast<const char*>(source) + bytes;


        while (bytes)
        {
            if (bytes & 1) {
                __REVERSED_MEMMOVE_CALL(1);
            }
            else if (bytes & 2) {
                __REVERSED_MEMMOVE_CALL(2);
            }
            else if (bytes & 4) {
                __REVERSED_MEMMOVE_CALL(4);
            }
            else if (bytes & 8) {
                __REVERSED_MEMMOVE_CALL(8);
            }
            else if (bytes & 16) {
                __REVERSED_MEMMOVE_CALL(16);
            }
            else if (bytes & 32) {
                __REVERSED_MEMMOVE_CALL(32);
            }
            else if (bytes & 64) {
                __REVERSED_MEMMOVE_CALL(64);
            }
            else if (bytes & 128) {
                __REVERSED_MEMMOVE_CALL(128);
            }
            else if (bytes & 256) {
                __REVERSED_MEMMOVE_CALL(256);
            }
            else {
                offset = bytes; 

                nextDestination = (char*)nextDestination - offset; 
                nextSource      = (char*)nextSource - offset;

                _VectorizedMoveImplementation_::Move<512>(nextDestination, nextSource, bytes >> 9);
                bytes = 0;
            }
        }
    }
};

template <
    bool _Aligned_,
    bool _Streaming_>
struct _MemmoveVectorizedReversedChooser<_Aligned_, _Streaming_, arch::CpuFeature::AVX2> {
    static_assert(_Aligned_ >= _Streaming_, "Streaming loads/stores must be aligned. ");

    simd_stl_always_inline void operator()(
        void*       destination, 
        const void* source,
        sizetype    bytes) noexcept 
    {
        using _VectorizedMoveImplementation_ = _MoveVectorized<arch::CpuFeature::AVX2, _Aligned_>;
        sizetype offset = 0;

        void* nextDestination   = static_cast<char*>(destination) + bytes;
        const void* nextSource  = static_cast<const char*>(source) + bytes;

        while (bytes)
        {
            if (bytes & 1) {
                __REVERSED_MEMMOVE_CALL(1);
            }
            else if (bytes & 2) {
                __REVERSED_MEMMOVE_CALL(2);
            }
            else if (bytes & 4) {
                __REVERSED_MEMMOVE_CALL(4);
            }
            else if (bytes & 8) {
                __REVERSED_MEMMOVE_CALL(8);
            }
            else if (bytes & 16) {
                __REVERSED_MEMMOVE_CALL_WITH_DISPATCH(16);
            }
            else if (bytes & 32) {
                __REVERSED_MEMMOVE_CALL_WITH_DISPATCH(32);
            }
            else if (bytes & 64) {
                __REVERSED_MEMMOVE_CALL_WITH_DISPATCH(64);
            }
            else if (bytes & 128) {
                __REVERSED_MEMMOVE_CALL_WITH_DISPATCH(128);
            }
            else if (bytes & 256) {
                __REVERSED_MEMMOVE_CALL_WITH_DISPATCH(256);
            }
            else {
                offset = bytes;

                nextDestination = (char*)nextDestination - offset; 
                nextSource      = (char*)nextSource - offset;
                
                if constexpr (_Streaming_)
                    _VectorizedMoveImplementation_::MoveStreamAligned<512>(nextDestination, nextSource, bytes >> 9);
                else
                    _VectorizedMoveImplementation_::Move<512>(nextDestination, nextSource, bytes >> 9);

                bytes = 0;
            }
        }
    }
};

template <
    bool _Aligned_,
    bool _Streaming_>
struct _MemmoveVectorizedReversedChooser<_Aligned_, _Streaming_, arch::CpuFeature::AVX512F> {
    static_assert(_Aligned_ >= _Streaming_, "Streaming loads/stores must be aligned. ");

    simd_stl_always_inline void operator()(
        void*       destination, 
        const void* source,
        sizetype    bytes) noexcept 
    {
        using _VectorizedMoveImplementation_ = _MoveVectorized<arch::CpuFeature::AVX512F, _Aligned_>;
        sizetype offset = 0;

        void* nextDestination   = static_cast<char*>(destination) + bytes;
        const void* nextSource  = static_cast<const char*>(source) + bytes;

        while (bytes)
        {
            if (bytes & 1) {
                __REVERSED_MEMMOVE_CALL(1);
            }
            else if (bytes & 2) {
                __REVERSED_MEMMOVE_CALL(2);
            }
            else if (bytes & 4) {
                __REVERSED_MEMMOVE_CALL(4);
            }
            else if (bytes & 8) {
                __REVERSED_MEMMOVE_CALL(8);
            }
            else if (bytes & 16) {
                __REVERSED_MEMMOVE_CALL_WITH_DISPATCH(16);
            }
            else if (bytes & 32) {
                __REVERSED_MEMMOVE_CALL_WITH_DISPATCH(32);
            }
            else if (bytes & 64) {
                __REVERSED_MEMMOVE_CALL_WITH_DISPATCH(64);
            }
            else if (bytes & 128) {
                __REVERSED_MEMMOVE_CALL_WITH_DISPATCH(128);
            }
            else if (bytes & 256) {
                __REVERSED_MEMMOVE_CALL_WITH_DISPATCH(256);
            }
            else if (bytes & 512) {
                __REVERSED_MEMMOVE_CALL_WITH_DISPATCH(512);
            }
            else if (bytes & 1024) {
                __REVERSED_MEMMOVE_CALL_WITH_DISPATCH(1024);
            }
            else if (bytes & 2048) {
                __REVERSED_MEMMOVE_CALL_WITH_DISPATCH(2048);
            }
            else {
                offset = bytes; 

                nextDestination = (char*)nextDestination - offset;
                nextSource      = (char*)nextSource - offset; 
                
                if constexpr (_Streaming_)
                    _VectorizedMoveImplementation_::MoveStreamAligned<4096>(nextDestination, nextSource, bytes >> 12);
                else
                    _VectorizedMoveImplementation_::Move<4096>(nextDestination, nextSource, bytes >> 12);
                
                bytes = 0;
            }
        }
    }
};

template <arch::CpuFeature _SimdGeneration_>
simd_stl_always_inline void* _MemmoveVectorizedInternal(
    void*       destination,
    const void* source,
    sizetype    bytes) noexcept
{
    using _SimdType_ = type_traits::__deduce_simd_vector_type<_SimdGeneration_, int>;
    void* returnValue = destination;

    if((((uintptr)source & (sizeof(_SimdType_) - 1)) == 0) && (((uintptr)destination & (sizeof(_SimdType_) - 1)) == 0))
    {
        if (destination < source) {
            if constexpr (type_traits::is_streaming_supported_v<_SimdGeneration_>) {
                if (bytes > __SIMD_STL_COPY_CACHE_SIZE_LIMIT) {
                    _MemmoveVectorizedChooser<true, true, _SimdGeneration_>()(destination, source, bytes);
                    return returnValue;
                }
            }

            _MemmoveVectorizedChooser<true, false, _SimdGeneration_>()(destination, source, bytes);
        }
        else {
            if constexpr (type_traits::is_streaming_supported_v<_SimdGeneration_>) {
                if (bytes > __SIMD_STL_COPY_CACHE_SIZE_LIMIT) {
                    _MemmoveVectorizedReversedChooser<true, true, _SimdGeneration_>()(destination, source, bytes);
                    return returnValue;
                }
            }

            _MemmoveVectorizedReversedChooser<true, false, _SimdGeneration_>()(destination, source, bytes);
        }
    }
    else
    {
        sizetype alignedBytes = (sizeof(_SimdType_)) - ((uintptr)destination & (sizeof(_SimdType_) - 1));

        if (destination < source) {
            if (bytes > alignedBytes) {
                void* destinationWithOffset = static_cast<char*>(destination) + alignedBytes;
                const void* sourceWithOffset = static_cast<const char*>(source) + alignedBytes;

                _MemmoveVectorizedChooser<false, false, _SimdGeneration_>()(destination, source, alignedBytes);
                _MemmoveVectorizedChooser<false, false, _SimdGeneration_>()(destinationWithOffset, sourceWithOffset, bytes - alignedBytes);
            }
            else
                _MemmoveVectorizedChooser<false, false, _SimdGeneration_>()(destination, source, bytes);
        }
        else {
            if (bytes > alignedBytes)
            {
                void* destinationWithOffset = static_cast<char*>(destination) + alignedBytes;
                const void* sourceWithOffset = static_cast<const char*>(source) + alignedBytes;

                _MemmoveVectorizedReversedChooser<false, false, _SimdGeneration_>()(destinationWithOffset, sourceWithOffset, bytes - alignedBytes);
                _MemmoveVectorizedReversedChooser<false, false, _SimdGeneration_>()(destination, source, alignedBytes);
            }
            else
                _MemmoveVectorizedReversedChooser<false, false, _SimdGeneration_>()(destination, source, bytes);
        }
    }

    return returnValue;
}

template <>
simd_stl_always_inline void* _MemmoveVectorizedInternal<arch::CpuFeature::None>(
    void*       destination,
    const void* source,
    sizetype    bytes) noexcept
{

    if (destination < source)
        _MemmoveVectorizedChooser<false, false, arch::CpuFeature::None>()(destination, source, bytes);
    else
        _MemmoveVectorizedReversedChooser<false, false, arch::CpuFeature::None>()(destination, source, bytes);

    return destination;
}

void* __memmove_vectorized(
    void*       destination,
    const void* source,
    sizetype    bytes) noexcept
{
    if (arch::ProcessorFeatures::AVX512F())
        return _MemmoveVectorizedInternal<arch::CpuFeature::AVX512F>(destination, source, bytes);
    else if (arch::ProcessorFeatures::AVX2())
        return _MemmoveVectorizedInternal<arch::CpuFeature::AVX2>(destination, source, bytes);
    else if (arch::ProcessorFeatures::AVX())
        return _MemmoveVectorizedInternal<arch::CpuFeature::AVX>(destination, source, bytes);
    else if (arch::ProcessorFeatures::SSE41())
        return _MemmoveVectorizedInternal<arch::CpuFeature::SSE41>(destination, source, bytes);
    else if (arch::ProcessorFeatures::SSE2())
        return _MemmoveVectorizedInternal<arch::CpuFeature::SSE2>(destination, source, bytes);

    return _MemmoveVectorizedInternal<arch::CpuFeature::None>(destination, source, bytes);
}

__SIMD_STL_ALGORITHM_NAMESPACE_END
