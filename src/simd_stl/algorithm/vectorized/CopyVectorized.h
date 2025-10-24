#pragma once

#include <src/simd_stl/algorithm/AdvanceBytes.h>
#include <simd_stl/compatibility/SimdCompatibility.h>

#include <simd_stl/compatibility/FunctionAttributes.h>
#include <simd_stl/arch/ProcessorFeatures.h>

__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

#ifndef __SIMD_STL_DEFINE_DEFAULT_MEMCPY
#define __SIMD_STL_DEFINE_DEFAULT_MEMCPY(...)                           \
    template <sizetype byteCount, bool aligned>                     \
    static void* Memcpy(         \
        void*               destination,                            \
        const void* const   source,                                 \
        sizetype            size) noexcept                          \
    {                                                               \
        static_assert(                                              \
            simd_stl::arch::Contains<byteCount, __VA_ARGS__>::value,    \
            "simd_stl::Memcpy: Unsupported byteCount. ");       \
    }   
#endif // __SIMD_STL_DEFINE_DEFAULT_MEMCPY



#ifndef __SIMD_STL_DEFINE_MEMCPY
#define __SIMD_STL_DEFINE_MEMCPY(bytesCount, aligned, copyType, copyCommand)            \
    template <>                                                                     \
    static void* Memcpy<bytesCount, aligned>(      \
        void*               destination,                                            \
        const void* const   source,                                                 \
        sizetype            size) noexcept                                          \
    {                                                                               \
        copyType* dest          = reinterpret_cast<copyType*>(destination);         \
        const copyType* src     = reinterpret_cast<const copyType*>(source);        \
        while (size--) { copyCommand; }                                             \
        return destination;                                                         \
    }
#endif // __SIMD_STL_DEFINE_MEMCPY



SIMD_STL_DECLARE_CPU_FEATURE_GUARDED_CLASS(
	template <arch::CpuFeature feature> 
    class MemcpyImplementationInternal,
    feature,
    "",
    arch::CpuFeature::None, arch::CpuFeature::SSE2,
    arch::CpuFeature::AVX, arch::CpuFeature::AVX512F
);

template <>
class MemcpyImplementationInternal<arch::CpuFeature::None> {
public:

#if defined(simd_stl_processor_x86_64)
    __SIMD_STL_DEFINE_DEFAULT_MEMCPY(1, 2, 4, 8)
#else 
    __SIMD_STL_DEFINE_DEFAULT_MEMCPY(1, 2, 4)
#endif

    __SIMD_STL_DEFINE_MEMCPY(1, false, uint8, SIMD_STL_ECHO(*dest++ = *src++;));
    __SIMD_STL_DEFINE_MEMCPY(2, false, uint16, SIMD_STL_ECHO(*dest++ = *src++;));
    __SIMD_STL_DEFINE_MEMCPY(4, false, uint32, SIMD_STL_ECHO(*dest++ = *src++;));

#if defined(simd_stl_processor_x86_64)
    __SIMD_STL_DEFINE_MEMCPY(8, false, uint64, SIMD_STL_ECHO(*dest++ = *src++;));
#endif
};

template <>
class MemcpyImplementationInternal<arch::CpuFeature::SSE2> {
public:
    __SIMD_STL_DEFINE_DEFAULT_MEMCPY(16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)

    // Unaligned

    __SIMD_STL_DEFINE_MEMCPY(16,    false, __m128i, SIMD_STL_ECHO(                                         _mm_storeu_si128(dest++, _mm_loadu_si128(src++))))
    __SIMD_STL_DEFINE_MEMCPY(32,    false, __m128i, SIMD_STL_ECHO(__SIMD_STL_REPEAT_N(2,                       _mm_storeu_si128(dest++, _mm_loadu_si128(src++)))))
    __SIMD_STL_DEFINE_MEMCPY(64,    false, __m128i, SIMD_STL_ECHO(__SIMD_STL_REPEAT_N(4,                       _mm_storeu_si128(dest++, _mm_loadu_si128(src++)))))
    __SIMD_STL_DEFINE_MEMCPY(128,   false, __m128i, SIMD_STL_ECHO(__SIMD_STL_REPEAT_N(8,                       _mm_storeu_si128(dest++, _mm_loadu_si128(src++)))))
    __SIMD_STL_DEFINE_MEMCPY(256,   false, __m128i, SIMD_STL_ECHO(__SIMD_STL_REPEAT_N(16,                      _mm_storeu_si128(dest++, _mm_loadu_si128(src++)))))
    __SIMD_STL_DEFINE_MEMCPY(512,   false, __m128i, SIMD_STL_ECHO(__SIMD_STL_REPEAT_N(32,                      _mm_storeu_si128(dest++, _mm_loadu_si128(src++)))))
    __SIMD_STL_DEFINE_MEMCPY(1024,  false, __m128i, SIMD_STL_ECHO(__SIMD_STL_REPEAT_N(64,                      _mm_storeu_si128(dest++, _mm_loadu_si128(src++)))))
    __SIMD_STL_DEFINE_MEMCPY(2048,  false, __m128i, SIMD_STL_ECHO(__SIMD_STL_REPEAT_N(2, (__SIMD_STL_REPEAT_N(64,  _mm_storeu_si128(dest++, _mm_loadu_si128(src++)))))))
    __SIMD_STL_DEFINE_MEMCPY(4096,  false, __m128i, SIMD_STL_ECHO(__SIMD_STL_REPEAT_N(4, (__SIMD_STL_REPEAT_N(64,  _mm_storeu_si128(dest++, _mm_loadu_si128(src++)))))))
    __SIMD_STL_DEFINE_MEMCPY(8192,  false, __m128i, SIMD_STL_ECHO(__SIMD_STL_REPEAT_N(8, (__SIMD_STL_REPEAT_N(64,  _mm_storeu_si128(dest++, _mm_loadu_si128(src++)))))))

    // Aligned

    __SIMD_STL_DEFINE_MEMCPY(16,    true, __m128i, SIMD_STL_ECHO(                                         _mm_store_si128(dest++, _mm_load_si128(src++))))
    __SIMD_STL_DEFINE_MEMCPY(32,    true, __m128i, SIMD_STL_ECHO(__SIMD_STL_REPEAT_N(2,                       _mm_store_si128(dest++, _mm_load_si128(src++)))))
    __SIMD_STL_DEFINE_MEMCPY(64,    true, __m128i, SIMD_STL_ECHO(__SIMD_STL_REPEAT_N(4,                       _mm_store_si128(dest++, _mm_load_si128(src++)))))
    __SIMD_STL_DEFINE_MEMCPY(128,   true, __m128i, SIMD_STL_ECHO(__SIMD_STL_REPEAT_N(8,                       _mm_store_si128(dest++, _mm_load_si128(src++)))))
    __SIMD_STL_DEFINE_MEMCPY(256,   true, __m128i, SIMD_STL_ECHO(__SIMD_STL_REPEAT_N(16,                      _mm_store_si128(dest++, _mm_load_si128(src++)))))
    __SIMD_STL_DEFINE_MEMCPY(512,   true, __m128i, SIMD_STL_ECHO(__SIMD_STL_REPEAT_N(32,                      _mm_store_si128(dest++, _mm_load_si128(src++)))))
    __SIMD_STL_DEFINE_MEMCPY(1024,  true, __m128i, SIMD_STL_ECHO(__SIMD_STL_REPEAT_N(64,                      _mm_store_si128(dest++, _mm_load_si128(src++)))))
    __SIMD_STL_DEFINE_MEMCPY(2048,  true, __m128i, SIMD_STL_ECHO(__SIMD_STL_REPEAT_N(2, (__SIMD_STL_REPEAT_N(64,  _mm_store_si128(dest++, _mm_load_si128(src++)))))))
    __SIMD_STL_DEFINE_MEMCPY(4096,  true, __m128i, SIMD_STL_ECHO(__SIMD_STL_REPEAT_N(4, (__SIMD_STL_REPEAT_N(64,  _mm_store_si128(dest++, _mm_load_si128(src++)))))))
    __SIMD_STL_DEFINE_MEMCPY(8192,  true, __m128i, SIMD_STL_ECHO(__SIMD_STL_REPEAT_N(8, (__SIMD_STL_REPEAT_N(64,  _mm_store_si128(dest++, _mm_load_si128(src++)))))))
};

template <>
class MemcpyImplementationInternal<arch::CpuFeature::AVX> {
public:
    __SIMD_STL_DEFINE_DEFAULT_MEMCPY(32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384)
       
    // Unaligned

    __SIMD_STL_DEFINE_MEMCPY(32,    false, __m256i, SIMD_STL_ECHO(                                         _mm256_storeu_si256(dest++, _mm256_lddqu_si256(src++))))
    __SIMD_STL_DEFINE_MEMCPY(64,    false, __m256i, SIMD_STL_ECHO(__SIMD_STL_REPEAT_N(2,                       _mm256_storeu_si256(dest++, _mm256_lddqu_si256(src++)))))
    __SIMD_STL_DEFINE_MEMCPY(128,   false, __m256i, SIMD_STL_ECHO(__SIMD_STL_REPEAT_N(4,                       _mm256_storeu_si256(dest++, _mm256_lddqu_si256(src++)))))
    __SIMD_STL_DEFINE_MEMCPY(256,   false, __m256i, SIMD_STL_ECHO(__SIMD_STL_REPEAT_N(8,                       _mm256_storeu_si256(dest++, _mm256_lddqu_si256(src++)))))
    __SIMD_STL_DEFINE_MEMCPY(512,   false, __m256i, SIMD_STL_ECHO(__SIMD_STL_REPEAT_N(16,                      _mm256_storeu_si256(dest++, _mm256_lddqu_si256(src++)))))
    __SIMD_STL_DEFINE_MEMCPY(1024,  false, __m256i, SIMD_STL_ECHO(__SIMD_STL_REPEAT_N(32,                      _mm256_storeu_si256(dest++, _mm256_lddqu_si256(src++)))))
    __SIMD_STL_DEFINE_MEMCPY(2048,  false, __m256i, SIMD_STL_ECHO(__SIMD_STL_REPEAT_N(64,                      _mm256_storeu_si256(dest++, _mm256_lddqu_si256(src++)))))
    __SIMD_STL_DEFINE_MEMCPY(4096,  false, __m256i, SIMD_STL_ECHO(__SIMD_STL_REPEAT_N(2, (__SIMD_STL_REPEAT_N(64,   _mm256_storeu_si256(dest++, _mm256_lddqu_si256(src++)))))))
    __SIMD_STL_DEFINE_MEMCPY(8192,  false, __m256i, SIMD_STL_ECHO(__SIMD_STL_REPEAT_N(4, (__SIMD_STL_REPEAT_N(64,   _mm256_storeu_si256(dest++, _mm256_lddqu_si256(src++)))))))
    __SIMD_STL_DEFINE_MEMCPY(16384, false, __m256i, SIMD_STL_ECHO(__SIMD_STL_REPEAT_N(8, (__SIMD_STL_REPEAT_N(64,   _mm256_storeu_si256(dest++, _mm256_lddqu_si256(src++)))))))

    // Aligned

    __SIMD_STL_DEFINE_MEMCPY(32,    true, __m256i, SIMD_STL_ECHO(                                               _mm256_store_si256(dest++, _mm256_load_si256(src++))))
    __SIMD_STL_DEFINE_MEMCPY(64,    true, __m256i, SIMD_STL_ECHO(__SIMD_STL_REPEAT_N(2,                         _mm256_store_si256(dest++, _mm256_load_si256(src++)))))
    __SIMD_STL_DEFINE_MEMCPY(128,   true, __m256i, SIMD_STL_ECHO(__SIMD_STL_REPEAT_N(4,                         _mm256_store_si256(dest++, _mm256_load_si256(src++)))))
    __SIMD_STL_DEFINE_MEMCPY(256,   true, __m256i, SIMD_STL_ECHO(__SIMD_STL_REPEAT_N(8,                         _mm256_store_si256(dest++, _mm256_load_si256(src++)))))
    __SIMD_STL_DEFINE_MEMCPY(512,   true, __m256i, SIMD_STL_ECHO(__SIMD_STL_REPEAT_N(16,                        _mm256_store_si256(dest++, _mm256_load_si256(src++)))))
    __SIMD_STL_DEFINE_MEMCPY(1024,  true, __m256i, SIMD_STL_ECHO(__SIMD_STL_REPEAT_N(32,                        _mm256_store_si256(dest++, _mm256_load_si256(src++)))))
    __SIMD_STL_DEFINE_MEMCPY(2048,  true, __m256i, SIMD_STL_ECHO(__SIMD_STL_REPEAT_N(64,                        _mm256_store_si256(dest++, _mm256_load_si256(src++)))))
    __SIMD_STL_DEFINE_MEMCPY(4096,  true, __m256i, SIMD_STL_ECHO(__SIMD_STL_REPEAT_N(2, (__SIMD_STL_REPEAT_N(64,     _mm256_store_si256(dest++, _mm256_load_si256(src++)))))))
    __SIMD_STL_DEFINE_MEMCPY(8192,  true, __m256i, SIMD_STL_ECHO(__SIMD_STL_REPEAT_N(4, (__SIMD_STL_REPEAT_N(64,     _mm256_store_si256(dest++, _mm256_load_si256(src++)))))))
    __SIMD_STL_DEFINE_MEMCPY(16384, true, __m256i, SIMD_STL_ECHO(__SIMD_STL_REPEAT_N(8, (__SIMD_STL_REPEAT_N(64,     _mm256_store_si256(dest++, _mm256_load_si256(src++)))))))
};

template <> 
class MemcpyImplementationInternal<arch::CpuFeature::AVX512F> {
public:
    __SIMD_STL_DEFINE_DEFAULT_MEMCPY(64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768)

    // Unaligned

    __SIMD_STL_DEFINE_MEMCPY(64,    false, __m512i, SIMD_STL_ECHO(                                         _mm512_storeu_si512(dest++, _mm512_loadu_si512(src++))))
    __SIMD_STL_DEFINE_MEMCPY(128,   false, __m512i, SIMD_STL_ECHO(__SIMD_STL_REPEAT_N(2,                       _mm512_storeu_si512(dest++, _mm512_loadu_si512(src++)))))
    __SIMD_STL_DEFINE_MEMCPY(256,   false, __m512i, SIMD_STL_ECHO(__SIMD_STL_REPEAT_N(4,                       _mm512_storeu_si512(dest++, _mm512_loadu_si512(src++)))))
    __SIMD_STL_DEFINE_MEMCPY(512,   false, __m512i, SIMD_STL_ECHO(__SIMD_STL_REPEAT_N(8,                       _mm512_storeu_si512(dest++, _mm512_loadu_si512(src++)))))
    __SIMD_STL_DEFINE_MEMCPY(1024,  false, __m512i, SIMD_STL_ECHO(__SIMD_STL_REPEAT_N(16,                      _mm512_storeu_si512(dest++, _mm512_loadu_si512(src++)))))
    __SIMD_STL_DEFINE_MEMCPY(2048,  false, __m512i, SIMD_STL_ECHO(__SIMD_STL_REPEAT_N(32,                      _mm512_storeu_si512(dest++, _mm512_loadu_si512(src++)))))
    __SIMD_STL_DEFINE_MEMCPY(4096,  false, __m512i, SIMD_STL_ECHO(__SIMD_STL_REPEAT_N(64,                      _mm512_storeu_si512(dest++, _mm512_loadu_si512(src++)))))
    __SIMD_STL_DEFINE_MEMCPY(8192,  false, __m512i, SIMD_STL_ECHO(__SIMD_STL_REPEAT_N(2, (__SIMD_STL_REPEAT_N(64,   _mm512_storeu_si512(dest++, _mm512_loadu_si512(src++)))))))
    __SIMD_STL_DEFINE_MEMCPY(16384, false, __m512i, SIMD_STL_ECHO(__SIMD_STL_REPEAT_N(4, (__SIMD_STL_REPEAT_N(64,   _mm512_storeu_si512(dest++, _mm512_loadu_si512(src++)))))))
    __SIMD_STL_DEFINE_MEMCPY(32768, false, __m512i, SIMD_STL_ECHO(__SIMD_STL_REPEAT_N(8, (__SIMD_STL_REPEAT_N(64,   _mm512_storeu_si512(dest++, _mm512_loadu_si512(src++)))))))

    // Aligned

    __SIMD_STL_DEFINE_MEMCPY(64,    true, __m512i, SIMD_STL_ECHO(                                         _mm512_store_si512(dest++, _mm512_load_si512(src++))))
    __SIMD_STL_DEFINE_MEMCPY(128,   true, __m512i, SIMD_STL_ECHO(__SIMD_STL_REPEAT_N(2,                       _mm512_store_si512(dest++, _mm512_load_si512(src++)))))
    __SIMD_STL_DEFINE_MEMCPY(256,   true, __m512i, SIMD_STL_ECHO(__SIMD_STL_REPEAT_N(4,                       _mm512_store_si512(dest++, _mm512_load_si512(src++)))))
    __SIMD_STL_DEFINE_MEMCPY(512,   true, __m512i, SIMD_STL_ECHO(__SIMD_STL_REPEAT_N(8,                       _mm512_store_si512(dest++, _mm512_load_si512(src++)))))
    __SIMD_STL_DEFINE_MEMCPY(1024,  true, __m512i, SIMD_STL_ECHO(__SIMD_STL_REPEAT_N(16,                      _mm512_store_si512(dest++, _mm512_load_si512(src++)))))
    __SIMD_STL_DEFINE_MEMCPY(2048,  true, __m512i, SIMD_STL_ECHO(__SIMD_STL_REPEAT_N(32,                      _mm512_store_si512(dest++, _mm512_load_si512(src++)))))
    __SIMD_STL_DEFINE_MEMCPY(4096,  true, __m512i, SIMD_STL_ECHO(__SIMD_STL_REPEAT_N(64,                      _mm512_store_si512(dest++, _mm512_load_si512(src++)))))
    __SIMD_STL_DEFINE_MEMCPY(8192,  true, __m512i, SIMD_STL_ECHO(__SIMD_STL_REPEAT_N(2, (__SIMD_STL_REPEAT_N(64,   _mm512_store_si512(dest++, _mm512_load_si512(src++)))))))
    __SIMD_STL_DEFINE_MEMCPY(16384, true, __m512i, SIMD_STL_ECHO(__SIMD_STL_REPEAT_N(4, (__SIMD_STL_REPEAT_N(64,   _mm512_store_si512(dest++, _mm512_load_si512(src++)))))))
    __SIMD_STL_DEFINE_MEMCPY(32768, true, __m512i, SIMD_STL_ECHO(__SIMD_STL_REPEAT_N(8, (__SIMD_STL_REPEAT_N(64,   _mm512_store_si512(dest++, _mm512_load_si512(src++)))))))
};

template <arch::CpuFeature _SimdGeneration_>
simd_stl_declare_const_function simd_stl_always_inline void CopyVectorizedInternal(
    const void*     source,
    void*           destination,
    sizetype	    bytes) noexcept
{
    AssertUnreachable();
    return nullptr;
}

template <>
simd_stl_declare_const_function simd_stl_always_inline void CopyVectorizedInternal<arch::CpuFeature::None>(
    const void*     source,
    void*           destination,
    sizetype	    bytes) noexcept
{
    using _None_ = MemcpyImplementationInternal<arch::CpuFeature::None>;
    sizetype offset = 0;

    while (bytes)
    {
        if (bytes < 2) // 1 byte
        {
            _None_::Memcpy<1, false>(destination, source, bytes);
            offset = bytes & -1;
            destination = (char*)destination + offset;
            source = (char*)source + offset;
            bytes = 0;

        }
        else if (bytes < 4) // 2 bytes
        {
            _None_::Memcpy<2, false>(destination, source, bytes >> 1);
            offset = bytes & -2;
            destination = (char*)destination + offset;
            source = (char*)source + offset;
            bytes &= 1;
        }
        else if (bytes < 8) // 4 bytes
        {
            _None_::Memcpy<4, false>(destination, source, bytes >> 2);
            offset = bytes & -4;
            destination = (char*)destination + offset;
            source = (char*)source + offset;
            bytes &= 3;
        }
        else // 8 bytes
        {
            _None_::Memcpy<8, false>(destination, source, bytes >> 3);
            offset = bytes & -8;
            destination = (char*)destination + offset;
            source = (char*)source + offset;
            bytes &= 7;
        }
    }
}

template <>
simd_stl_declare_const_function simd_stl_always_inline void CopyVectorizedInternal<arch::CpuFeature::SSE2>(
    const void* source,
    void*       destination,
    sizetype    size) noexcept
{
    using _None_ = MemcpyImplementationInternal<arch::CpuFeature::None>;
    using _Sse_ = MemcpyImplementationInternal<arch::CpuFeature::SSE2>;

    sizetype offset = 0;

    while (size)
    {
        if (size < 2) // 1 byte
        {
            _None_::Memcpy<1, false>(destination, source, size);
            offset = size & -1;
            destination = (char*)destination + offset;
            source = (char*)source + offset;
            size = 0;

        }
        else if (size < 4) // 2 bytes
        {
            _None_::Memcpy<2, false>(destination, source, size >> 1);
            offset = size & -2;
            destination = (char*)destination + offset;
            source = (char*)source + offset;
            size &= 1;
        }
        else if (size < 8) // 4 bytes
        {
            _None_::Memcpy<4, false>(destination, source, size >> 2);
            offset = size & -4;
            destination = (char*)destination + offset;
            source = (char*)source + offset;
            size &= 3;
        }
        else if (size < 16) // 8 bytes
        {
            _None_::Memcpy<8, false>(destination, source, size >> 3);
            offset = size & -8;
            destination = (char*)destination + offset;
            source = (char*)source + offset;
            size &= 7;
        }
        else if (size < 32) // 16 bytes
        {
            _Sse_::Memcpy<16, false>(destination, source, size >> 4);
            offset = size & -16;
            destination = (char*)destination + offset;
            source = (char*)source + offset;
            size &= 15;
        }
        else if (size < 64) // 32 bytes
        {
            _Sse_::Memcpy<32, false>(destination, source, size >> 5);
            offset = size & -32;
            destination = (char*)destination + offset;
            source = (char*)source + offset;
            size &= 31;
        }
        else if (size < 128) // 64 bytes
        {
            _Sse_::Memcpy<64, false>(destination, source, size >> 6);
            offset = size & -64;
            destination = (char*)destination + offset;
            source = (char*)source + offset;
            size &= 63;
        }
        else if (size < 256) // 128 bytes
        {
            _Sse_::Memcpy<128, false>(destination, source, size >> 7);
            offset = size & -128;
            destination = (char*)destination + offset;
            source = (char*)source + offset;
            size &= 127;
        }
        else // 256 bytes
        {
            _Sse_::Memcpy<256, false>(destination, source, size >> 8);
            offset = size & -256;
            destination = (char*)destination + offset;
            source = (char*)source + offset;
            size &= 255;
        }
    }
}

template <>
simd_stl_declare_const_function simd_stl_always_inline void CopyVectorizedInternal<arch::CpuFeature::AVX>(
    const void* source,
    void*       destination,
    sizetype    size) noexcept
{
    using _None_ = MemcpyImplementationInternal<arch::CpuFeature::None>;

    using _Sse_ = MemcpyImplementationInternal<arch::CpuFeature::SSE2>;
    using _Avx_ = MemcpyImplementationInternal<arch::CpuFeature::AVX>;

    sizetype offset = 0;

    while (size)
    {
        if (size < 2) // 1 byte
        {
            _None_::Memcpy<1, false>(destination, source, size);
            offset = size & -1;
            destination = (char*)destination + offset;
            source = (char*)source + offset;
            size = 0;

        }
        else if (size < 4) // 2 bytes
        {
            _None_::Memcpy<2, false>(destination, source, size >> 1);
            offset = size & -2;
            destination = (char*)destination + offset;
            source = (char*)source + offset;
            size &= 1;
        }
        else if (size < 8) // 4 bytes
        {
            _None_::Memcpy<4, false>(destination, source, size >> 2);
            offset = size & -4;
            destination = (char*)destination + offset;
            source = (char*)source + offset;
            size &= 3;
        }
        else if (size < 16) // 8 bytes
        {
            _None_::Memcpy<8, false>(destination, source, size >> 3);
            offset = size & -8;
            destination = (char*)destination + offset;
            source = (char*)source + offset;
            size &= 7;
        }
        else if (size < 32) // 16 bytes
        {
            _Sse_::Memcpy<16, false>(destination, source, size >> 4);
            offset = size & -16;
            destination = (char*)destination + offset;
            source = (char*)source + offset;
            size &= 15;
        }
        else if (size < 64) // 32 bytes
        {
            _Avx_::Memcpy<32, false>(destination, source, size >> 5);
            offset = size & -32;
            destination = (char*)destination + offset;
            source = (char*)source + offset;
            size &= 31;
        }
        else if (size < 128) // 64 bytes
        {
            _Avx_::Memcpy<64, false>(destination, source, size >> 6);
            offset = size & -64;
            destination = (char*)destination + offset;
            source = (char*)source + offset;
            size &= 63;
        }
        else if (size < 256) // 128 bytes
        {
            _Avx_::Memcpy<128, false>(destination, source, size >> 7);
            offset = size & -128;
            destination = (char*)destination + offset;
            source = (char*)source + offset;
            size &= 127;
        }
        else if (size < 512) // 256 bytes
        {
            _Avx_::Memcpy<256, false>(destination, source, size >> 8);
            offset = size & -256;
            destination = (char*)destination + offset;
            source = (char*)source + offset;
            size &= 255;
        }
        else // 512 bytes
        {
            _Avx_::Memcpy<512, false>(destination, source, size >> 9);
            offset = size & -512;
            destination = (char*)destination + offset;
            source = (char*)source + offset;
            size &= 511;
        }
    }
}

template <>
simd_stl_declare_const_function simd_stl_always_inline void CopyVectorizedInternal<arch::CpuFeature::AVX512F>(
    const void* source,
    void*       destination,
    sizetype    size) noexcept
{
    using _None_ = MemcpyImplementationInternal<arch::CpuFeature::None>;

    using _Sse_ = MemcpyImplementationInternal<arch::CpuFeature::SSE2>;
    using _Avx_ = MemcpyImplementationInternal<arch::CpuFeature::AVX>;
    using _Avx512F_ = MemcpyImplementationInternal<arch::CpuFeature::AVX512F>;
	
    sizetype offset = 0;

    while (size)
    {
        if (size < 2) // 1 byte
        {
            _None_::Memcpy<1, false>(destination, source, size);
            offset = size & -1;
            destination = (char*)destination + offset;
            source = (char*)source + offset;
            size = 0;
            
        }
        else if (size < 4) // 2 bytes
        {
            _None_::Memcpy<2, false>(destination, source, size >> 1);
            offset = size & -2;
            destination = (char*)destination + offset;
            source = (char*)source + offset;
            size &= 1;
        }
        else if (size < 8) // 4 bytes
        {
            _None_::Memcpy<4, false>(destination, source, size >> 2);
            offset = size & -4;
            destination = (char*)destination + offset;
            source = (char*)source + offset;
            size &= 3;
        }
        else if (size < 16) // 8 bytes
        {
            _None_::Memcpy<8, false>(destination, source, size >> 3);
            offset = size & -8;
            destination = (char*)destination + offset;
            source = (char*)source + offset;
            size &= 7;
        }
        else if (size < 32) // 16 bytes
        {
            _Sse_::Memcpy<16, false>(destination, source, size >> 4);
            offset = size & -16;
            destination = (char*)destination + offset;
            source = (char*)source + offset;
            size &= 15;
        }
        else if (size < 64) // 32 bytes
        {
            _Avx_::Memcpy<32, false>(destination, source, size >> 5);
            offset = size & -32;
            destination = (char*)destination + offset;
            source = (char*)source + offset;
            size &= 31;
        }
        else if (size < 128) // 64 bytes
        {
            _Avx512F_::Memcpy<64, false>(destination, source, size >> 6);
            offset = size & -64;
            destination = (char*)destination + offset;
            source = (char*)source + offset;
            size &= 63;
        }
        else if (size < 256) // 128 bytes
        {
            _Avx512F_::Memcpy<128, false>(destination, source, size >> 7);
            offset = size & -128;
            destination = (char*)destination + offset;
            source = (char*)source + offset;
            size &= 127;
        }
        else if (size < 512) // 256 bytes
        {
            _Avx512F_::Memcpy<256, false>(destination, source, size >> 8);
            offset = size & -256;
            destination = (char*)destination + offset;
            source = (char*)source + offset;
            size &= 255;
        }
        else if (size < 1024) // 512 bytes
        {
            _Avx512F_::Memcpy<512, false>(destination, source, size >> 9);
            offset = size & -512;
            destination = (char*)destination + offset;
            source = (char*)source + offset;
            size &= 511;
        }
        else if (size < 2048) // 1024 bytes (1 kB)
        {
            _Avx512F_::Memcpy<1024, false>(destination, source, size >> 10);
            offset = size & -1024;
            destination = (char*)destination + offset;
            source = (char*)source + offset;
            size &= 1023;
        }
        else if (size < 4096) // 2048 bytes (2 kB)
        {
            _Avx512F_::Memcpy<2048, false>(destination, source, size >> 11);
            offset = size & -2048;
            destination = (char*)destination + offset;
            source = (char*)source + offset;
            size &= 2047;
        }
        else if (size < 8192) // 4096 bytes (4 kB)
        {
            _Avx512F_::Memcpy<4096, false>(destination, source, size >> 12);
            offset = size & -4096;
            destination = (char*)destination + offset;
            source = (char*)source + offset;
            size &= 4095;
        }
        else if (size < 16384) // 8192 bytes (8 kB)
        {
            _Avx512F_::Memcpy<8192, false>(destination, source, size >> 13);
            offset = size & -8192;
            destination = (char*)destination + offset;
            source = (char*)source + offset;
            size &= 8191;
        }
        else if (size < 32768) // 8192 bytes (8 kB)
        {
            _Avx512F_::Memcpy<16384, false>(destination, source, size >> 14);
            offset = size & -16384;
            destination = (char*)destination + offset;
            source = (char*)source + offset;
            size &= 16383;
        }
        else 
        {
            _Avx512F_::Memcpy<32768, false>(destination, source, size >> 15);
            offset = size & -32768;
            destination = (char*)destination + offset;
            source = (char*)source + offset;
            size &= 32767;
        }
    }
}

simd_stl_declare_const_function simd_stl_always_inline void CopyVectorized(
	const void*	from,
	void*		to,
	sizetype	bytes) noexcept
{
    if (arch::ProcessorFeatures::AVX512F())
        return CopyVectorizedInternal<arch::CpuFeature::AVX512F>(from, to, bytes);
    else if (arch::ProcessorFeatures::AVX())
        return CopyVectorizedInternal<arch::CpuFeature::AVX>(from, to, bytes);
    else if (arch::ProcessorFeatures::SSE2())
        return CopyVectorizedInternal<arch::CpuFeature::SSE2>(from, to, bytes);

    return CopyVectorizedInternal<arch::CpuFeature::None>(from, to, bytes);
}

__SIMD_STL_ALGORITHM_NAMESPACE_END
