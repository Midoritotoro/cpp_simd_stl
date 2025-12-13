#pragma once

#include <simd_stl/numeric/BasicSimd.h>
#include <simd_stl/memory/Alignment.h>

#define __SIMD_STL_FILL_CACHE_SIZE_LIMIT 3*1024*1024


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <sizetype _Step_>
struct _Memset_step_type {
    using type = void;
};

template <>
struct _Memset_step_type<1> {
    using type = uint8;
};

template <>
struct _Memset_step_type<2> {
    using type = uint16;
};

template <>
struct _Memset_step_type<4> {
    using type = uint32;
};

template <>
struct _Memset_step_type<8> {
    using type = uint64;
};

template <>
struct _Memset_step_type<16> {
    using type = __m128i;
};

template <>
struct _Memset_step_type<32> {
    using type = __m256i;
};

template <>
struct _Memset_step_type<64> {
    using type = __m512i;
};

template <sizetype _Count>
using _Step_type = typename _Memset_step_type<_Count>::type;

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Type_,
    bool                _Aligned_>
class _VectorizedMemsetImplementation;

template <typename _Type_>
simd_stl_always_inline void* _MemsetScalar(
    void*       _Destination,
    _Type_      _Value,
    sizetype    _Length) noexcept
{
    _Type_* _Pointer = static_cast<_Type_*>(_Destination);

    while (_Length--)
        *_Pointer++ = _Value;

    return _Destination;
}

#pragma region Vectorized Memcpy Sse2

template <typename _Type_>
class _VectorizedMemsetImplementation<arch::CpuFeature::SSE2, _Type_, false>
{
public:
    template <sizetype _Step_>
    static void* _Memset(
        void*       _Destination,
        _Type_      _Value,
        sizetype    _Length) noexcept
    {
        AssertUnreachable();
        return nullptr;
    }

    template <>
    static void* _Memset<16>(
        void*       _Destination,
        _Type_      _Value,
        sizetype    _Length) noexcept
    {
        __m128i* _Pointer = static_cast<__m128i*>(_Destination);

        while (_Length--) {
            _mm_storeu_si128(_Pointer++, _Value);
        }

        return _Destination;
    }

    template <>
    static void* _Memset<32>(
        void*       _Destination,
        _Type_      _Value,
        sizetype    _Length) noexcept
    {
        __m128i* _Pointer = static_cast<__m128i*>(_Destination);

        while (_Length--) {
            __SIMD_STL_REPEAT_N(2, _mm_storeu_si128(_Pointer++, _Value));
        }

        return _Destination;
    }

    template <>
    static void* _Memset<64>(
        void*       _Destination,
        _Type_      _Value,
        sizetype    _Length) noexcept
    {
        __m128i* _Pointer = static_cast<__m128i*>(_Destination);

        while (_Length--) {
            __SIMD_STL_REPEAT_N(4, _mm_storeu_si128(_Pointer++, _Value));
        }

        return _Destination;
    }

    template <>
    static void* _Memset<128>(
        void*       _Destination,
        _Type_      _Value,
        sizetype    _Length) noexcept
    {
        __m128i* _Pointer = static_cast<__m128i*>(_Destination);

        while (_Length--) {
            __SIMD_STL_REPEAT_N(8, _mm_storeu_si128(_Pointer++, _Value));
        }

        return _Destination;
    }

    template <>
    static void* _Memset<256>(
        void* _Destination,
        _Type_      _Value,
        sizetype    _Length) noexcept
    {
        __m128i* _Pointer = static_cast<__m128i*>(_Destination);

        while (_Length--) {
            __SIMD_STL_REPEAT_N(16, _mm_storeu_si128(_Pointer++, _Value));
        }

        return _Destination;
    }
};

template <typename _Type_>
class _VectorizedMemsetImplementation<arch::CpuFeature::SSE2, _Type_, true>
{
public:
    template <sizetype _Step_>
    static void* _Memset(
        void*       _Destination,
        _Type_      _Value,
        sizetype    _Length) noexcept
    {
        AssertUnreachable();
        return nullptr;
    }

    template <>
    static void* _Memset<16>(
        void*       _Destination,
        _Type_      _Value,
        sizetype    _Length) noexcept
    {
        __m128i* _Pointer = static_cast<__m128i*>(_Destination);

        while (_Length--) {
            _mm_store_si128(_Pointer++, _Value);
        }

        return _Destination;
    }

    template <>
    static void* _Memset<32>(
        void*       _Destination,
        _Type_      _Value,
        sizetype    _Length) noexcept
    {
        __m128i* _Pointer = static_cast<__m128i*>(_Destination);

        while (_Length--) {
            __SIMD_STL_REPEAT_N(2, _mm_store_si128(_Pointer++, _Value));
        }

        return _Destination;
    }

    template <>
    static void* _Memset<64>(
        void*       _Destination,
        _Type_      _Value,
        sizetype    _Length) noexcept
    {
        __m128i* _Pointer = static_cast<__m128i*>(_Destination);

        while (_Length--) {
            __SIMD_STL_REPEAT_N(4, _mm_store_si128(_Pointer++, _Value));
        }

        return _Destination;
    }

    template <>
    static void* _Memset<128>(
        void*       _Destination,
        _Type_      _Value,
        sizetype    _Length) noexcept
    {
        __m128i* _Pointer = static_cast<__m128i*>(_Destination);

        while (_Length--) {
            __SIMD_STL_REPEAT_N(8, _mm_store_si128(_Pointer++, _Value));
        }

        return _Destination;
    }

    template <>
    static void* _Memset<256>(
        void* _Destination,
        _Type_      _Value,
        sizetype    _Length) noexcept
    {
        __m128i* _Pointer = static_cast<__m128i*>(_Destination);

        while (_Length--) {
            __SIMD_STL_REPEAT_N(16, _mm_store_si128(_Pointer++, _Value));
        }

        return _Destination;
    }

    template <sizetype _Step_>
    static void* _MemsetAlignedStreaming(
        void*       _Destination,
        _Type_      _Value,
        sizetype    _Length) noexcept
    {
        AssertUnreachable();
        return nullptr;
    }
    

    template <>
    static void* _MemsetAlignedStreaming<16>(
        void*       _Destination,
        _Type_      _Value,
        sizetype    _Length) noexcept
    {
        __m128i* _Pointer = static_cast<__m128i*>(_Destination);

        while (_Length--) {
            _mm_stream_si128(_Pointer++, _Value);
        }

        _mm_sfence();

        return _Destination;
    }

    template <>
    static void* _MemsetAlignedStreaming<32>(
        void*       _Destination,
        _Type_      _Value,
        sizetype    _Length) noexcept
    {
        __m128i* _Pointer = static_cast<__m128i*>(_Destination);

        while (_Length--) {
            __SIMD_STL_REPEAT_N(2, _mm_stream_si128(_Pointer++, _Value));
        }

        _mm_sfence();

        return _Destination;
    }

    template <>
    static void* _MemsetAlignedStreaming<64>(
        void*       _Destination,
        _Type_      _Value,
        sizetype    _Length) noexcept
    {
        __m128i* _Pointer = static_cast<__m128i*>(_Destination);

        while (_Length--) {
            __SIMD_STL_REPEAT_N(4, _mm_stream_si128(_Pointer++, _Value));
        }

        _mm_sfence();

        return _Destination;
    }

    template <>
    static void* _MemsetAlignedStreaming<128>(
        void*       _Destination,
        _Type_      _Value,
        sizetype    _Length) noexcept
    {
        __m128i* _Pointer = static_cast<__m128i*>(_Destination);

        while (_Length--) {
            __SIMD_STL_REPEAT_N(8, _mm_stream_si128(_Pointer++, _Value));
        }

        _mm_sfence();

        return _Destination;
    }

    template <>
    static void* _MemsetAlignedStreaming<256>(
        void* _Destination,
        _Type_      _Value,
        sizetype    _Length) noexcept
    {
        __m128i* _Pointer = static_cast<__m128i*>(_Destination);

        while (_Length--) {
            __SIMD_STL_REPEAT_N(16, _mm_stream_si128(_Pointer++, _Value));
        }

        _mm_sfence();

        return _Destination;
    }
};

#pragma endregion

#pragma region Vectorized Memcpy Avx

template <typename _Type_>
class _VectorizedMemsetImplementation<arch::CpuFeature::AVX, _Type_, false>
{
public:
    template <sizetype _Step_>
    static void* _Memset(
        void*       _Destination,
        _Type_      _Value,
        sizetype    _Length) noexcept
    {
        AssertUnreachable();
        return nullptr;
    }
    
    template <>
    static void* _Memset<16>(
        void*       _Destination,
        _Type_      _Value,
        sizetype    _Length) noexcept
    {
        __m128i* _Pointer = static_cast<__m128i*>(_Destination);

        while (_Length--) {
            _mm_storeu_si128(_Pointer++, _Value);
        }

        return _Destination;
    }

    template <>
    static void* _Memset<32>(
        void*       _Destination,
        _Type_      _Value,
        sizetype    _Length) noexcept
    {
        __m256i* _Pointer = static_cast<__m256i*>(_Destination);

        while (_Length--) {
            _mm256_storeu_si256(_Pointer++, _Value);
        }

        return _Destination;
    }

    template <>
    static void* _Memset<64>(
        void*       _Destination,
        _Type_      _Value,
        sizetype    _Length) noexcept
    {
        __m256i* _Pointer = static_cast<__m256i*>(_Destination);

        while (_Length--) {
            __SIMD_STL_REPEAT_N(2, _mm256_storeu_si256(_Pointer++, _Value));
        }

        return _Destination;
    }

    template <>
    static void* _Memset<128>(
        void*       _Destination,
        _Type_      _Value,
        sizetype    _Length) noexcept
    {
        __m256i* _Pointer = static_cast<__m256i*>(_Destination);

        while (_Length--) {
            __SIMD_STL_REPEAT_N(4, _mm256_storeu_si256(_Pointer++, _Value));
        }

        return _Destination;
    }

    template <>
    static void* _Memset<256>(
        void*       _Destination,
        _Type_      _Value,
        sizetype    _Length) noexcept
    {
        __m256i* _Pointer = static_cast<__m256i*>(_Destination);

        while (_Length--) {
            __SIMD_STL_REPEAT_N(8, _mm256_storeu_si256(_Pointer++, _Value));
        }

        return _Destination;
    }

    template <>
    static void* _Memset<512>(
        void*       _Destination,
        _Type_      _Value,
        sizetype    _Length) noexcept
    {
        __m256i* _Pointer = static_cast<__m256i*>(_Destination);

        while (_Length--) {
            __SIMD_STL_REPEAT_N(16, _mm256_storeu_si256(_Pointer++, _Value));
        }

        return _Destination;
    }
};


template <typename _Type_>
class _VectorizedMemsetImplementation<arch::CpuFeature::AVX, _Type_, true>
{
public:
    template <sizetype _Step_>
    static void* _Memset(
        void*       _Destination,
        _Type_      _Value,
        sizetype    _Length) noexcept
    {
        AssertUnreachable();
        return nullptr;
    }

    template <>
    static void* _Memset<16>(
        void* _Destination,
        _Type_      _Value,
        sizetype    _Length) noexcept
    {
        __m128i* _Pointer = static_cast<__m128i*>(_Destination);

        while (_Length--) {
            _mm_store_si128(_Pointer++, _Value);
        }

        return _Destination;
    }

    template <>
    static void* _Memset<32>(
        void*       _Destination,
        _Type_      _Value,
        sizetype    _Length) noexcept
    {
        __m256i* _Pointer = static_cast<__m256i*>(_Destination);

        while (_Length--) {
            _mm256_store_si256(_Pointer++, _Value);
        }

        return _Destination;
    }

    template <>
    static void* _Memset<64>(
        void*       _Destination,
        _Type_      _Value,
        sizetype    _Length) noexcept
    {
        __m256i* _Pointer = static_cast<__m256i*>(_Destination);

        while (_Length--) {
            __SIMD_STL_REPEAT_N(2, _mm256_store_si256(_Pointer++, _Value));
        }

        return _Destination;
    }

    template <>
    static void* _Memset<128>(
        void*       _Destination,
        _Type_      _Value,
        sizetype    _Length) noexcept
    {
        __m256i* _Pointer = static_cast<__m256i*>(_Destination);

        while (_Length--) {
            __SIMD_STL_REPEAT_N(4, _mm256_store_si256(_Pointer++, _Value));
        }

        return _Destination;
    }

    template <>
    static void* _Memset<256>(
        void*       _Destination,
        _Type_      _Value,
        sizetype    _Length) noexcept
    {
        __m256i* _Pointer = static_cast<__m256i*>(_Destination);

        while (_Length--) {
            __SIMD_STL_REPEAT_N(8, _mm256_store_si256(_Pointer++, _Value));
        }

        return _Destination;
    }

    template <>
    static void* _Memset<512>(
        void*       _Destination,
        _Type_      _Value,
        sizetype    _Length) noexcept
    {
        __m256i* _Pointer = static_cast<__m256i*>(_Destination);

        while (_Length--) {
            __SIMD_STL_REPEAT_N(16, _mm256_store_si256(_Pointer++, _Value));
        }

        return _Destination;
    }

    template <sizetype _Step_>
    static void* _MemsetAlignedStreaming(
        void*       _Destination,
        _Type_      _Value,
        sizetype    _Length) noexcept
    {
        AssertUnreachable();
        return nullptr;
    }
   
    template <>
    static void* _MemsetAlignedStreaming<32>(
        void*       _Destination,
        _Type_      _Value,
        sizetype    _Length) noexcept
    {
        __m256i* _Pointer = static_cast<__m256i*>(_Destination);

        while (_Length--) {
            _mm256_stream_si256(_Pointer++, _Value);
        }

        _mm_sfence();

        return _Destination;
    }

    template <>
    static void* _MemsetAlignedStreaming<64>(
        void*       _Destination,
        _Type_      _Value,
        sizetype    _Length) noexcept
    {
        __m256i* _Pointer = static_cast<__m256i*>(_Destination);

        while (_Length--) {
            __SIMD_STL_REPEAT_N(2, _mm256_stream_si256(_Pointer++, _Value));
        }

        _mm_sfence();

        return _Destination;
    }

    template <>
    static void* _MemsetAlignedStreaming<128>(
        void*       _Destination,
        _Type_      _Value,
        sizetype    _Length) noexcept
    {
        __m256i* _Pointer = static_cast<__m256i*>(_Destination);

        while (_Length--) {
            __SIMD_STL_REPEAT_N(4, _mm256_stream_si256(_Pointer++, _Value));
        }

        _mm_sfence();

        return _Destination;
    }

    template <>
    static void* _MemsetAlignedStreaming<256>(
        void*       _Destination,
        _Type_      _Value,
        sizetype    _Length) noexcept
    {
        __m256i* _Pointer = static_cast<__m256i*>(_Destination);

        while (_Length--) {
            __SIMD_STL_REPEAT_N(8, _mm256_stream_si256(_Pointer++, _Value));
        }

        _mm_sfence();

        return _Destination;
    }

    template <>
    static void* _MemsetAlignedStreaming<512>(
        void* _Destination,
        _Type_      _Value,
        sizetype    _Length) noexcept
    {
        __m256i* _Pointer = static_cast<__m256i*>(_Destination);

        while (_Length--) {
            __SIMD_STL_REPEAT_N(16, _mm256_stream_si256(_Pointer++, _Value));
        }

        _mm_sfence();

        return _Destination;
    }
};

#pragma endregion 

#pragma region Vectorized Memcpy Avx512

template <typename _Type_>
class _VectorizedMemsetImplementation<arch::CpuFeature::AVX512F, _Type_, false>
{
public:
    template <sizetype _Step_>
    simd_stl_always_inline static void* _Memset(
        void*       _Destination,
        _Type_      _Value,
        sizetype    _Length) noexcept
    {
        AssertUnreachable();
        return nullptr;
    }

    template <>
    simd_stl_always_inline static void* _Memset<16>(
        void* _Destination,
        _Type_      _Value,
        sizetype    _Length) noexcept
    {
        __m128i* _Pointer = static_cast<__m128i*>(_Destination);

        while (_Length--) {
            _mm_storeu_si128(_Pointer++, _Value);
        }

        return _Destination;
    }

    template <>
    simd_stl_always_inline static void* _Memset<32>(
        void*       _Destination,
        _Type_      _Value,
        sizetype    _Length) noexcept
    {
        __m256i* _Pointer = static_cast<__m256i*>(_Destination);

        while (_Length--) {
            _mm256_storeu_si256(_Pointer++, _Value);
        }

        return _Destination;
    }

    template <>
    simd_stl_always_inline static void* _Memset<64>(
        void*       _Destination,
        _Type_      _Value,
        sizetype    _Length) noexcept
    {
        __m512i* _Pointer = static_cast<__m512i*>(_Destination);

        while (_Length--) {
            _mm512_storeu_si512(_Pointer++, numeric::_IntrinBitcast<__m512i>(_Value));
        }

        return _Destination;
    }

    template <>
    simd_stl_always_inline static void* _Memset<128>(
        void*       _Destination,
        _Type_      _Value,
        sizetype    _Length) noexcept
    {
        __m512i* _Pointer = static_cast<__m512i*>(_Destination);

        while (_Length--) {
            __SIMD_STL_REPEAT_N(2, _mm512_storeu_si512(_Pointer++, numeric::_IntrinBitcast<__m512i>(_Value)));
        }

        return _Destination;
    }

    template <>
    simd_stl_always_inline static void* _Memset<256>(
        void*       _Destination,
        _Type_      _Value,
        sizetype    _Length) noexcept
    {
        __m512i* _Pointer = static_cast<__m512i*>(_Destination);

        while (_Length--) {
            __SIMD_STL_REPEAT_N(4, _mm512_storeu_si512(_Pointer++, numeric::_IntrinBitcast<__m512i>(_Value)));
        }

        return _Destination;
    }

    template <>
    simd_stl_always_inline static void* _Memset<512>(
        void*       _Destination,
        _Type_      _Value,
        sizetype    _Length) noexcept
    {
        __m512i* _Pointer = static_cast<__m512i*>(_Destination);

        while (_Length--) {
            __SIMD_STL_REPEAT_N(8, _mm512_storeu_si512(_Pointer++, numeric::_IntrinBitcast<__m512i>(_Value)));
        }

        return _Destination;
    }

    template <>
    simd_stl_always_inline static void* _Memset<1024>(
        void*       _Destination,
        _Type_      _Value,
        sizetype    _Length) noexcept
    {
        __m512i* _Pointer = static_cast<__m512i*>(_Destination);

        while (_Length--) {
            __SIMD_STL_REPEAT_N(16, _mm512_storeu_si512(_Pointer++, numeric::_IntrinBitcast<__m512i>(_Value)));
        }

        return _Destination;
    }

    template <>
    simd_stl_always_inline static void* _Memset<2048>(
        void*       _Destination,
        _Type_      _Value,
        sizetype    _Length) noexcept
    {
        __m512i* _Pointer = static_cast<__m512i*>(_Destination);

        while (_Length--) {
            __SIMD_STL_REPEAT_N(32, _mm512_storeu_si512(_Pointer++, numeric::_IntrinBitcast<__m512i>(_Value)));
        }

        return _Destination;
    }

    template <>
    simd_stl_always_inline static void* _Memset<4096>(
        void* _Destination,
        _Type_      _Value,
        sizetype    _Length) noexcept
    {
        __m512i* _Pointer = static_cast<__m512i*>(_Destination);

        while (_Length--) {
            __SIMD_STL_REPEAT_N(64, _mm512_storeu_si512(_Pointer++, numeric::_IntrinBitcast<__m512i>(_Value)));
        }

        return _Destination;
    }
};

template <typename _Type_>
class _VectorizedMemsetImplementation<arch::CpuFeature::AVX512F, _Type_, true>
{
public:
    template <sizetype _Step_>
    simd_stl_always_inline static void* _Memset(
        void*       _Destination,
        _Type_      _Value,
        sizetype    _Length) noexcept
    {
        AssertUnreachable();
        return nullptr;
    }

    template <>
    simd_stl_always_inline static void* _Memset<16>(
        void* _Destination,
        _Type_      _Value,
        sizetype    _Length) noexcept
    {
        __m128i* _Pointer = static_cast<__m128i*>(_Destination);

        while (_Length--) {
            _mm_store_si128(_Pointer++, _Value);
        }

        return _Destination;
    }

    template <>
    simd_stl_always_inline static void* _Memset<32>(
        void*       _Destination,
        _Type_      _Value,
        sizetype    _Length) noexcept
    {
        __m256i* _Pointer = static_cast<__m256i*>(_Destination);

        while (_Length--) {
            _mm256_store_si256(_Pointer++, _Value);
        }

        return _Destination;
    }

    template <>
    simd_stl_always_inline static void* _Memset<64>(
        void*       _Destination,
        _Type_      _Value,
        sizetype    _Length) noexcept
    {
        __m512i* _Pointer = static_cast<__m512i*>(_Destination);

        while (_Length--) {
            _mm512_store_si512(_Pointer++, numeric::_IntrinBitcast<__m512i>(_Value));
        }

        return _Destination;
    }

    template <>
    simd_stl_always_inline static void* _Memset<128>(
        void*       _Destination,
        _Type_      _Value,
        sizetype    _Length) noexcept
    {
        __m512i* _Pointer = static_cast<__m512i*>(_Destination);

        while (_Length--) {
            __SIMD_STL_REPEAT_N(2, _mm512_store_si512(_Pointer++, numeric::_IntrinBitcast<__m512i>(_Value)));
        }

        return _Destination;
    }

    template <>
    simd_stl_always_inline static void* _Memset<256>(
        void*       _Destination,
        _Type_      _Value,
        sizetype    _Length) noexcept
    {
        __m512i* _Pointer = static_cast<__m512i*>(_Destination);

        while (_Length--) {
            __SIMD_STL_REPEAT_N(4, _mm512_store_si512(_Pointer++, numeric::_IntrinBitcast<__m512i>(_Value)));
        }

        return _Destination;
    }

    template <>
    simd_stl_always_inline static void* _Memset<512>(
        void*       _Destination,
        _Type_      _Value,
        sizetype    _Length) noexcept
    {
        __m512i* _Pointer = static_cast<__m512i*>(_Destination);

        while (_Length--) {
            __SIMD_STL_REPEAT_N(8, _mm512_store_si512(_Pointer++, numeric::_IntrinBitcast<__m512i>(_Value)));
        }

        return _Destination;
    }

    template <>
    simd_stl_always_inline static void* _Memset<1024>(
        void*       _Destination,
        _Type_      _Value,
        sizetype    _Length) noexcept
    {
        __m512i* _Pointer = static_cast<__m512i*>(_Destination);

        while (_Length--) {
            __SIMD_STL_REPEAT_N(16, _mm512_store_si512(_Pointer++, _Value));
        }

        return _Destination;
    }

    template <>
    simd_stl_always_inline static void* _Memset<2048>(
        void*       _Destination,
        _Type_      _Value,
        sizetype    _Length) noexcept
    {
        __m512i* _Pointer = static_cast<__m512i*>(_Destination);

        while (_Length--) {
            __SIMD_STL_REPEAT_N(32, _mm512_store_si512(_Pointer++, _Value));
        }

        return _Destination;
    }

    template <>
    simd_stl_always_inline static void* _Memset<4096>(
        void* _Destination,
        _Type_      _Value,
        sizetype    _Length) noexcept
    {
        __m512i* _Pointer = static_cast<__m512i*>(_Destination);

        while (_Length--) {
            __SIMD_STL_REPEAT_N(64, _mm512_store_si512(_Pointer++, _Value));
        }

        return _Destination;
    }

    template <sizetype _Step_>
    simd_stl_always_inline static void* _MemsetAlignedStreaming(
        void*       _Destination,
        _Type_      _Value,
        sizetype    _Length) noexcept
    {
        AssertUnreachable();
        return nullptr;
    }

    template <>
    simd_stl_always_inline static void* _MemsetAlignedStreaming<16>(
        void* _Destination,
        _Type_      _Value,
        sizetype    _Length) noexcept
    {
        __m128i* _Pointer = static_cast<__m128i*>(_Destination);

        while (_Length--) {
            _mm_stream_si128(_Pointer++, numeric::_IntrinBitcast<__m128i>(_Value));
        }

        _mm_sfence();

        return _Destination;
    }
    
    template <>
    simd_stl_always_inline static void* _MemsetAlignedStreaming<32>(
        void*       _Destination,
        _Type_      _Value,
        sizetype    _Length) noexcept
    {
        __m256i* _Pointer = static_cast<__m256i*>(_Destination);

        while (_Length--) {
            _mm256_stream_si256(_Pointer++, numeric::_IntrinBitcast<__m256i>(_Value));
        }

        _mm_sfence();

        return _Destination;
    }

    template <>
    simd_stl_always_inline static void* _MemsetAlignedStreaming<64>(
        void*       _Destination,
        _Type_      _Value,
        sizetype    _Length) noexcept
    {
        __m512i* _Pointer = static_cast<__m512i*>(_Destination);

        while (_Length--) {
            _mm512_stream_si512(_Pointer++, numeric::_IntrinBitcast<__m512i>(_Value));
        }

        _mm_sfence();

        return _Destination;
    }

    template <>
    simd_stl_always_inline static void* _MemsetAlignedStreaming<128>(
        void*       _Destination,
        _Type_      _Value,
        sizetype    _Length) noexcept
    {
        __m512i* _Pointer = static_cast<__m512i*>(_Destination);

        while (_Length--) {
            __SIMD_STL_REPEAT_N(2, _mm512_stream_si512(_Pointer++, numeric::_IntrinBitcast<__m512i>(_Value)));
        }

        _mm_sfence();

        return _Destination;
    }

    template <>
    simd_stl_always_inline static void* _MemsetAlignedStreaming<256>(
        void*       _Destination,
        _Type_      _Value,
        sizetype    _Length) noexcept
    {
        __m512i* _Pointer = static_cast<__m512i*>(_Destination);

        while (_Length--) {
            __SIMD_STL_REPEAT_N(4, _mm512_stream_si512(_Pointer++, numeric::_IntrinBitcast<__m512i>(_Value)));
        }

        _mm_sfence();

        return _Destination;
    }

    template <>
    simd_stl_always_inline static void* _MemsetAlignedStreaming<512>(
        void*       _Destination,
        _Type_      _Value,
        sizetype    _Length) noexcept
    {
        __m512i* _Pointer = static_cast<__m512i*>(_Destination);

        while (_Length--) {
            __SIMD_STL_REPEAT_N(8, _mm512_stream_si512(_Pointer++, numeric::_IntrinBitcast<__m512i>(_Value)));
        }

        _mm_sfence();

        return _Destination;
    }

    template <>
    simd_stl_always_inline static void* _MemsetAlignedStreaming<1024>(
        void*       _Destination,
        _Type_      _Value,
        sizetype    _Length) noexcept
    {
        __m512i* _Pointer = static_cast<__m512i*>(_Destination);

        while (_Length--) {
            __SIMD_STL_REPEAT_N(16, _mm512_stream_si512(_Pointer++, numeric::_IntrinBitcast<__m512i>(_Value)));
        }

        _mm_sfence();

        return _Destination;
    }

    template <>
    simd_stl_always_inline static void* _MemsetAlignedStreaming<2048>(
        void*       _Destination,
        _Type_      _Value,
        sizetype    _Length) noexcept
    {
        __m512i* _Pointer = static_cast<__m512i*>(_Destination);

        while (_Length--) {
            __SIMD_STL_REPEAT_N(32, _mm512_stream_si512(_Pointer++, numeric::_IntrinBitcast<__m512i>(_Value)));
        }

        _mm_sfence();

        return _Destination;
    }

    template <>
    simd_stl_always_inline static void* _MemsetAlignedStreaming<4096>(
        void* _Destination,
        _Type_      _Value,
        sizetype    _Length) noexcept
    {
        __m512i* _Pointer = static_cast<__m512i*>(_Destination);

        while (_Length--) {
            __SIMD_STL_REPEAT_N(64, _mm512_stream_si512(_Pointer++, numeric::_IntrinBitcast<__m512i>(_Value)));
        }

        _mm_sfence();

        return _Destination;
    }
};

#pragma endregion

#pragma region Zero memset

template <
    bool                _Aligned_,
    bool                _Streaming_,
    arch::CpuFeature    _SimdGeneration_>
struct _MemsetZerosVectorizedChooser;

template <
    bool _Aligned_,
    bool _Streaming_>
struct _MemsetZerosVectorizedChooser<_Aligned_, _Streaming_, arch::CpuFeature::SSE2> {
    template <typename _MemsetType_>
    using _MemsetImplementation = _VectorizedMemsetImplementation<arch::CpuFeature::SSE2, _MemsetType_, _Aligned_>;

    simd_stl_always_inline void* operator()(
        void*       _Destination, 
        sizetype    _Bytes) const noexcept 
    {
        void* _Return       = _Destination;
        sizetype _Offset    = 0;

        __m128i _Broadcasted;
        if (_Bytes > 16)
            _Broadcasted = _mm_setzero_si128();

        while (_Bytes) {
            if (_Bytes < 16)
            {
                _MemsetScalar(_Destination, '\0', _Bytes);
                _Offset = _Bytes;
                AdvanceBytes(_Destination, _Offset);
                _Bytes = 0;
            }
            else if (_Bytes < 32)
            {
                if constexpr (_Streaming_ && _Aligned_)
                    _MemsetImplementation<__m128i>::_MemsetAlignedStreaming<16>(_Destination, _Broadcasted, _Bytes >> 4);
                else
                    _MemsetImplementation<__m128i>::_Memset<16>(_Destination, _Broadcasted, _Bytes >> 4);

                _Offset = _Bytes & -16;
                AdvanceBytes(_Destination, _Offset);
                _Bytes &= 15;
            }
            else if (_Bytes < 64)
            {
                if constexpr (_Streaming_ && _Aligned_)
                    _MemsetImplementation<__m128i>::_MemsetAlignedStreaming<32>(_Destination, _Broadcasted, _Bytes >> 5);
                else
                    _MemsetImplementation<__m128i>::_Memset<32>(_Destination, _Broadcasted, _Bytes >> 5);

                _Offset = _Bytes & -32;
                AdvanceBytes(_Destination, _Offset);
                _Bytes &= 31;
            }
            else if (_Bytes < 128)
            {
                if constexpr (_Streaming_ && _Aligned_)
                    _MemsetImplementation<__m128i>::_MemsetAlignedStreaming<64>(_Destination, _Broadcasted, _Bytes >> 6);
                else
                    _MemsetImplementation<__m128i>::_Memset<64>(_Destination, _Broadcasted, _Bytes >> 6);

                _Offset = _Bytes & -64;
                AdvanceBytes(_Destination, _Offset);
                _Bytes &= 63;
            }
            else if (_Bytes < 256)
            {
                if constexpr (_Streaming_ && _Aligned_)
                    _MemsetImplementation<__m128i>::_MemsetAlignedStreaming<128>(_Destination, _Broadcasted, _Bytes >> 7);
                else
                    _MemsetImplementation<__m128i>::_Memset<128>(_Destination, _Broadcasted, _Bytes >> 7);

                _Offset = _Bytes & -128;
                AdvanceBytes(_Destination, _Offset);
                _Bytes &= 127;
            }
            else
            {
                if constexpr (_Streaming_ && _Aligned_)
                    _MemsetImplementation<__m128i>::_MemsetAlignedStreaming<256>(_Destination, _Broadcasted, _Bytes >> 8);
                else
                    _MemsetImplementation<__m128i>::_Memset<256>(_Destination, _Broadcasted, _Bytes >> 8);

                _Offset = _Bytes & -256;
                AdvanceBytes(_Destination, _Offset);
                _Bytes &= 255;
            }
        }

        return _Return;
    }
};

template <
    bool _Aligned_,
    bool _Streaming_>
struct _MemsetZerosVectorizedChooser<_Aligned_, _Streaming_, arch::CpuFeature::AVX> {
    template <typename _MemsetType_>
    using _MemsetImplementation = _VectorizedMemsetImplementation<arch::CpuFeature::AVX, _MemsetType_, _Aligned_>;

    simd_stl_always_inline void* operator()(
        void*       _Destination, 
        sizetype    _Bytes) const noexcept 
    {
        void* _Return       = _Destination;
        sizetype _Offset    = 0;

        __m256i _Broadcasted;
        if (_Bytes > 16)
            _Broadcasted = _mm256_setzero_si256();

        while (_Bytes) {
            if (_Bytes < 16)
            {
                _MemsetScalar(_Destination, '\0', _Bytes);
                _Offset = _Bytes;
                AdvanceBytes(_Destination, _Offset);
                _Bytes = 0;
            }
            else if (_Bytes < 32)
            {
                if constexpr (_Streaming_ && _Aligned_)
                    _MemsetImplementation<__m128i>::_MemsetAlignedStreaming<16>(_Destination, numeric::_IntrinBitcast<__m128i>(_Broadcasted), _Bytes >> 4);
                else
                    _MemsetImplementation<__m128i>::_Memset<16>(_Destination, numeric::_IntrinBitcast<__m128i>(_Broadcasted), _Bytes >> 4);

                _Offset = _Bytes & -16;
                AdvanceBytes(_Destination, _Offset);
                _Bytes &= 15;
            }
            else if (_Bytes < 64)
            {
                if constexpr (_Streaming_ && _Aligned_)
                    _MemsetImplementation<__m256i>::_MemsetAlignedStreaming<32>(_Destination, _Broadcasted, _Bytes >> 5);
                else
                    _MemsetImplementation<__m256i>::_Memset<32>(_Destination, _Broadcasted, _Bytes >> 5);

                _Offset = _Bytes & -32;
                AdvanceBytes(_Destination, _Offset);
                _Bytes &= 31;
            }
            else if (_Bytes < 128)
            {
                if constexpr (_Streaming_ && _Aligned_)
                    _MemsetImplementation<__m256i>::_MemsetAlignedStreaming<64>(_Destination, _Broadcasted, _Bytes >> 6);
                else
                    _MemsetImplementation<__m256i>::_Memset<64>(_Destination, _Broadcasted, _Bytes >> 6);

                _Offset = _Bytes & -64;
                AdvanceBytes(_Destination, _Offset);
                _Bytes &= 63;
            }
            else if (_Bytes < 256)
            {
                if constexpr (_Streaming_ && _Aligned_)
                    _MemsetImplementation<__m256i>::_MemsetAlignedStreaming<128>(_Destination, _Broadcasted, _Bytes >> 7);
                else
                    _MemsetImplementation<__m256i>::_Memset<128>(_Destination, _Broadcasted, _Bytes >> 7);

                _Offset = _Bytes & -128;
                AdvanceBytes(_Destination, _Offset);
                _Bytes &= 127;
            }
            else if (_Bytes < 512)
            {
                if constexpr (_Streaming_ && _Aligned_)
                    _MemsetImplementation<__m256i>::_MemsetAlignedStreaming<256>(_Destination, _Broadcasted, _Bytes >> 8);
                else
                    _MemsetImplementation<__m256i>::_Memset<256>(_Destination, _Broadcasted, _Bytes >> 8);

                _Offset = _Bytes & -256;
                AdvanceBytes(_Destination, _Offset);
                _Bytes &= 255;
            }
            else
            {
                if constexpr (_Streaming_ && _Aligned_)
                    _MemsetImplementation<__m256i>::_MemsetAlignedStreaming<512>(_Destination, _Broadcasted, _Bytes >> 9);
                else
                    _MemsetImplementation<__m256i>::_Memset<512>(_Destination, _Broadcasted, _Bytes >> 9);

                _Offset = _Bytes & -512;
                AdvanceBytes(_Destination, _Offset);
                _Bytes &= 511;
            }
        }

        return _Return;
    }
};

template <
    bool _Aligned_,
    bool _Streaming_>
struct _MemsetZerosVectorizedChooser<_Aligned_, _Streaming_, arch::CpuFeature::AVX512F> {
    template <typename _MemsetType_>
    using _MemsetImplementation = _VectorizedMemsetImplementation<arch::CpuFeature::AVX512F, _MemsetType_, _Aligned_>;

    simd_stl_always_inline void* operator()(
        void* _Destination,
        sizetype    _Bytes) const noexcept
    {
        void* _Return = _Destination;
        sizetype _Offset = 0;

        __m512i _Broadcasted;
        if (_Bytes > 16)
            _Broadcasted = _mm512_setzero_si512();

        while (_Bytes) {
            if (_Bytes < 16)
            {
                _MemsetScalar(_Destination, '\0', _Bytes);
                _Offset = _Bytes;
                AdvanceBytes(_Destination, _Offset);
                _Bytes = 0;
            }
            else if (_Bytes < 32)
            {
                if constexpr (_Streaming_ && _Aligned_)
                    _MemsetImplementation<__m128i>::_MemsetAlignedStreaming<16>(_Destination, numeric::_IntrinBitcast<__m128i>(_Broadcasted), _Bytes >> 4);
                else
                    _MemsetImplementation<__m128i>::_Memset<16>(_Destination, numeric::_IntrinBitcast<__m128i>(_Broadcasted), _Bytes >> 4);

                _Offset = _Bytes & -16;
                AdvanceBytes(_Destination, _Offset);
                _Bytes &= 15;
            }
            else if (_Bytes < 64)
            {
                if constexpr (_Streaming_ && _Aligned_)
                    _MemsetImplementation<__m256i>::_MemsetAlignedStreaming<32>(_Destination, numeric::_IntrinBitcast<__m256i>(_Broadcasted), _Bytes >> 5);
                else
                    _MemsetImplementation<__m256i>::_Memset<32>(_Destination, numeric::_IntrinBitcast<__m256i>(_Broadcasted), _Bytes >> 5);

                _Offset = _Bytes & -32;
                AdvanceBytes(_Destination, _Offset);
                _Bytes &= 31;
            }
            else if (_Bytes < 128)
            {
                if constexpr (_Streaming_ && _Aligned_)
                    _MemsetImplementation<__m512i>::_MemsetAlignedStreaming<64>(_Destination, _Broadcasted, _Bytes >> 6);
                else
                    _MemsetImplementation<__m512i>::_Memset<64>(_Destination, _Broadcasted, _Bytes >> 6);

                _Offset = _Bytes & -64;
                AdvanceBytes(_Destination, _Offset);
                _Bytes &= 63;
            }
            else if (_Bytes < 256)
            {
                if constexpr (_Streaming_ && _Aligned_)
                    _MemsetImplementation<__m512i>::_MemsetAlignedStreaming<128>(_Destination, _Broadcasted, _Bytes >> 7);
                else
                    _MemsetImplementation<__m512i>::_Memset<128>(_Destination, _Broadcasted, _Bytes >> 7);

                _Offset = _Bytes & -128;
                AdvanceBytes(_Destination, _Offset);
                _Bytes &= 127;
            }
            else if (_Bytes < 512)
            {
                if constexpr (_Streaming_ && _Aligned_)
                    _MemsetImplementation<__m512i>::_MemsetAlignedStreaming<256>(_Destination, _Broadcasted, _Bytes >> 8);
                else
                    _MemsetImplementation<__m512i>::_Memset<256>(_Destination, _Broadcasted, _Bytes >> 8);

                _Offset = _Bytes & -256;
                AdvanceBytes(_Destination, _Offset);
                _Bytes &= 255;
            }
            else if (_Bytes < 1024)
            {
                if constexpr (_Streaming_ && _Aligned_)
                    _MemsetImplementation<__m512i>::_MemsetAlignedStreaming<512>(_Destination, _Broadcasted, _Bytes >> 9);
                else
                    _MemsetImplementation<__m512i>::_Memset<512>(_Destination, _Broadcasted, _Bytes >> 9);

                _Offset = _Bytes & -512;
                AdvanceBytes(_Destination, _Offset);
                _Bytes &= 511;
            }
            else if (_Bytes < 2048)
            {
                if constexpr (_Streaming_ && _Aligned_)
                    _MemsetImplementation<__m512i>::_MemsetAlignedStreaming<1024>(_Destination, _Broadcasted, _Bytes >> 10);
                else
                    _MemsetImplementation<__m512i>::_Memset<1024>(_Destination, _Broadcasted, _Bytes >> 10);

                _Offset = _Bytes & -1024;
                AdvanceBytes(_Destination, _Offset);
                _Bytes &= 1023;
            }
            else if (_Bytes < 4096)
            {
                if constexpr (_Streaming_ && _Aligned_)
                    _MemsetImplementation<__m512i>::_MemsetAlignedStreaming<2048>(_Destination, _Broadcasted, _Bytes >> 11);
                else
                    _MemsetImplementation<__m512i>::_Memset<2048>(_Destination, _Broadcasted, _Bytes >> 11);

                _Offset = _Bytes & -2048;
                AdvanceBytes(_Destination, _Offset);
                _Bytes &= 2047;
            }
            else
            {
                if constexpr (_Streaming_ && _Aligned_)
                    _MemsetImplementation<__m512i>::_MemsetAlignedStreaming<4096>(_Destination, _Broadcasted, _Bytes >> 12);
                else
                    _MemsetImplementation<__m512i>::_Memset<4096>(_Destination, _Broadcasted, _Bytes >> 12);

                _Offset = _Bytes & -4096;
                AdvanceBytes(_Destination, _Offset);
                _Bytes &= 4095;
            }
        }

        return _Return;
    }
};

#pragma endregion 

#pragma region Memset

template <
    bool                _Aligned_,
    bool                _Streaming_,
    arch::CpuFeature    _SimdGeneration_>
struct _MemsetVectorizedChooser;

template <
    bool _Aligned_,
    bool _Streaming_>
struct _MemsetVectorizedChooser<_Aligned_, _Streaming_, arch::CpuFeature::SSE2> {
    template <typename _MemsetType_>
    using _MemsetImplementation = _VectorizedMemsetImplementation<arch::CpuFeature::SSE2, _MemsetType_, _Aligned_>;

    template <typename _Type_>
    simd_stl_always_inline void* operator()(
        void*       _Destination,
        _Type_      _Value,
        sizetype    _Bytes) const noexcept
    {
        void* _Return = _Destination;
        sizetype _Offset = 0;

        __m128i _Broadcasted;
        if (_Bytes > 16)
            _Broadcasted = numeric::_SimdBroadcast<arch::CpuFeature::SSE2, numeric::xmm128, __m128i>(_Value);

        while (_Bytes) {
            if (_Bytes < 16)
            {
                _MemsetScalar(_Destination, _Value, _Bytes / sizeof(_Type_));
                _Offset = _Bytes;
                AdvanceBytes(_Destination, _Offset);
                _Bytes = 0;
            }
            else if (_Bytes < 32)
            {
                if constexpr (_Streaming_ && _Aligned_)
                    _MemsetImplementation<__m128i>::_MemsetAlignedStreaming<16>(_Destination, _Broadcasted, _Bytes >> 4);
                else
                    _MemsetImplementation<__m128i>::_Memset<16>(_Destination, _Broadcasted, _Bytes >> 4);

                _Offset = _Bytes & -16;
                AdvanceBytes(_Destination, _Offset);
                _Bytes &= 15;
            }
            else if (_Bytes < 64)
            {
                if constexpr (_Streaming_ && _Aligned_)
                    _MemsetImplementation<__m128i>::_MemsetAlignedStreaming<32>(_Destination, _Broadcasted, _Bytes >> 5);
                else
                    _MemsetImplementation<__m128i>::_Memset<32>(_Destination, _Broadcasted, _Bytes >> 5);

                _Offset = _Bytes & -32;
                AdvanceBytes(_Destination, _Offset);
                _Bytes &= 31;
            }
            else if (_Bytes < 128)
            {
                if constexpr (_Streaming_ && _Aligned_)
                    _MemsetImplementation<__m128i>::_MemsetAlignedStreaming<64>(_Destination, _Broadcasted, _Bytes >> 6);
                else
                    _MemsetImplementation<__m128i>::_Memset<64>(_Destination, _Broadcasted, _Bytes >> 6);

                _Offset = _Bytes & -64;
                AdvanceBytes(_Destination, _Offset);
                _Bytes &= 63;
            }
            else if (_Bytes < 256)
            {
                if constexpr (_Streaming_ && _Aligned_)
                    _MemsetImplementation<__m128i>::_MemsetAlignedStreaming<128>(_Destination, _Broadcasted, _Bytes >> 7);
                else
                    _MemsetImplementation<__m128i>::_Memset<128>(_Destination, _Broadcasted, _Bytes >> 7);

                _Offset = _Bytes & -128;
                AdvanceBytes(_Destination, _Offset);
                _Bytes &= 127;
            }
            else
            {
                if constexpr (_Streaming_ && _Aligned_)
                    _MemsetImplementation<__m128i>::_MemsetAlignedStreaming<256>(_Destination, _Broadcasted, _Bytes >> 8);
                else
                    _MemsetImplementation<__m128i>::_Memset<256>(_Destination, _Broadcasted, _Bytes >> 8);

                _Offset = _Bytes & -256;
                AdvanceBytes(_Destination, _Offset);
                _Bytes &= 255;
            }
        }

        return _Return;
    }
};

template <
    bool _Aligned_,
    bool _Streaming_>
struct _MemsetVectorizedChooser<_Aligned_, _Streaming_, arch::CpuFeature::AVX> {
    template <typename _MemsetType_>
    using _MemsetImplementation = _VectorizedMemsetImplementation<arch::CpuFeature::AVX, _MemsetType_, _Aligned_>;

    template <typename _Type_>
    simd_stl_always_inline void* operator()(
        void*       _Destination,
        _Type_      _Value,
        sizetype    _Bytes) const noexcept
    {
        void* _Return = _Destination;
        sizetype _Offset = 0;

        __m256i _Broadcasted;
        if (_Bytes > 16)
            _Broadcasted = numeric::_SimdBroadcast<arch::CpuFeature::AVX, numeric::ymm256, __m256i>(_Value);

        while (_Bytes) {
            if (_Bytes < 16)
            {
                _MemsetScalar(_Destination, _Value, _Bytes / sizeof(_Type_));
                _Offset = _Bytes;
                AdvanceBytes(_Destination, _Offset);
                _Bytes = 0;
            }
            else if (_Bytes < 32)
            {
                if constexpr (_Streaming_ && _Aligned_)
                    _MemsetImplementation<__m128i>::_MemsetAlignedStreaming<16>(_Destination, numeric::_IntrinBitcast<__m128i>(_Broadcasted), _Bytes >> 4);
                else
                    _MemsetImplementation<__m128i>::_Memset<16>(_Destination, numeric::_IntrinBitcast<__m128i>(_Broadcasted), _Bytes >> 4);

                _Offset = _Bytes & -16;
                AdvanceBytes(_Destination, _Offset);
                _Bytes &= 15;
            }
            else if (_Bytes < 64)
            {
                if constexpr (_Streaming_ && _Aligned_)
                    _MemsetImplementation<__m256i>::_MemsetAlignedStreaming<32>(_Destination, _Broadcasted, _Bytes >> 5);
                else
                    _MemsetImplementation<__m256i>::_Memset<32>(_Destination, _Broadcasted, _Bytes >> 5);

                _Offset = _Bytes & -32;
                AdvanceBytes(_Destination, _Offset);
                _Bytes &= 31;
            }
            else if (_Bytes < 128)
            {
                if constexpr (_Streaming_ && _Aligned_)
                    _MemsetImplementation<__m256i>::_MemsetAlignedStreaming<64>(_Destination, _Broadcasted, _Bytes >> 6);
                else
                    _MemsetImplementation<__m256i>::_Memset<64>(_Destination, _Broadcasted, _Bytes >> 6);

                _Offset = _Bytes & -64;
                AdvanceBytes(_Destination, _Offset);
                _Bytes &= 63;
            }
            else if (_Bytes < 256)
            {
                if constexpr (_Streaming_ && _Aligned_)
                    _MemsetImplementation<__m256i>::_MemsetAlignedStreaming<128>(_Destination, _Broadcasted, _Bytes >> 7);
                else
                    _MemsetImplementation<__m256i>::_Memset<128>(_Destination, _Broadcasted, _Bytes >> 7);

                _Offset = _Bytes & -128;
                AdvanceBytes(_Destination, _Offset);
                _Bytes &= 127;
            }
            else if (_Bytes < 512)
            {
                if constexpr (_Streaming_ && _Aligned_)
                    _MemsetImplementation<__m256i>::_MemsetAlignedStreaming<256>(_Destination, _Broadcasted, _Bytes >> 8);
                else
                    _MemsetImplementation<__m256i>::_Memset<256>(_Destination, _Broadcasted, _Bytes >> 8);

                _Offset = _Bytes & -256;
                AdvanceBytes(_Destination, _Offset);
                _Bytes &= 255;
            }
            else
            {
                if constexpr (_Streaming_ && _Aligned_)
                    _MemsetImplementation<__m256i>::_MemsetAlignedStreaming<512>(_Destination, _Broadcasted, _Bytes >> 9);
                else
                    _MemsetImplementation<__m256i>::_Memset<512>(_Destination, _Broadcasted, _Bytes >> 9);

                _Offset = _Bytes & -512;
                AdvanceBytes(_Destination, _Offset);
                _Bytes &= 511;
            }
        }

        return _Return;
    }
};

template <
    bool _Aligned_,
    bool _Streaming_>
struct _MemsetVectorizedChooser<_Aligned_, _Streaming_, arch::CpuFeature::AVX512F> {
    template <typename _MemsetType_>
    using _MemsetImplementation = _VectorizedMemsetImplementation<arch::CpuFeature::AVX512F, _MemsetType_, _Aligned_>;

    template <typename _Type_>
    simd_stl_always_inline void* operator()(
        void* _Destination,
        _Type_      _Value,
        sizetype    _Bytes) const noexcept
    {
        void* _Return = _Destination;
        sizetype _Offset = 0;

        __m512i _Broadcasted;
        if (_Bytes > 16)
            _Broadcasted = numeric::_SimdBroadcast<arch::CpuFeature::AVX512F, numeric::zmm512, __m512i>(_Value);

        while (_Bytes) {
            if (_Bytes < 16)
            {
                _MemsetScalar(_Destination, _Value, _Bytes / sizeof(_Type_));
                _Offset = _Bytes;
                AdvanceBytes(_Destination, _Offset);
                _Bytes = 0;
            }
            else if (_Bytes < 32)
            {
                if constexpr (_Streaming_ && _Aligned_)
                    _MemsetImplementation<__m128i>::_MemsetAlignedStreaming<16>(_Destination, numeric::_IntrinBitcast<__m128i>(_Broadcasted), _Bytes >> 4);
                else
                    _MemsetImplementation<__m128i>::_Memset<16>(_Destination, numeric::_IntrinBitcast<__m128i>(_Broadcasted), _Bytes >> 4);

                _Offset = _Bytes & -16;
                AdvanceBytes(_Destination, _Offset);
                _Bytes &= 15;
            }
            else if (_Bytes < 64)
            {
                if constexpr (_Streaming_ && _Aligned_)
                    _MemsetImplementation<__m256i>::_MemsetAlignedStreaming<32>(_Destination, numeric::_IntrinBitcast<__m256i>(_Broadcasted), _Bytes >> 5);
                else
                    _MemsetImplementation<__m256i>::_Memset<32>(_Destination, numeric::_IntrinBitcast<__m256i>(_Broadcasted), _Bytes >> 5);

                _Offset = _Bytes & -32;
                AdvanceBytes(_Destination, _Offset);
                _Bytes &= 31;
            }
            else if (_Bytes < 128)
            {
                if constexpr (_Streaming_ && _Aligned_)
                    _MemsetImplementation<__m512i>::_MemsetAlignedStreaming<64>(_Destination, _Broadcasted, _Bytes >> 6);
                else
                    _MemsetImplementation<__m512i>::_Memset<64>(_Destination, _Broadcasted, _Bytes >> 6);

                _Offset = _Bytes & -64;
                AdvanceBytes(_Destination, _Offset);
                _Bytes &= 63;
            }
            else if (_Bytes < 256)
            {
                if constexpr (_Streaming_ && _Aligned_)
                    _MemsetImplementation<__m512i>::_MemsetAlignedStreaming<128>(_Destination, _Broadcasted, _Bytes >> 7);
                else
                    _MemsetImplementation<__m512i>::_Memset<128>(_Destination, _Broadcasted, _Bytes >> 7);

                _Offset = _Bytes & -128;
                AdvanceBytes(_Destination, _Offset);
                _Bytes &= 127;
            }
            else if (_Bytes < 512)
            {
                if constexpr (_Streaming_ && _Aligned_)
                    _MemsetImplementation<__m512i>::_MemsetAlignedStreaming<256>(_Destination, _Broadcasted, _Bytes >> 8);
                else
                    _MemsetImplementation<__m512i>::_Memset<256>(_Destination, _Broadcasted, _Bytes >> 8);

                _Offset = _Bytes & -256;
                AdvanceBytes(_Destination, _Offset);
                _Bytes &= 255;
            }
            else if (_Bytes < 1024)
            {
                if constexpr (_Streaming_ && _Aligned_)
                    _MemsetImplementation<__m512i>::_MemsetAlignedStreaming<512>(_Destination, _Broadcasted, _Bytes >> 9);
                else
                    _MemsetImplementation<__m512i>::_Memset<512>(_Destination, _Broadcasted, _Bytes >> 9);

                _Offset = _Bytes & -512;
                AdvanceBytes(_Destination, _Offset);
                _Bytes &= 511;
            }
            else if (_Bytes < 2048)
            {
                if constexpr (_Streaming_ && _Aligned_)
                    _MemsetImplementation<__m512i>::_MemsetAlignedStreaming<1024>(_Destination, _Broadcasted, _Bytes >> 10);
                else
                    _MemsetImplementation<__m512i>::_Memset<1024>(_Destination, _Broadcasted, _Bytes >> 10);

                _Offset = _Bytes & -1024;
                AdvanceBytes(_Destination, _Offset);
                _Bytes &= 1023;
            }
            else if (_Bytes < 4096)
            {
                if constexpr (_Streaming_ && _Aligned_)
                    _MemsetImplementation<__m512i>::_MemsetAlignedStreaming<2048>(_Destination, _Broadcasted, _Bytes >> 11);
                else
                    _MemsetImplementation<__m512i>::_Memset<2048>(_Destination, _Broadcasted, _Bytes >> 11);

                _Offset = _Bytes & -2048;
                AdvanceBytes(_Destination, _Offset);
                _Bytes &= 2047;
            }
            else
            {
                if constexpr (_Streaming_ && _Aligned_)
                    _MemsetImplementation<__m512i>::_MemsetAlignedStreaming<4096>(_Destination, _Broadcasted, _Bytes >> 12);
                else
                    _MemsetImplementation<__m512i>::_Memset<4096>(_Destination, _Broadcasted, _Bytes >> 12);

                _Offset = _Bytes & -4096;
                AdvanceBytes(_Destination, _Offset);
                _Bytes &= 4095;
            }
        }

        return _Return;
    }
};

#pragma endregion

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Type_> 
simd_stl_always_inline void* _MemsetVectorizedInternal(
    void*       _Destination,
    _Type_      _Value,
    sizetype    _Bytes) noexcept
{
    using _SimdType_ = type_traits::__deduce_simd_vector_type<_SimdGeneration_, _Type_>;
    void* _Return = _Destination;

    const auto _DestinationAligned = ((uintptr)_Destination & (sizeof(_SimdType_) - 1));

    if (_DestinationAligned == 0)
    {
        if (_Value == 0)
        {
            if (_Bytes > __SIMD_STL_FILL_CACHE_SIZE_LIMIT)
                _MemsetZerosVectorizedChooser<true, true, _SimdGeneration_>()(_Destination, _Bytes);
            else
                _MemsetZerosVectorizedChooser<true, false, _SimdGeneration_>()(_Destination, _Bytes);
        }
        else
        {
            if (_Bytes > __SIMD_STL_FILL_CACHE_SIZE_LIMIT)
                _MemsetVectorizedChooser<true, true, _SimdGeneration_>()(_Destination, _Value, _Bytes);
            else
                _MemsetVectorizedChooser<true, false, _SimdGeneration_>()(_Destination, _Value, _Bytes);
        }
    }
    else
    {
        sizetype _BytesToAlign = sizeof(_SimdType_) - _DestinationAligned;

        void* _DestinationOffset = _Destination;
        AdvanceBytes(_DestinationOffset, _BytesToAlign);

        if (_Value == 0)
        {
            if (_Bytes > _BytesToAlign)
            {
                _MemsetZerosVectorizedChooser<false, false, _SimdGeneration_>()(_Destination, _BytesToAlign);

                if ((_Bytes - _BytesToAlign) > __SIMD_STL_FILL_CACHE_SIZE_LIMIT)
                    _MemsetZerosVectorizedChooser<true, true, _SimdGeneration_>()(_DestinationOffset, _Bytes - _BytesToAlign);
                else
                    _MemsetZerosVectorizedChooser<true, false, _SimdGeneration_>()(_DestinationOffset, _Bytes - _BytesToAlign);
            }
            else
            {
                _MemsetZerosVectorizedChooser<false, false, _SimdGeneration_>()(_Destination, _Bytes);
            }
        }
        else
        {
            if (_Bytes > _BytesToAlign)
            {
                _MemsetVectorizedChooser<false, false, _SimdGeneration_>()(_Destination, _Value, _BytesToAlign);

                if ((_Bytes - _BytesToAlign) > __SIMD_STL_FILL_CACHE_SIZE_LIMIT)
                    _MemsetVectorizedChooser<true, true, _SimdGeneration_>()(_DestinationOffset, _Value, _Bytes - _BytesToAlign);
                else
                    _MemsetVectorizedChooser<true, false, _SimdGeneration_>()(_DestinationOffset, _Value, _Bytes - _BytesToAlign);
            }
            else
            {
                _MemsetVectorizedChooser<false, false, _SimdGeneration_>()(_Destination, _Value, _Bytes);
            }
        }
    }

    return _Return;
}

template <typename _Type_>
void* simd_stl_stdcall _MemsetVectorized(
    void*       _Destination,
    _Type_      _Value,
    sizetype    _Bytes) noexcept
{
    if (arch::ProcessorFeatures::AVX512F())
        return _MemsetVectorizedInternal<arch::CpuFeature::AVX512F>(_Destination, _Value, _Bytes);
    else if (arch::ProcessorFeatures::AVX())
        return _MemsetVectorizedInternal<arch::CpuFeature::AVX>(_Destination, _Value, _Bytes);
     else if (arch::ProcessorFeatures::SSE2())
        return _MemsetVectorizedInternal<arch::CpuFeature::SSE2>(_Destination, _Value, _Bytes);

    return _MemsetScalar(_Destination, _Value, _Bytes / sizeof(_Type_));
}

__SIMD_STL_ALGORITHM_NAMESPACE_END
