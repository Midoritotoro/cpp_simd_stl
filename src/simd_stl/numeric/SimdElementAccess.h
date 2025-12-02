#pragma once 

#include <src/simd_stl/numeric/SimdMemoryAccess.h>
#include <simd_stl/memory/pointerToIntegral.h>


__SIMD_STL_NUMERIC_NAMESPACE_BEGIN

template <
    arch::CpuFeature    _SimdGeneration_, 
    class               _RegisterPolicy_>
class _SimdElementAccess;


template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_,
    typename            _DesiredType_,
    typename            _VectorType_>
simd_stl_always_inline void _SimdInsert(
    _VectorType_&       _Vector,
    const uint8         _Position,
    const _DesiredType_ _Value) noexcept;

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_,
    typename            _DesiredType_,
    typename            _VectorType_>
simd_stl_always_inline _DesiredType_ _SimdExtract(
    _VectorType_    _Vector,
    const uint8     _Where) noexcept;

#pragma region Sse2-Sse4.2 Simd element access

template <>
class _SimdElementAccess<arch::CpuFeature::SSE2, xmm128> {
    static constexpr auto _Generation   = arch::CpuFeature::SSE2;
    using _RegisterPolicy               = xmm128;
public:
    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline void _Insert(
        _VectorType_&       _Vector,
        const uint8         _Position,
        const _DesiredType_ _Value) noexcept
    {
        if constexpr (_Is_epi64_v<_DesiredType_> || _Is_epu64_v<_DesiredType_>) {
#if defined(simd_stl_processor_x86_64)
            auto _VectorValue = _mm_cvtsi64_si128(memory::pointerToIntegral(_Value));
#else
            union {
                __m128i _Vector;
                int64   _Number;
            } _Convert;

            _Convert._Number = _Value;
            auto _VectorValue = _SimdLoadLowerHalf<_Generation, _RegisterPolicy, __m128i>(&_Convert._Vector);
#endif
            _Vector = (_Position == 0) 
                ? _IntrinBitcast<_VectorType_>(_mm_unpackhi_epi64(
                    _mm_unpacklo_epi64(_VectorValue, _VectorValue), _IntrinBitcast<__m128i>(_Vector)))
                : _IntrinBitcast<_VectorType_>(_mm_unpacklo_epi64(
                    _IntrinBitcast<__m128i>(_Vector), _VectorValue));
        }
        else if constexpr (_Is_epi16_v<_DesiredType_> || _Is_epu16_v<_DesiredType_>) {
            switch (_Position) {
                case 0: 
                    _Vector = _IntrinBitcast<_VectorType_>(_mm_insert_epi16(_IntrinBitcast<__m128i>(_Vector), _Value, 0)); 
                    break;
                case 1:
                    _Vector = _IntrinBitcast<_VectorType_>(_mm_insert_epi16(_IntrinBitcast<__m128i>(_Vector), _Value, 1));
                    break;
                case 2:
                    _Vector = _IntrinBitcast<_VectorType_>(_mm_insert_epi16(_IntrinBitcast<__m128i>(_Vector), _Value, 2)); 
                    break;
                case 3:
                    _Vector = _IntrinBitcast<_VectorType_>(_mm_insert_epi16(_IntrinBitcast<__m128i>(_Vector), _Value, 3)); 
                    break;
                case 4: 
                    _Vector = _IntrinBitcast<_VectorType_>(_mm_insert_epi16(_IntrinBitcast<__m128i>(_Vector), _Value, 4)); 
                    break;
                case 5: 
                    _Vector = _IntrinBitcast<_VectorType_>(_mm_insert_epi16(_IntrinBitcast<__m128i>(_Vector), _Value, 5));
                    break;
                case 6: 
                    _Vector = _IntrinBitcast<_VectorType_>(_mm_insert_epi16(_IntrinBitcast<__m128i>(_Vector), _Value, 6));
                    break;
                default:
                    _Vector = _IntrinBitcast<_VectorType_>(_mm_insert_epi16(_IntrinBitcast<__m128i>(_Vector), _Value, 7));
                    break;
            }
        }
        else if constexpr (_Is_pd_v<_DesiredType_>) {
            const auto _Broadcasted = _mm_set_sd(_Value);

            _Vector = (_Position == 0)
                ? _mm_shuffle_pd(_Broadcasted, _IntrinBitcast<__m128d>(_Vector), 2)
                : _mm_shuffle_pd(_IntrinBitcast<__m128d>(_Vector), _Broadcasted, 0);
        }
        else {
            const auto _Mask = _MakeSimdInsertMask<_VectorType_, _DesiredType_>();

            const auto _Broadcasted = _SimdBroadcast<_Generation, _RegisterPolicy, _VectorType_>(memory::pointerToIntegral(_Value));
            const auto _InsertMask = _SimdLoadUnaligned<_Generation, _RegisterPolicy, _VectorType_>(
                (_Mask._Array + _Mask._Offset - (_Position & (_Mask._Offset - 1))));

            _Vector = _SimdBlend<_Generation, _RegisterPolicy, _DesiredType_>(_Vector, _Broadcasted, _InsertMask);
        }
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _DesiredType_ _Extract(
        _VectorType_    _Vector,
        const uint8     _Where) noexcept
    {
        if constexpr (_Is_epi64_v<_DesiredType_> || _Is_epu64_v<_DesiredType_>) {
            if (_Where == 0) {
#if defined(simd_stl_processor_x86_64)
                return static_cast<_DesiredType_>(_mm_cvtsi128_si64(_IntrinBitcast<__m128i>(_Vector)));
#else
                const auto _HighDword = _mm_cvtsi128_si32(_IntrinBitcast<__m128i>(_Vector));
                const auto _LowDword = _mm_cvtsi128_si32(_mm_shuffle_epi32(_IntrinBitcast<__m128i>(_Vector), 0x55));

                return (static_cast<int64>(_HighDword) << 32) | static_cast<int64>(_LowDword);
#endif // defined(simd_stl_processor_x86_64)
            }

#if defined(simd_stl_processor_x86_64)
            return static_cast<_DesiredType_>(_mm_cvtsi128_si64(_mm_shuffle_epi32(_IntrinBitcast<__m128i>(_Vector), 0xEE)));
#else
            const auto _HighDword = _mm_cvtsi128_si32(_mm_shuffle_epi32(_IntrinBitcast<__m128i>(_Vector), 0xEE));
            const auto _LowDword = _mm_cvtsi128_si32(_mm_shuffle_epi32(_IntrinBitcast<__m128i>(_Vector), 0xFF));

            return (static_cast<int64>(_HighDword) << 32) | static_cast<int64>(_LowDword);
#endif // defined(simd_stl_processor_x86_64)
        }
        else if constexpr (_Is_epi32_v<_DesiredType_> || _Is_epu32_v<_DesiredType_>) {
            switch (_Where) {
                case 0: 
                    return static_cast<_DesiredType_>(_mm_cvtsi128_si32(_IntrinBitcast<__m128i>(_Vector)));

                case 1:
                    return static_cast<_DesiredType_>(_mm_cvtsi128_si32(_mm_shuffle_epi32(_IntrinBitcast<__m128i>(_Vector), 0x55)));

                case 2:
                    return static_cast<_DesiredType_>(_mm_cvtsi128_si32(_mm_shuffle_epi32(_IntrinBitcast<__m128i>(_Vector), 0xEE)));

                default:
                    return static_cast<_DesiredType_>(_mm_cvtsi128_si32(_mm_shuffle_epi32(_IntrinBitcast<__m128i>(_Vector), 0xFF)));
            }
        }
        else if constexpr (_Is_epi16_v<_DesiredType_> || _Is_epu16_v<_DesiredType_>) {
            switch (_Where) {
                case 0: 
                    return static_cast<_DesiredType_>(_mm_extract_epi16(_IntrinBitcast<__m128i>(_Vector), 0));
                case 1: 
                    return static_cast<_DesiredType_>(_mm_extract_epi16(_IntrinBitcast<__m128i>(_Vector), 1));
                case 2: 
                    return static_cast<_DesiredType_>(_mm_extract_epi16(_IntrinBitcast<__m128i>(_Vector), 2));
                case 3: 
                    return static_cast<_DesiredType_>(_mm_extract_epi16(_IntrinBitcast<__m128i>(_Vector), 3));
                case 4: 
                    return static_cast<_DesiredType_>(_mm_extract_epi16(_IntrinBitcast<__m128i>(_Vector), 4));
                case 5: 
                    return static_cast<_DesiredType_>(_mm_extract_epi16(_IntrinBitcast<__m128i>(_Vector), 5));
                case 6: 
                    return static_cast<_DesiredType_>(_mm_extract_epi16(_IntrinBitcast<__m128i>(_Vector), 6));
                default: 
                    return static_cast<_DesiredType_>(_mm_extract_epi16(_IntrinBitcast<__m128i>(_Vector), 7));
            }
        }
        else {
            constexpr auto _Length = sizeof(_VectorType_) / sizeof(_DesiredType_);
            _DesiredType_ _Array[_Length];

            _SimdStoreUnaligned<_Generation, _RegisterPolicy>(_Array, _Vector);
            return _Array[_Where & (_Length - 1)];
        }
    }
};

template <>
class _SimdElementAccess<arch::CpuFeature::SSE3, xmm128>:
    public _SimdElementAccess<arch::CpuFeature::SSE2, xmm128>
{};

template <>
class _SimdElementAccess<arch::CpuFeature::SSSE3, xmm128> :
    public _SimdElementAccess<arch::CpuFeature::SSE3, xmm128>
{};

template <>
class _SimdElementAccess<arch::CpuFeature::SSE41, xmm128> :
    public _SimdElementAccess<arch::CpuFeature::SSSE3, xmm128>
{
    static constexpr auto _Generation   = arch::CpuFeature::SSE2;
    using _RegisterPolicy               = xmm128;
public:
    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline void _Insert(
        _VectorType_&       _Vector,
        const uint8         _Position,
        const _DesiredType_ _Value) noexcept
    {
        if constexpr (_Is_epi64_v<_DesiredType_> || _Is_epu64_v<_DesiredType_>) {
            auto _QwordValue = memory::pointerToIntegral(_Value);

            switch (_Position) {
                case 0: 
                    _Vector = _IntrinBitcast<_VectorType_>(_mm_insert_epi64(_IntrinBitcast<__m128i>(_Vector), _QwordValue, 0));
                    break;
                default:
                    _Vector = _IntrinBitcast<_VectorType_>(_mm_insert_epi64(_IntrinBitcast<__m128i>(_Vector), _QwordValue, 1));
                    break;
            }
        }
        else if constexpr (_Is_epi32_v<_DesiredType_> || _Is_epu32_v<_DesiredType_>) {
            auto _DwordValue = memory::pointerToIntegral(_Value);

            switch (_Position) {
                case 0: 
                    _Vector = _IntrinBitcast<_VectorType_>(_mm_insert_epi32(_IntrinBitcast<__m128i>(_Vector), _DwordValue, 0));
                    break;
                case 1: 
                    _Vector = _IntrinBitcast<_VectorType_>(_mm_insert_epi32(_IntrinBitcast<__m128i>(_Vector), _DwordValue, 1));
                    break;
                case 2: 
                    _Vector = _IntrinBitcast<_VectorType_>(_mm_insert_epi32(_IntrinBitcast<__m128i>(_Vector), _DwordValue, 2));
                    break;
                default: 
                    _Vector = _IntrinBitcast<_VectorType_>(_mm_insert_epi32(_IntrinBitcast<__m128i>(_Vector), _DwordValue, 3));
                    break;
            }
        }
        else if constexpr (_Is_epi16_v<_DesiredType_> || _Is_epu16_v<_DesiredType_>) {
            switch (_Position) {
                case 0: 
                    _Vector = _IntrinBitcast<_VectorType_>(_mm_insert_epi16(_IntrinBitcast<__m128i>(_Vector), _Value, 0));
                    break;
                case 1: 
                    _Vector = _IntrinBitcast<_VectorType_>(_mm_insert_epi16(_IntrinBitcast<__m128i>(_Vector), _Value, 1));
                    break;
                case 2: 
                    _Vector = _IntrinBitcast<_VectorType_>(_mm_insert_epi16(_IntrinBitcast<__m128i>(_Vector), _Value, 2)); 
                    break;
                case 3: 
                    _Vector = _IntrinBitcast<_VectorType_>(_mm_insert_epi16(_IntrinBitcast<__m128i>(_Vector), _Value, 3));
                    break;
                case 4: 
                    _Vector = _IntrinBitcast<_VectorType_>(_mm_insert_epi16(_IntrinBitcast<__m128i>(_Vector), _Value, 4)); 
                    break;
                case 5:
                    _Vector = _IntrinBitcast<_VectorType_>(_mm_insert_epi16(_IntrinBitcast<__m128i>(_Vector), _Value, 5));
                    break;
                case 6: 
                    _Vector = _IntrinBitcast<_VectorType_>(_mm_insert_epi16(_IntrinBitcast<__m128i>(_Vector), _Value, 6));
                    break;
                default: 
                    _Vector = _IntrinBitcast<_VectorType_>(_mm_insert_epi16(_IntrinBitcast<__m128i>(_Vector), _Value, 7));
                    break;
            }
        }
        else {
            return _SimdInsert<arch::CpuFeature::SSE2, _RegisterPolicy>(_Vector, _Position, _Value);
        }
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _DesiredType_ _Extract(
        _VectorType_    _Vector,
        const uint8     _Where) noexcept
    {
        if constexpr (_Is_epi64_v<_DesiredType_> || _Is_epu64_v<_DesiredType_>) {
            switch (_Where) {
                case 0: 
                    return static_cast<_DesiredType_>(_mm_extract_epi64(_IntrinBitcast<__m128i>(_Vector), 0));
                default: 
                    return static_cast<_DesiredType_>(_mm_extract_epi64(_IntrinBitcast<__m128i>(_Vector), 1));
            }
        }
        else if constexpr (_Is_epi32_v<_DesiredType_> || _Is_epu32_v<_DesiredType_>) {
            switch (_Where) {
                case 0: 
                    return static_cast<_DesiredType_>(_mm_extract_epi32(_IntrinBitcast<__m128i>(_Vector), 0));
                case 1: 
                    return static_cast<_DesiredType_>(_mm_extract_epi32(_IntrinBitcast<__m128i>(_Vector), 1));
                case 2: 
                    return static_cast<_DesiredType_>(_mm_extract_epi32(_IntrinBitcast<__m128i>(_Vector), 2));
                default: 
                    return static_cast<_DesiredType_>(_mm_extract_epi32(_IntrinBitcast<__m128i>(_Vector), 3));
            }
        }
        else if constexpr (_Is_epi16_v<_DesiredType_> || _Is_epu16_v<_DesiredType_>) {
            switch (_Where) {
                case 0: 
                    return static_cast<_DesiredType_>(_mm_extract_epi16(_IntrinBitcast<__m128i>(_Vector), 0));
                case 1:
                    return static_cast<_DesiredType_>(_mm_extract_epi16(_IntrinBitcast<__m128i>(_Vector), 1));
                case 2: 
                    return static_cast<_DesiredType_>(_mm_extract_epi16(_IntrinBitcast<__m128i>(_Vector), 2));
                case 3:
                    return static_cast<_DesiredType_>(_mm_extract_epi16(_IntrinBitcast<__m128i>(_Vector), 3));
                case 4: 
                    return static_cast<_DesiredType_>(_mm_extract_epi16(_IntrinBitcast<__m128i>(_Vector), 4));
                case 5:
                    return static_cast<_DesiredType_>(_mm_extract_epi16(_IntrinBitcast<__m128i>(_Vector), 5));
                case 6: 
                    return static_cast<_DesiredType_>(_mm_extract_epi16(_IntrinBitcast<__m128i>(_Vector), 6));
                default:
                    return static_cast<_DesiredType_>(_mm_extract_epi16(_IntrinBitcast<__m128i>(_Vector), 7));
            }
        }
        else {
            return _SimdExtract<arch::CpuFeature::SSE2, _RegisterPolicy, _DesiredType_>(_Vector, _Where);
        }
    }
};

template <>
class _SimdElementAccess<arch::CpuFeature::SSE42, xmm128>:
    public _SimdElementAccess<arch::CpuFeature::SSE41, xmm128>
{};

#pragma endregion

#pragma region Avx-Avx2 Simd element access 

template <>
class _SimdElementAccess<arch::CpuFeature::AVX, ymm256>
{
    static constexpr auto _Generation   = arch::CpuFeature::AVX;
    using _RegisterPolicy               = numeric::ymm256;
public:
    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline void _Insert(
        _VectorType_&       _Vector,
        const uint8         _Position,
        const _DesiredType_ _Value) noexcept
    {
        if constexpr (_Is_pd_v<_DesiredType_>) {
            const auto _Broadcasted = _mm256_broadcast_sd(&_Value);

            switch (_Position) {
                case 0:
                    _Vector = _mm256_blend_pd(_IntrinBitcast<__m256d>(_Vector), _Broadcasted, 1);
                    break;
                case 1:
                    _Vector = _mm256_blend_pd(_IntrinBitcast<__m256d>(_Vector), _Broadcasted, 2);
                    break;
                case 2:
                    _Vector = _mm256_blend_pd(_IntrinBitcast<__m256d>(_Vector), _Broadcasted, 4);
                    break;
                default:
                    _Vector = _mm256_blend_pd(_IntrinBitcast<__m256d>(_Vector), _Broadcasted, 8);
                    break;
            }
        }
        else if constexpr (_Is_ps_v<_DesiredType_>) {
            const auto _Broadcasted = _mm256_broadcast_ss(&_Value);

            switch (_Position) {
                case 0:
                    _Vector = _mm256_blend_ps(_IntrinBitcast<__m256>(_Vector), _Broadcasted, 1);
                    break;
                case 1:
                    _Vector = _mm256_blend_ps(_IntrinBitcast<__m256>(_Vector), _Broadcasted, 2);
                    break;
                case 2:
                    _Vector = _mm256_blend_ps(_IntrinBitcast<__m256>(_Vector), _Broadcasted, 4);
                    break;
                case 3:
                    _Vector = _mm256_blend_ps(_IntrinBitcast<__m256>(_Vector), _Broadcasted, 8);
                    break;
                case 4:
                    _Vector = _mm256_blend_ps(_IntrinBitcast<__m256>(_Vector), _Broadcasted, 0x10);
                    break;
                case 5:
                    _Vector = _mm256_blend_ps(_IntrinBitcast<__m256>(_Vector), _Broadcasted, 0x20);
                    break;
                case 6:
                    _Vector = _mm256_blend_ps(_IntrinBitcast<__m256>(_Vector), _Broadcasted, 0x40);
                    break;
                default:
                    _Vector = _mm256_blend_ps(_IntrinBitcast<__m256>(_Vector), _Broadcasted, 0x80);
                    break;
            }
        }
        else if constexpr (_Is_epi64_v<_DesiredType_> || _Is_epu64_v<_DesiredType_>) {
            auto _QwordValue = memory::pointerToIntegral(_Value);

            switch (_Position) {
                case 0: 
                    _Vector = _IntrinBitcast<_VectorType_>(_mm256_insert_epi64(_IntrinBitcast<__m256i>(_Vector), _QwordValue, 0));
                    break;
                case 1: 
                    _Vector = _IntrinBitcast<_VectorType_>(_mm256_insert_epi64(_IntrinBitcast<__m256i>(_Vector), _QwordValue, 1));
                    break;
                case 2:
                    _Vector = _IntrinBitcast<_VectorType_>(_mm256_insert_epi64(_IntrinBitcast<__m256i>(_Vector), _QwordValue, 2));
                    break;
                default: 
                    _Vector = _IntrinBitcast<_VectorType_>(_mm256_insert_epi64(_IntrinBitcast<__m256i>(_Vector), _QwordValue, 3));
                    break;
            }
        }
        else if constexpr (_Is_epi32_v<_DesiredType_> || _Is_epu32_v<_DesiredType_>) {
            auto _DwordValue = memory::pointerToIntegral(_Value);

            switch (_Position) {
                case 0: 
                    _Vector = _IntrinBitcast<_VectorType_>(_mm256_insert_epi32(_IntrinBitcast<__m256i>(_Vector), _DwordValue, 0));
                    break;
                case 1: 
                    _Vector = _IntrinBitcast<_VectorType_>(_mm256_insert_epi32(_IntrinBitcast<__m256i>(_Vector), _DwordValue, 1));
                    break;
                case 2: 
                    _Vector = _IntrinBitcast<_VectorType_>(_mm256_insert_epi32(_IntrinBitcast<__m256i>(_Vector), _DwordValue, 2));
                    break;
                case 3: 
                    _Vector = _IntrinBitcast<_VectorType_>(_mm256_insert_epi32(_IntrinBitcast<__m256i>(_Vector), _DwordValue, 3));
                    break;
                case 4: 
                    _Vector = _IntrinBitcast<_VectorType_>(_mm256_insert_epi32(_IntrinBitcast<__m256i>(_Vector), _DwordValue, 4));
                    break;
                case 5: 
                    _Vector = _IntrinBitcast<_VectorType_>(_mm256_insert_epi32(_IntrinBitcast<__m256i>(_Vector), _DwordValue, 5));
                    break;
                case 6:
                    _Vector = _IntrinBitcast<_VectorType_>(_mm256_insert_epi32(_IntrinBitcast<__m256i>(_Vector), _DwordValue, 6));
                    break;
                default: 
                    _Vector = _IntrinBitcast<_VectorType_>(_mm256_insert_epi32(_IntrinBitcast<__m256i>(_Vector), _DwordValue, 7));
                    break;
            }
        }
        else {
            const auto _Mask = _MakeSimdInsertMask<_VectorType_, _DesiredType_>();

            const auto _Broadcasted = _SimdBroadcast<_Generation, _RegisterPolicy, _VectorType_>(_Value);
            const auto _InsertMask  = _SimdLoadUnaligned<_Generation, _RegisterPolicy, _VectorType_>(
                (_Mask._Array + _Mask._Offset - (_Position & (_Mask._Offset - 1))));

            _Vector = _SimdBlend<_Generation, _RegisterPolicy, _DesiredType_>(_Vector, _Broadcasted, _InsertMask);
        }
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _DesiredType_ _Extract(
        _VectorType_    _Vector,
        const uint8     _Where) noexcept
    {
        if constexpr (_Is_epi64_v<_DesiredType_> || _Is_epu64_v<_DesiredType_>) {
            switch (_Where) {
                case 0:
                    return static_cast<_DesiredType_>(_mm256_extract_epi64(_IntrinBitcast<__m256i>(_Vector), 0));
                case 1: 
                    return static_cast<_DesiredType_>(_mm256_extract_epi64(_IntrinBitcast<__m256i>(_Vector), 1));
                case 2:
                    return static_cast<_DesiredType_>(_mm256_extract_epi64(_IntrinBitcast<__m256i>(_Vector), 2));
                case 3:
                    return static_cast<_DesiredType_>(_mm256_extract_epi64(_IntrinBitcast<__m256i>(_Vector), 3));
            }
        }
        else if constexpr (_Is_epi32_v<_DesiredType_> || _Is_epu32_v<_DesiredType_>) {
            switch (_Where) {
                case 0: 
                    return static_cast<_DesiredType_>(_mm256_extract_epi32(_IntrinBitcast<__m256i>(_Vector), 0));
                case 1:
                    return static_cast<_DesiredType_>(_mm256_extract_epi32(_IntrinBitcast<__m256i>(_Vector), 1));
                case 2: 
                    return static_cast<_DesiredType_>(_mm256_extract_epi32(_IntrinBitcast<__m256i>(_Vector), 2));
                case 3: 
                    return static_cast<_DesiredType_>(_mm256_extract_epi32(_IntrinBitcast<__m256i>(_Vector), 3));
                case 4: 
                    return static_cast<_DesiredType_>(_mm256_extract_epi32(_IntrinBitcast<__m256i>(_Vector), 4));
                case 5: 
                    return static_cast<_DesiredType_>(_mm256_extract_epi32(_IntrinBitcast<__m256i>(_Vector), 5));
                case 6: 
                    return static_cast<_DesiredType_>(_mm256_extract_epi32(_IntrinBitcast<__m256i>(_Vector), 6));
                case 7: 
                    return static_cast<_DesiredType_>(_mm256_extract_epi32(_IntrinBitcast<__m256i>(_Vector), 7));
            }
        }
        else {
            constexpr auto _Length = sizeof(_VectorType_) / sizeof(_DesiredType_);
            _DesiredType_ _Array[_Length];

            _SimdStoreUnaligned<_Generation, _RegisterPolicy>(_Array, _Vector);
            return _Array[_Where & (_Length - 1)];
        }
    }
};

template <>
class _SimdElementAccess<arch::CpuFeature::AVX2, ymm256>:
    public _SimdElementAccess<arch::CpuFeature::AVX, ymm256>
{};

#pragma endregion

#pragma region Avx512 Simd element access

template <>
class _SimdElementAccess<arch::CpuFeature::AVX512F, zmm512>
{
    static constexpr auto _Generation   = arch::CpuFeature::AVX512F;
    using _RegisterPolicy               = zmm512;
public:
    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline void _Insert(
        _VectorType_&       _Vector,
        const uint8         _Position,
        const _DesiredType_ _Value) noexcept
    {
        if constexpr (_Is_epi64_v<_DesiredType_> || _Is_epu64_v<_DesiredType_>) {
            _Vector = _IntrinBitcast<_VectorType_>(_mm512_mask_set1_epi64(
                _IntrinBitcast<__m512i>(_Vector), static_cast<uint8>(1u << _Position), memory::pointerToIntegral(_Value)));
        }
        else if constexpr (_Is_epi32_v<_DesiredType_> || _Is_epu32_v<_DesiredType_>) {
            _Vector = _IntrinBitcast<_VectorType_>(_mm512_mask_set1_epi32(
                _IntrinBitcast<__m512i>(_Vector), static_cast<uint16>(1u << _Position), memory::pointerToIntegral(_Value)));
        }
        else if constexpr (_Is_ps_v<_DesiredType_>) {
            _Vector = _IntrinBitcast<_VectorType_>(_mm512_mask_broadcastss_ps(
                _IntrinBitcast<__m512>(_Vector), static_cast<uint16>(1u << _Position), _mm_set_ss(_Value)));
        }
        else if constexpr (_Is_pd_v<_DesiredType_>) {
            _Vector = _IntrinBitcast<_VectorType_>(_mm512_mask_broadcastsd_pd(
                _IntrinBitcast<__m512d>(_Vector), static_cast<uint8>(1u << _Position), _mm_set_sd(_Value)));
        }
        else {
            const auto _Mask = _MakeSimdInsertMask<_VectorType_, _DesiredType_>();

            const auto _Broadcasted = _SimdBroadcast<_Generation, _RegisterPolicy, _VectorType_>(_Value);
            const auto _InsertMask  = _SimdLoadUnaligned<_Generation, _RegisterPolicy, _VectorType_>(
                (_Mask._Array + _Mask._Offset - (_Position & (_Mask._Offset - 1))));

            _Vector = _SimdBlend<_Generation, _RegisterPolicy, _DesiredType_>(_Vector, _Broadcasted, _InsertMask);
        }
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _DesiredType_ _Extract(
        _VectorType_    _Vector,
        const uint8     _Where) noexcept
    {
        if constexpr (_Is_pd_v<_DesiredType_>) {
            return _mm512_cvtsd_f64(_mm512_maskz_compress_pd(static_cast<uint8>(1u << _Where), _IntrinBitcast<__m512d>(_Vector)));
        }
        else if constexpr (_Is_ps_v<_DesiredType_>) {
            return _mm512_cvtss_f32(_mm512_maskz_compress_ps(static_cast<uint16>(1u << _Where), _IntrinBitcast<__m512>(_Vector)));
        }
        else if constexpr (_Is_epi32_v<_DesiredType_> || _Is_epu32_v<_DesiredType_>) {
            return _mm_cvtsi128_si32(_IntrinBitcast<__m128i>(_mm512_maskz_compress_epi32(
                static_cast<uint16>(1u << _Where), _IntrinBitcast<__m512i>(_Vector))));
        }
        else if constexpr (_Is_epi64_v<_DesiredType_> || _Is_epu64_v<_DesiredType_>) {
            return _mm_cvtsi128_si64(_IntrinBitcast<__m128i>(_mm512_maskz_compress_epi64(
                static_cast<uint16>(1u << _Where), _IntrinBitcast<__m512i>(_Vector))));
        }
        else {
            constexpr auto _Length = sizeof(_VectorType_) / sizeof(_DesiredType_);
            _DesiredType_ _Array[_Length];

            _SimdStoreUnaligned<_Generation, _RegisterPolicy>(_Array, _Vector);
            return _Array[_Where & (_Length - 1)];
        }
    }
};

template <>
class _SimdElementAccess<arch::CpuFeature::AVX512BW, zmm512>:
    public _SimdElementAccess<arch::CpuFeature::AVX512F, zmm512>
{};

template <>
class _SimdElementAccess<arch::CpuFeature::AVX512DQ, zmm512> :
    public _SimdElementAccess<arch::CpuFeature::AVX512BW, zmm512>
{};

template <>
class _SimdElementAccess<arch::CpuFeature::AVX512VL, zmm512> :
    public _SimdElementAccess<arch::CpuFeature::AVX512DQ, zmm512>
{};

#pragma endregion 


template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_,
    typename            _DesiredType_,
    typename            _VectorType_>
simd_stl_always_inline void _SimdInsert(
    _VectorType_&       _Vector,
    const uint8         _Position,
    const _DesiredType_ _Value) noexcept
{
    _VerifyRegisterPolicy(_SimdGeneration_, _RegisterPolicy_);
    _SimdElementAccess<_SimdGeneration_, _RegisterPolicy_>::template _Insert(_Vector, _Position, _Value);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_,
    typename            _DesiredType_,
    typename            _VectorType_>
simd_stl_always_inline _DesiredType_ _SimdExtract(
    _VectorType_    _Vector,
    const uint8     _Where) noexcept
{
    _VerifyRegisterPolicy(_SimdGeneration_, _RegisterPolicy_);
    return _SimdElementAccess<_SimdGeneration_, _RegisterPolicy_>::template _Extract<_DesiredType_>(_Vector, _Where);
}

__SIMD_STL_NUMERIC_NAMESPACE_END
