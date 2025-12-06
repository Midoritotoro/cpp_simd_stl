#pragma once 

#include <src/simd_stl/numeric/IntrinBitcast.h>
#include <simd_stl/math/IntegralTypesConversions.h>


__SIMD_STL_NUMERIC_NAMESPACE_BEGIN

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_>
class _SimdConvertImplementation;

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_,
    typename            _DesiredType_,
    typename            _VectorType_>
simd_stl_always_inline auto _SimdToMask(_VectorType_ _Vector) noexcept;

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_,
    typename            _VectorType_,
    typename            _DesiredType_,
    typename            _MaskType_>
simd_stl_always_inline _VectorType_ _SimdToVector(_MaskType_ _Mask) noexcept;

#pragma region Sse2-Sse4.2 Simd convert

template <>
class _SimdConvertImplementation<arch::CpuFeature::SSE2, xmm128> {
    static constexpr auto _Generation   = arch::CpuFeature::SSE2;
    using _RegisterPolicy               = xmm128;

    template <typename _DesiredType_>
    using _Simd_mask_type = type_traits::__deduce_simd_mask_type<_Generation, _DesiredType_, _RegisterPolicy>;
public:
    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _Simd_mask_type<_DesiredType_> _ToMask(_VectorType_ _Vector) noexcept {
        if      constexpr (sizeof(_DesiredType_) == 8)
            return _mm_movemask_pd(_IntrinBitcast<__m128d>(_Vector));

        else if constexpr (sizeof(_DesiredType_) == 4)
            return _mm_movemask_ps(_IntrinBitcast<__m128>(_Vector));

        else if constexpr (sizeof(_DesiredType_) == 2)
            return _mm_movemask_epi8(_mm_packs_epi16(_IntrinBitcast<__m128i>(_Vector), _mm_setzero_si128()));

        else if constexpr (sizeof(_DesiredType_) == 1)
            return _mm_movemask_epi8(_IntrinBitcast<__m128i>(_Vector));
    }

    template <
        typename _VectorType_,
        typename _DesiredType_>
    static simd_stl_always_inline _VectorType_ _ToVector(_Simd_mask_type<_DesiredType_> _Mask) noexcept {
        constexpr auto _Bits = sizeof(_VectorType_) / sizeof(_DesiredType_);

        if constexpr (_Bits == 2) {
            const auto _First = (_Mask >> 1) & 1;
            const auto _Second = _Mask & 1;

            const auto _Broadcasted = _mm_set_epi32(_First, _First, _Second, _Second);
            return _IntrinBitcast<_VectorType_>(_mm_cmpeq_epi32(_Broadcasted, _mm_set1_epi32(1)));
        }
        else if constexpr (_Bits == 4) {
            const auto _Broadcasted = _mm_set_epi32((_Mask >> 3) & 1, (_Mask >> 2) & 1, (_Mask >> 1) & 1, _Mask & 1);
            return _IntrinBitcast<_VectorType_>(_mm_cmpeq_epi32(_Broadcasted, _mm_set1_epi32(1)));
        }
        else if constexpr (_Bits == 8) {
            const auto _Broadcasted = _mm_set_epi16((_Mask >> 7) & 1, (_Mask >> 6) & 1, (_Mask >> 5) & 1, (_Mask >> 4) & 1,
                (_Mask >> 3) & 1, (_Mask >> 2) & 1, (_Mask >> 1) & 1, _Mask & 1);

            return _IntrinBitcast<_VectorType_>(_mm_cmpeq_epi16(_Broadcasted, _mm_set1_epi16(1)));
        }
        else if constexpr (_Bits == 16) {
            const auto _Broadcasted = _mm_set_epi8((_Mask >> 15) & 1, (_Mask >> 14) & 1, (_Mask >> 13) & 1, (_Mask >> 12) & 1,
                (_Mask >> 11) & 1, (_Mask >> 10) & 1, (_Mask >> 9) & 1, (_Mask >> 8) & 1, (_Mask >> 7) & 1, 
                (_Mask >> 6) & 1, (_Mask >> 5) & 1, (_Mask >> 4) & 1,
                (_Mask >> 3) & 1, (_Mask >> 2) & 1, (_Mask >> 1) & 1, _Mask & 1);

            return _IntrinBitcast<_VectorType_>(_mm_cmpeq_epi8(_Broadcasted, _mm_set1_epi8(1)));
        }
    }
};

template <>
class _SimdConvertImplementation<arch::CpuFeature::SSE3, xmm128> :
    public _SimdConvertImplementation<arch::CpuFeature::SSE2, xmm128>
{};

template <>
class _SimdConvertImplementation<arch::CpuFeature::SSSE3, xmm128> :
    public _SimdConvertImplementation<arch::CpuFeature::SSE3, xmm128>
{
    static constexpr auto _Generation   = arch::CpuFeature::SSE2;
    using _RegisterPolicy               = xmm128;

    template <typename _DesiredType_>
    using _Simd_mask_type = type_traits::__deduce_simd_mask_type<_Generation, _DesiredType_, _RegisterPolicy>;
public:
    template <
        typename _VectorType_,
        typename _DesiredType_>
    static simd_stl_always_inline _VectorType_ _ToVector(_Simd_mask_type<_DesiredType_> _Mask) noexcept {
        constexpr auto _Bits = sizeof(_VectorType_) / sizeof(_DesiredType_);

        if constexpr (_Bits == 2) {
            const auto _First = (_Mask >> 1) & 1;
            const auto _Second = _Mask & 1;

            const auto _Broadcasted = _mm_set_epi32(_First, _First, _Second, _Second);
            return _IntrinBitcast<_VectorType_>(_mm_cmpeq_epi32(_Broadcasted, _mm_set1_epi32(1)));
        }
        else if constexpr (_Bits == 4) {
            const auto _Broadcasted = _mm_set_epi32((_Mask >> 3) & 1, (_Mask >> 2) & 1, (_Mask >> 1) & 1, _Mask & 1);
            return _IntrinBitcast<_VectorType_>(_mm_cmpeq_epi32(_Broadcasted, _mm_set1_epi32(1)));
        }
        else if constexpr (_Bits == 8) {
            const auto _Broadcasted = _mm_set_epi16((_Mask >> 7) & 1, (_Mask >> 6) & 1, (_Mask >> 5) & 1, (_Mask >> 4) & 1,
                (_Mask >> 3) & 1, (_Mask >> 2) & 1, (_Mask >> 1) & 1, _Mask & 1);

            return _IntrinBitcast<_VectorType_>(_mm_cmpeq_epi16(_Broadcasted, _mm_set1_epi16(1)));
        }
        else if constexpr (_Bits == 16) {
            const auto _Select      = _mm_set1_epi64x(0x8040201008040201ull);
            const auto _Shuffled    = _mm_shuffle_epi8(_mm_cvtsi32_si128(_Mask), 
                _mm_set_epi64x(0x0101010101010101ll, 0));

            return _IntrinBitcast<_VectorType_>(_mm_cmpeq_epi8(_mm_and_si128(_Shuffled, _Select), _Select));
        }
    }
};

template <>
class _SimdConvertImplementation<arch::CpuFeature::SSE41, xmm128> :
    public _SimdConvertImplementation<arch::CpuFeature::SSSE3, xmm128>
{};

template <>
class _SimdConvertImplementation<arch::CpuFeature::SSE42, xmm128> :
    public _SimdConvertImplementation<arch::CpuFeature::SSE41, xmm128>
{};

#pragma endregion

#pragma region Avx-Avx2 Simd convert

template <>
class _SimdConvertImplementation<arch::CpuFeature::AVX, ymm256>
{
    static constexpr auto _Generation   = arch::CpuFeature::AVX;
    using _RegisterPolicy               = ymm256;

    template <typename _DesiredType_>
    using _Simd_mask_type = type_traits::__deduce_simd_mask_type<_Generation, _DesiredType_, _RegisterPolicy>;
public:
    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _Simd_mask_type<_DesiredType_> _ToMask(_VectorType_ _Vector) noexcept {
        using _MaskType = _Simd_mask_type<_DesiredType_>;

        if      constexpr (sizeof(_DesiredType_) == 8) {
            return static_cast<_MaskType>(_mm256_movemask_pd(_IntrinBitcast<__m256d>(_Vector)));
        }
        else if constexpr (sizeof(_DesiredType_) == 4) {
            return static_cast<_MaskType>(_mm256_movemask_ps(_IntrinBitcast<__m256>(_Vector)));
        }
        else if constexpr (sizeof(_DesiredType_) == 2) {
            const auto _Zeros = _mm_setzero_si128();

            const auto _Low = _mm_movemask_epi8(_mm_packs_epi16(_IntrinBitcast<__m128i>(_Vector), _Zeros));
            const auto _High = _mm_movemask_epi8(_mm_packs_epi16(_mm256_extractf128_si256(_IntrinBitcast<__m256i>(_Vector), 1), _Zeros));

            return ((static_cast<_MaskType>(_High & 0xFF) << 8) | (static_cast<_MaskType>(_Low & 0xFF)));
        }
        else if constexpr (sizeof(_DesiredType_) == 1) {
            const auto _Low     = _SimdToMask<arch::CpuFeature::SSE2, xmm128, _DesiredType_>(_IntrinBitcast<__m128i>(_Vector));
            const auto _High    = _SimdToMask<arch::CpuFeature::SSE2, xmm128, _DesiredType_>(_mm256_extractf128_si256(
                _IntrinBitcast<__m256i>(_Vector), 1));

            return ((static_cast<_MaskType>(_High & 0xFFFF) << 16) | (static_cast<_MaskType>(_Low & 0xFFFF)));
        }
    }

    template <
        typename _VectorType_,
        typename _DesiredType_>
    static simd_stl_always_inline _VectorType_ _ToVector(_Simd_mask_type<_DesiredType_> _Mask) noexcept {
        constexpr auto _Bits = sizeof(_VectorType_) / sizeof(_DesiredType_);

        if constexpr (_Bits == 4) {
            const auto _Broadcasted = _mm256_set_pd((_Mask >> 3) & 1, (_Mask >> 2) & 1, (_Mask >> 1) & 1, _Mask & 1);
            return _IntrinBitcast<_VectorType_>(_mm256_cmp_pd(_Broadcasted, _mm256_set1_pd(1), 0));
        }
        else if constexpr (_Bits == 8) {
            const auto _Broadcasted = _mm256_set_ps((_Mask >> 7) & 1, (_Mask >> 6) & 1, (_Mask >> 5) & 1, (_Mask >> 4) & 1,
                (_Mask >> 3) & 1, (_Mask >> 2) & 1, (_Mask >> 1) & 1, _Mask & 1);
            return _IntrinBitcast<_VectorType_>(_mm256_cmp_ps(_Broadcasted, _mm256_set1_ps(1), 0));
        }
        else if constexpr (_Bits == 16) {
            const auto _BroadcastedHigh = _mm_set_epi16((_Mask >> 15) & 1, (_Mask >> 14) & 1, (_Mask >> 13) & 1, (_Mask >> 12) & 1,
                (_Mask >> 11) & 1, (_Mask >> 10) & 1, (_Mask >> 9) & 1, (_Mask >> 8) & 1);

            const auto _BroadcastedLow = _mm_set_epi16((_Mask >> 7) & 1, (_Mask >> 6) & 1, (_Mask >> 5) & 1, (_Mask >> 4) & 1,
                (_Mask >> 3) & 1, (_Mask >> 2) & 1, (_Mask >> 1) & 1, _Mask & 1);

            const auto _Comparand = _mm_set1_epi16(1);

            const auto _Low     = _mm_cmpeq_epi16(_BroadcastedLow, _Comparand);
            const auto _High    = _mm_cmpeq_epi16(_BroadcastedHigh, _Comparand);

            auto _Result = _IntrinBitcast<__m256i>(_Low);
            _Result = _mm256_insertf128_si256(_Result, _High, 1);

            return _IntrinBitcast<_VectorType_>(_Result);
        }
        else if constexpr (_Bits == 32) {
            const auto _Select      = _mm_set1_epi64x(0x8040201008040201ull);
            const auto _Shuffle     = _mm_set_epi64x(0x0101010101010101ll, 0);

            const auto _ShuffledLow     = _mm_shuffle_epi8(_mm_cvtsi32_si128(_Mask & 0xFFFF), _Shuffle);
            const auto _ShuffledHigh    = _mm_shuffle_epi8(_mm_cvtsi32_si128((_Mask >> 16) & 0xFFFF), _Shuffle);

            const auto _Low = _mm_cmpeq_epi8(_mm_and_si128(_ShuffledLow, _Select), _Select);
            const auto _High = _mm_cmpeq_epi8(_mm_and_si128(_ShuffledHigh, _Select), _Select);

            auto _Result = _IntrinBitcast<__m256i>(_Low);
            _Result = _mm256_insertf128_si256(_Result, _High, 1);

            return _IntrinBitcast<_VectorType_>(_Result);
        }
    }
};

template <>
class _SimdConvertImplementation<arch::CpuFeature::AVX2, ymm256> :
    public _SimdConvertImplementation<arch::CpuFeature::AVX, ymm256>
{
    static constexpr auto _Generation   = arch::CpuFeature::AVX2;
    using _RegisterPolicy               = ymm256;

    template <typename _DesiredType_>
    using _Simd_mask_type = type_traits::__deduce_simd_mask_type<_Generation, _DesiredType_, _RegisterPolicy>;
public:
    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline uint32 _ToMask(_VectorType_ _Vector) noexcept {
        if      constexpr (sizeof(_DesiredType_) == 8) {
            return _mm256_movemask_pd(_IntrinBitcast<__m256d>(_Vector));
        }
        else if constexpr (sizeof(_DesiredType_) == 4) {
            return _mm256_movemask_ps(_IntrinBitcast<__m256>(_Vector));
        }
        else if constexpr (sizeof(_DesiredType_) == 2) {
            const auto _Pack        = _mm256_packs_epi16(_IntrinBitcast<__m256i>(_Vector), _mm256_setzero_si256());
            const auto _Shuffled    = _mm256_permute4x64_epi64(_Pack, 0xD8);

            return _mm256_movemask_epi8(_Shuffled);
        }
        else if constexpr (sizeof(_DesiredType_) == 1) {
            return _mm256_movemask_epi8(_IntrinBitcast<__m256i>(_Vector));
        }
    }
    
    template <
        typename _VectorType_,
        typename _DesiredType_>
    static simd_stl_always_inline _VectorType_ _ToVector(_Simd_mask_type<_DesiredType_> _Mask) noexcept {
        constexpr auto _Bits = sizeof(_VectorType_) / sizeof(_DesiredType_);

        if constexpr (_Bits == 4) {
            const auto _Broadcasted = _mm256_set_pd((_Mask >> 3) & 1, (_Mask >> 2) & 1, (_Mask >> 1) & 1, _Mask & 1);
            return _IntrinBitcast<_VectorType_>(_mm256_cmp_pd(_Broadcasted, _mm256_set1_pd(1), 0));
        }
        else if constexpr (_Bits == 8) {
            const auto _Broadcasted = _mm256_set_ps((_Mask >> 7) & 1, (_Mask >> 6) & 1, (_Mask >> 5) & 1, (_Mask >> 4) & 1,
                (_Mask >> 3) & 1, (_Mask >> 2) & 1, (_Mask >> 1) & 1, _Mask & 1);
            return _IntrinBitcast<_VectorType_>(_mm256_cmp_ps(_Broadcasted, _mm256_set1_ps(1), 0));
        }
        else if constexpr (_Bits == 16) {
            const auto _Broadcasted = _mm256_set_epi16((_Mask >> 15) & 1, (_Mask >> 14) & 1, (_Mask >> 13) & 1, (_Mask >> 12) & 1,
                (_Mask >> 11) & 1, (_Mask >> 10) & 1, (_Mask >> 9) & 1, (_Mask >> 8) & 1, (_Mask >> 7) & 1, (_Mask >> 6) & 1,
                (_Mask >> 5) & 1, (_Mask >> 4) & 1, (_Mask >> 3) & 1, (_Mask >> 2) & 1, (_Mask >> 1) & 1, _Mask & 1);

            return _IntrinBitcast<_VectorType_>(_mm256_cmpeq_epi16(_Broadcasted, _mm256_set1_epi16(1)));
        }
        else if constexpr (_Bits == 32) {
            const auto _VectorMask = _mm256_setr_epi32(_Mask & 0xFFFF, 0, 0, 0, (_Mask >> 16) & 0xFFFF, 0, 0, 0);

            const auto _Select      = _mm256_set1_epi64x(0x8040201008040201ull);
            const auto _Shuffled    = _mm256_shuffle_epi8(_VectorMask,
                _mm256_set_epi64x(0x0101010101010101ll, 0, 0x0101010101010101ll, 0));

            return _IntrinBitcast<_VectorType_>(_mm256_cmpeq_epi8(_mm256_and_si256(_Shuffled, _Select), _Select));
        }
    }
};

#pragma endregion

#pragma region Avx512 Simd convert

template <>
class _SimdConvertImplementation<arch::CpuFeature::AVX512F, zmm512>
{
    static constexpr auto _Generation = arch::CpuFeature::AVX512F;
    using _RegisterPolicy = zmm512;

    template <typename _DesiredType_>
    using _Simd_mask_type = type_traits::__deduce_simd_mask_type<_Generation, _DesiredType_, _RegisterPolicy>;

    template <
        int32 _First_,
        int32 _Second_>
    static constexpr int32 _Max() noexcept {
        return (_First_ > _Second_) ? _First_ : _Second_;
    }
public:
    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _Simd_mask_type<_DesiredType_> _ToMask(_VectorType_ _Vector) noexcept {
        using _MaskType = _Simd_mask_type<_DesiredType_>;

        constexpr auto _Bits = sizeof(_VectorType_) / sizeof(_DesiredType_);

        if constexpr (_Bits == 16) {
            return static_cast<_MaskType>(_mm512_cmp_epi32_mask(
                _IntrinBitcast<__m512i>(_Vector), _mm512_setzero_si512(), _MM_CMPINT_LT));
        }
        else if constexpr (_Bits == 8) {
            return static_cast<_MaskType>(_mm512_cmp_epi64_mask(
                _IntrinBitcast<__m512i>(_Vector), _mm512_setzero_si512(), _MM_CMPINT_LT));
        }
        else {
            constexpr auto _YmmBits = _Bits >> 1;

            const auto _Low = _SimdToMask<arch::CpuFeature::AVX2, ymm256, _DesiredType_>(_IntrinBitcast<__m256d>(_Vector));
            const auto _High = _SimdToMask<arch::CpuFeature::AVX2, ymm256, _DesiredType_>(
                _mm512_extractf64x4_pd(_IntrinBitcast<__m512d>(_Vector), 1));

            return ((static_cast<_MaskType>(_High) << _YmmBits) | static_cast<_MaskType>(_Low));
        }
    }

    template <
        typename _VectorType_,
        typename _DesiredType_>
    static simd_stl_always_inline _VectorType_ _ToVector(_Simd_mask_type<_DesiredType_> _Mask) noexcept {
        using _MaskType = _Simd_mask_type<_DesiredType_>;

        constexpr auto _Bits = sizeof(_VectorType_) / sizeof(_DesiredType_);

        if constexpr (_Bits == 16) {
            return _IntrinBitcast<_VectorType_>(_mm512_maskz_mov_epi32(_Mask, _mm512_set1_epi32(-1)));
        }
        if constexpr (_Bits == 8) {
            return _IntrinBitcast<_VectorType_>(_mm512_maskz_mov_epi64(_Mask, _mm512_set1_epi64(-1)));
        }
        else {
            using _HalfType = IntegerForSize<_Max<(sizeof(_MaskType) >> 1), 1>()>::Unsigned;

            constexpr auto _Maximum = math::MaximumIntegralLimit<_HalfType>();
            constexpr auto _Shift = (sizeof(_MaskType) << 2);

            const auto _Low = _SimdToVector<arch::CpuFeature::AVX2, ymm256, __m256i, _DesiredType_>(_Mask & _Maximum);
            const auto _High = _SimdToVector<arch::CpuFeature::AVX2, ymm256, __m256i, _DesiredType_>((_Mask >> _Shift));

            return _IntrinBitcast<_VectorType_>(_mm512_inserti64x4(_IntrinBitcast<__m512i>(_Low), _High, 1));
        }
    }
};

template <>
class _SimdConvertImplementation<arch::CpuFeature::AVX512BW, zmm512>:
    public _SimdConvertImplementation<arch::CpuFeature::AVX512F, zmm512>
{
    static constexpr auto _Generation = arch::CpuFeature::AVX512BW;
    using _RegisterPolicy = zmm512;

    template <typename _DesiredType_>
    using _Simd_mask_type = type_traits::__deduce_simd_mask_type<_Generation, _DesiredType_, _RegisterPolicy>;
public:
    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _Simd_mask_type<_DesiredType_> _ToMask(_VectorType_ _Vector) noexcept {
        using _MaskType = _Simd_mask_type<_DesiredType_>;

        constexpr auto _Bits = sizeof(_VectorType_) / sizeof(_DesiredType_);


        if constexpr (_Bits == 64)
            return static_cast<_MaskType>(_mm512_movepi8_mask(_IntrinBitcast<__m512i>(_Vector))); 

        else if constexpr (_Bits == 32)
            return static_cast<_MaskType>(_mm512_movepi16_mask(_IntrinBitcast<__m512i>(_Vector)));

        else
            return _SimdToMask<arch::CpuFeature::AVX512F, zmm512, _DesiredType_>(_Vector);
    }

    template <
        typename _VectorType_,
        typename _DesiredType_>
    static simd_stl_always_inline _VectorType_ _ToVector(_Simd_mask_type<_DesiredType_> _Mask) noexcept {
        using _MaskType = _Simd_mask_type<_DesiredType_>;

        constexpr auto _Bits = sizeof(_VectorType_) / sizeof(_DesiredType_);

        if constexpr (_Bits == 64)
            return _IntrinBitcast<_VectorType_>(_mm512_movm_epi8(_Mask));

        else if constexpr (_Bits == 32)
            return _IntrinBitcast<_VectorType_>(_mm512_movm_epi16(_Mask));

        else if constexpr (_Bits == 16)
            return _IntrinBitcast<_VectorType_>(_mm512_maskz_mov_epi32(_Mask, _mm512_set1_epi32(-1)));

        else if constexpr (_Bits == 8)
            return _IntrinBitcast<_VectorType_>(_mm512_maskz_mov_epi64(_Mask, _mm512_set1_epi64(-1)));
    }
};

template <>
class _SimdConvertImplementation<arch::CpuFeature::AVX512DQ, zmm512>:
    public _SimdConvertImplementation<arch::CpuFeature::AVX512BW, zmm512>
{
    static constexpr auto _Generation = arch::CpuFeature::AVX512DQ;
    using _RegisterPolicy = zmm512;

    template <typename _DesiredType_>
    using _Simd_mask_type = type_traits::__deduce_simd_mask_type<_Generation, _DesiredType_, _RegisterPolicy>;
public:
    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _Simd_mask_type<_DesiredType_> _ToMask(_VectorType_ _Vector) noexcept {
        using _MaskType = _Simd_mask_type<_DesiredType_>;

        constexpr auto _Bits = sizeof(_VectorType_) / sizeof(_DesiredType_);

        if constexpr (_Bits == 64)
            return static_cast<_MaskType>(_mm512_movepi8_mask(_IntrinBitcast<__m512i>(_Vector)));

        else if constexpr (_Bits == 32)
            return static_cast<_MaskType>(_mm512_movepi16_mask(_IntrinBitcast<__m512i>(_Vector)));

        else if constexpr (_Bits == 16)
            return static_cast<_MaskType>(_mm512_movepi32_mask(_IntrinBitcast<__m512i>(_Vector)));

        else if constexpr (_Bits == 8)
            return static_cast<_MaskType>(_mm512_movepi64_mask(_IntrinBitcast<__m512i>(_Vector)));
    }

    template <
        typename _VectorType_,
        typename _DesiredType_>
    static simd_stl_always_inline _VectorType_ _ToVector(_Simd_mask_type<_DesiredType_> _Mask) noexcept {
        using _MaskType = _Simd_mask_type<_DesiredType_>;
        constexpr auto _Bits = sizeof(_VectorType_) / sizeof(_DesiredType_);


        if constexpr (_Bits == 64)
            return _IntrinBitcast<_VectorType_>(_mm512_movm_epi8(_Mask));

        else if constexpr (_Bits == 32)
            return _IntrinBitcast<_VectorType_>(_mm512_movm_epi16(_Mask));

        else if constexpr (_Bits == 16)
            return _IntrinBitcast<_VectorType_>(_mm512_movm_epi32(_Mask));

        else if constexpr (_Bits == 8)
            return _IntrinBitcast<_VectorType_>(_mm512_movm_epi64(_Mask));
    }
};

template <>
class _SimdConvertImplementation<arch::CpuFeature::AVX512VL, zmm512> :
    public _SimdConvertImplementation<arch::CpuFeature::AVX512DQ, zmm512>
{};

#pragma endregion

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_,
    typename            _DesiredType_,
    typename            _VectorType_>
simd_stl_always_inline auto _SimdToMask(_VectorType_ _Vector) noexcept {
    _VerifyRegisterPolicy(_SimdGeneration_, _RegisterPolicy_)
    return _SimdConvertImplementation<_SimdGeneration_, _RegisterPolicy_>::template _ToMask<_DesiredType_>(_Vector);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_,
    typename            _VectorType_,
    typename            _DesiredType_,
    typename            _MaskType_>
simd_stl_always_inline _VectorType_ _SimdToVector(_MaskType_ _Mask) noexcept {
    _VerifyRegisterPolicy(_SimdGeneration_, _RegisterPolicy_)
    return _SimdConvertImplementation<_SimdGeneration_, _RegisterPolicy_>::template _ToVector<_VectorType_, _DesiredType_>(_Mask);
}

__SIMD_STL_NUMERIC_NAMESPACE_END
