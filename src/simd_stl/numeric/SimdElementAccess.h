#pragma once 

#include <src/simd_stl/numeric/SimdMemoryAccess.h>
#include <simd_stl/memory/pointerToIntegral.h>


__SIMD_STL_NUMERIC_NAMESPACE_BEGIN

#define _Simd_stl_case_insert_epi16(_Index, _Vector, _Value) \
    case _Index: vector = _IntrinBitcast<_VectorType_>(_mm_insert_epi16(_IntrinBitcast<__m128i>(_Vector), _Value, _Index));

#define _Simd_stl_case_extract_epi16(_Index, _Vector, _DesiredType) \
    case _Index: return static_cast<_DesiredType>(_mm_extract_epi16(_IntrinBitcast<__m128i>(_Vector), _Index));


template <
    arch::CpuFeature    _SimdGeneration_, 
    class               _RegisterPolicy_>
class _SimdElementAccess;

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
        else if constexpr (_Is_epi32_v<_DesiredType_> || _Is_epu32_v<_DesiredType_>) {
            const auto _Broadcasted = _SimdBroadcast<_Generation, _RegisterPolicy, __m128i, _DesiredType_>(_Value);
            const int32 _MaskArray[8] = { 0, 0, 0, 0, -1, 0, 0, 0 };

            const auto _InsertMask = _SimdLoadUnaligned<_Generation, _RegisterPolicy, __m128i>(_MaskArray + 4 - (_Position & 3)); // FFFFFFFF at index position

            _Vector = _IntrinBitcast<_VectorType_>(_mm_or_si128(_mm_and_si128(_InsertMask, _Broadcasted),
                    _mm_andnot_si128(_InsertMask, _IntrinBitcast<__m128i>(_Vector))));
        }
        else if constexpr (_Is_epi16_v<_DesiredType_> || _Is_epu16_v<_DesiredType_>) {
            switch (_Position) {
                _Simd_stl_case_insert_epi16(0, _Vector, _Value)
                _Simd_stl_case_insert_epi16(1, _Vector, _Value)
                _Simd_stl_case_insert_epi16(2, _Vector, _Value)
                _Simd_stl_case_insert_epi16(3, _Vector, _Value)
                _Simd_stl_case_insert_epi16(4, _Vector, _Value)
                _Simd_stl_case_insert_epi16(5, _Vector, _Value)
                _Simd_stl_case_insert_epi16(6, _Vector, _Value)
                _Simd_stl_case_insert_epi16(7, _Vector, _Value)
            }
        }
        else if constexpr (_Is_epi8_v<_DesiredType_> || _Is_epu8_v<_DesiredType_>) {
            const int8 _MaskArray[32] = { 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 };
            const auto _Broadcasted = _SimdBroadcast<_Generation, _RegisterPolicy, __m128i, _DesiredType_>(_Value);

            const auto _InsertMask = _SimdLoadUnaligned<_Generation, _RegisterPolicy, __m128i>(_MaskArray + 16 - (_Position & 0x0F)); // FF at index position

            _Vector = _IntrinBitcast<_VectorType_>(_mm_or_si128(_mm_and_si128(_InsertMask, _Broadcasted),
                _mm_andnot_si128(_InsertMask, _IntrinBitcast<__m128i>(_Vector))));
        }
        else if constexpr (_Is_pd_v<_DesiredType_>) {
            const auto _Broadcasted = _mm_set_sd(_Value);

            _Vector = (_Position == 0)
                ? _mm_shuffle_pd(_Broadcasted, _IntrinBitcast<__m128d>(_Vector), 2)
                : _mm_shuffle_pd(_IntrinBitcast<__m128d>(_Vector), _Broadcasted, 0);
        }
        else if constexpr (is_ps_v<_DesiredType_>) {
            const int32 _MaskArray[8] = { 0,0,0,0,-1,0,0,0 };

            const auto _Broadcasted = _mm_set1_ps(_Value);
            const auto _InsertMask = _SimdLoadUnaligned<_Generation, _RegisterPolicy, __m128>(_MaskArray + 4 - (_Position & 3)); // FFFFFFFF at index position

            _Vector = _IntrinBitcast<_VectorType_>(_mm_or_ps(
                _mm_and_ps(_InsertMask, _Broadcasted),
                _mm_andnot_ps(_InsertMask, _IntrinBitcast<__m128>(vector))));
        }
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _DesiredType_ _Extract(
        _VectorType_    _Vector,
        const uint8     _Where) noexcept
    {
        if constexpr (_Is_epi16_v<_DesiredType_> || _Is_epu16_v<_DesiredType_>) {
            switch (_Where) {
                _Simd_stl_case_extract_epi16(0, _Vector, _DesiredType_)
                _Simd_stl_case_extract_epi16(1, _Vector, _DesiredType_)
                _Simd_stl_case_extract_epi16(2, _Vector, _DesiredType_)
                _Simd_stl_case_extract_epi16(3, _Vector, _DesiredType_)
                _Simd_stl_case_extract_epi16(4, _Vector, _DesiredType_)
                _Simd_stl_case_extract_epi16(5, _Vector, _DesiredType_)
                _Simd_stl_case_extract_epi16(6, _Vector, _DesiredType_)
                _Simd_stl_case_extract_epi16(7, _Vector, _DesiredType_)
            }
        }
        else {
            _DesiredType_ array[sizeof(_VectorType_) / sizeof(_DesiredType_)];
            _SimdStoreUnaligned<_Generation, _RegisterPolicy, _DesiredType_>(array, _Vector);

            return static_cast<_DesiredType_>(array[_Where]);
        }
    }
};

template <class _RegisterPolicy_>
class _SimdElementAccess<arch::CpuFeature::SSE3, _RegisterPolicy_>:
    public _SimdElementAccess<arch::CpuFeature::SSE2, _RegisterPolicy_>
{};

template <class _RegisterPolicy_>
class _SimdElementAccess<arch::CpuFeature::SSSE3, _RegisterPolicy_> :
    public _SimdElementAccess<arch::CpuFeature::SSE3, _RegisterPolicy_>
{};

template <class _RegisterPolicy_>
class _SimdElementAccess<arch::CpuFeature::SSE41, _RegisterPolicy_> :
    public _SimdElementAccess<arch::CpuFeature::SSSE3, _RegisterPolicy_>
{};

template <class _RegisterPolicy_>
class _SimdElementAccess<arch::CpuFeature::SSE42, _RegisterPolicy_>:
    public _SimdElementAccess<arch::CpuFeature::SSE41, _RegisterPolicy_>
{};

__SIMD_STL_NUMERIC_NAMESPACE_END
