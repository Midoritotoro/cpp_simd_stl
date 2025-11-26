#pragma once 

#include <src/simd_stl/numeric/SimdMemoryAccess.h>
#include <simd_stl/memory/pointerToIntegral.h>


__SIMD_STL_NUMERIC_NAMESPACE_BEGIN

#define _Simd_stl_case_insert_(_Postfix, _Index, _Vector, _Value) \
    case _Index: _Vector = _IntrinBitcast<_VectorType_>(SIMD_STL_PP_CAT(_mm_insert_, _Postfix)(_IntrinBitcast<__m128i>(_Vector), _Value, _Index));

#define _Simd_stl_case_extract_(_Postfix, _Index, _Vector, _DesiredType) \
    case _Index: return static_cast<_DesiredType>(SIMD_STL_PP_CAT(_mm_extract_, _Postfix)(_IntrinBitcast<__m128i>(_Vector), _Index));


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
                _Simd_stl_case_insert_(epi16, 0, _Vector, _Value)
                _Simd_stl_case_insert_(epi16, 1, _Vector, _Value)
                _Simd_stl_case_insert_(epi16, 2, _Vector, _Value)
                _Simd_stl_case_insert_(epi16, 3, _Vector, _Value)
                _Simd_stl_case_insert_(epi16, 4, _Vector, _Value)
                _Simd_stl_case_insert_(epi16, 5, _Vector, _Value)
                _Simd_stl_case_insert_(epi16, 6, _Vector, _Value)
                _Simd_stl_case_insert_(epi16, 7, _Vector, _Value)
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
        else if constexpr (_Is_ps_v<_DesiredType_>) {
            const int32 _MaskArray[8] = { 0,0,0,0,-1,0,0,0 };

            const auto _Broadcasted = _mm_set1_ps(_Value);
            const auto _InsertMask = _SimdLoadUnaligned<_Generation, _RegisterPolicy, __m128>(_MaskArray + 4 - (_Position & 3)); // FFFFFFFF at index position

            _Vector = _IntrinBitcast<_VectorType_>(_mm_or_ps(
                _mm_and_ps(_InsertMask, _Broadcasted),
                _mm_andnot_ps(_InsertMask, _IntrinBitcast<__m128>(_Vector))));
        }
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _DesiredType_ _Extract(
        _VectorType_    _Vector,
        const uint8     _Where) noexcept
    {
        if constexpr (_Is_epi64_v<_DesiredType_> || _Is_epu64_v<_DesiredType_> || _Is_pd_v<_DesiredType_>) {
            if (_Where == 0) {
#if defined(simd_stl_processor_x86_64)
                return static_cast<_DesiredType_>(_mm_cvtsi128_si64(_IntrinBitcast<__m128i>(_Vector)));
#else
                const auto _HighDword = _mm_cvtsi128_si32(_IntrinBitcast<__m128i>(_Vector));
                const auto _LowDword = _mm_cvtsi128_si32(_mm_shuffle_epi32(_IntrinBitcast<__m128i>(_Vector), 0x55));

                return (static_cast<_DesiredType_>(_HighDword) << 32) | static_cast<_DesiredType_>(_LowDword);
#endif // defined(simd_stl_processor_x86_64)
            }

#if !defined(simd_stl_processor_x86_64)
            return static_cast<_DesiredType_>(_mm_cvtsi128_si64(
                _mm_shuffle_epi32(_IntrinBitcast<__m128i>(_Vector), 0xEE)));
#else
            const auto _HighDword = _mm_cvtsi128_si32(_mm_shuffle_epi32(_IntrinBitcast<__m128i>(_Vector), 0xEE));
            const auto _LowDword = _mm_cvtsi128_si32(_mm_shuffle_epi32(_IntrinBitcast<__m128i>(_Vector), 0xFF));

            return (static_cast<_DesiredType_>(_HighDword) << 32) | static_cast<_DesiredType_>(_LowDword);
#endif // defined(simd_stl_processor_x86_64)
        }
        else if constexpr (_Is_epi32_v<_DesiredType_> || _Is_epu32_v<_DesiredType_> || _Is_ps_v<_DesiredType_>) {
            constexpr std::array<int32, 4> _Shuffle = { 0, 0x55, 0xEE, 0xFF };

            if (_Where == 0)
                return static_cast<_DesiredType_>(_mm_cvtsi128_si32(_IntrinBitcast<__m128i>(_Vector)));
            
            return static_cast<_DesiredType_>(_mm_cvtsi128_si32(_mm_shuffle_epi32(_IntrinBitcast<__m128i>(_Vector), _Shuffle[_Where])));
        }
        else if constexpr (_Is_epi16_v<_DesiredType_> || _Is_epu16_v<_DesiredType_>) {
            switch (_Where) {
                _Simd_stl_case_extract_(epi16, 0, _Vector, _DesiredType_)
                _Simd_stl_case_extract_(epi16, 1, _Vector, _DesiredType_)
                _Simd_stl_case_extract_(epi16, 2, _Vector, _DesiredType_)
                _Simd_stl_case_extract_(epi16, 3, _Vector, _DesiredType_)
                _Simd_stl_case_extract_(epi16, 4, _Vector, _DesiredType_)
                _Simd_stl_case_extract_(epi16, 5, _Vector, _DesiredType_)
                _Simd_stl_case_extract_(epi16, 6, _Vector, _DesiredType_)
                _Simd_stl_case_extract_(epi16, 7, _Vector, _DesiredType_)
            }
        }
        else if constexpr (_Is_epi8_v<_DesiredType_> || _Is_epu8_v<_DesiredType_>) {
            if (_Where <= 3)
                return _Extract<int32, _VectorType_>(_Vector, _Where >> 2) >> (_Where << 3);
            else
                return (_Where & 1)
                    ? _Extract<int16, _VectorType_>(_Vector, _Where >> 1) >> 8
                    : _Extract<int16, _VectorType_>(_Vector, _Where >> 1);
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
{
public:
    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline void _Insert(
        _VectorType_&       _Vector,
        const uint8         _Position,
        const _DesiredType_ _Value) noexcept
    {
        if constexpr (_Is_epi64_v<_DesiredType_> || _Is_epu64_v<_DesiredType_> || _Is_pd_v<_DesiredType_>) {
            switch (_Position) {
                _Simd_stl_case_insert_(epi64, 0, _Vector, _Value)
                _Simd_stl_case_insert_(epi64, 1, _Vector, _Value)
            }
        }
        else if constexpr (_Is_epi32_v<_DesiredType_> || _Is_epu32_v<_DesiredType_>) {
            switch (_Position) {
                _Simd_stl_case_insert_(epi32, 0, _Vector, _Value)
                _Simd_stl_case_insert_(epi32, 1, _Vector, _Value)
                _Simd_stl_case_insert_(epi32, 2, _Vector, _Value)
                _Simd_stl_case_insert_(epi32, 3, _Vector, _Value)
            }
        }
        else if constexpr (_Is_epi16_v<_DesiredType_> || _Is_epu16_v<_DesiredType_>) {
            switch (_Position) {
                _Simd_stl_case_insert_(epi16, 0, _Vector, _Value)
                _Simd_stl_case_insert_(epi16, 1, _Vector, _Value)
                _Simd_stl_case_insert_(epi16, 2, _Vector, _Value)
                _Simd_stl_case_insert_(epi16, 3, _Vector, _Value)
                _Simd_stl_case_insert_(epi16, 4, _Vector, _Value)
                _Simd_stl_case_insert_(epi16, 5, _Vector, _Value)
                _Simd_stl_case_insert_(epi16, 6, _Vector, _Value)
                _Simd_stl_case_insert_(epi16, 7, _Vector, _Value)
            }
        }
        else if constexpr (_Is_epi8_v<_DesiredType_> || _Is_epu8_v<_DesiredType_>) {
            switch (_Position) {
                _Simd_stl_case_insert_(epi8, 0, _Vector, _Value)
                _Simd_stl_case_insert_(epi8, 1, _Vector, _Value)
                _Simd_stl_case_insert_(epi8, 2, _Vector, _Value)
                _Simd_stl_case_insert_(epi8, 3, _Vector, _Value)
                _Simd_stl_case_insert_(epi8, 4, _Vector, _Value)
                _Simd_stl_case_insert_(epi8, 5, _Vector, _Value)
                _Simd_stl_case_insert_(epi8, 6, _Vector, _Value)
                _Simd_stl_case_insert_(epi8, 7, _Vector, _Value)
                _Simd_stl_case_insert_(epi8, 8, _Vector, _Value)
                _Simd_stl_case_insert_(epi8, 9, _Vector, _Value)
                _Simd_stl_case_insert_(epi8, 10, _Vector, _Value)
                _Simd_stl_case_insert_(epi8, 11, _Vector, _Value)
                _Simd_stl_case_insert_(epi8, 12, _Vector, _Value)
                _Simd_stl_case_insert_(epi8, 13, _Vector, _Value)
                _Simd_stl_case_insert_(epi8, 14, _Vector, _Value)
                _Simd_stl_case_insert_(epi8, 15, _Vector, _Value)
            }
        }
        else if constexpr (_Is_ps_v<_DesiredType_>) {
            switch (_Position) {
                _Simd_stl_case_insert_(ps, 0, _Vector, _Value)
                _Simd_stl_case_insert_(ps, 1, _Vector, _Value)
                _Simd_stl_case_insert_(ps, 2, _Vector, _Value)
                _Simd_stl_case_insert_(ps, 3, _Vector, _Value)
            }
        }
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _DesiredType_ _Extract(
        _VectorType_    _Vector,
        const uint8     _Where) noexcept
    {
        if constexpr (_Is_epi64_v<_DesiredType_> || _Is_epu64_v<_DesiredType_> || _Is_pd_v<_DesiredType_>) {
            switch (_Where) {
                _Simd_stl_case_extract_(epi64, 0, _Vector, _DesiredType_)
                _Simd_stl_case_extract_(epi64, 1, _Vector, _DesiredType_)
            }
        }
        else if constexpr (_Is_epi32_v<_DesiredType_> || _Is_epu32_v<_DesiredType_>) {
            switch (_Where) {
                _Simd_stl_case_extract_(epi32, 0, _Vector, _DesiredType_)
                _Simd_stl_case_extract_(epi32, 1, _Vector, _DesiredType_)
                _Simd_stl_case_extract_(epi32, 2, _Vector, _DesiredType_)
                _Simd_stl_case_extract_(epi32, 3, _Vector, _DesiredType_)
            }
        }
        else if constexpr (_Is_ps_v<_DesiredType_>) {
            switch (_Where) {
                _Simd_stl_case_extract_(ps, 0, _Vector, _DesiredType_)
                _Simd_stl_case_extract_(ps, 1, _Vector, _DesiredType_)
                _Simd_stl_case_extract_(ps, 2, _Vector, _DesiredType_)
                _Simd_stl_case_extract_(ps, 3, _Vector, _DesiredType_)
            }
        }
        else if constexpr (_Is_epi16_v<_DesiredType_> || _Is_epu16_v<_DesiredType_>) {
            switch (_Where) {
                _Simd_stl_case_extract_(epi16, 0, _Vector, _DesiredType_)
                _Simd_stl_case_extract_(epi16, 1, _Vector, _DesiredType_)
                _Simd_stl_case_extract_(epi16, 2, _Vector, _DesiredType_)
                _Simd_stl_case_extract_(epi16, 3, _Vector, _DesiredType_)
                _Simd_stl_case_extract_(epi16, 4, _Vector, _DesiredType_)
                _Simd_stl_case_extract_(epi16, 5, _Vector, _DesiredType_)
                _Simd_stl_case_extract_(epi16, 6, _Vector, _DesiredType_)
                _Simd_stl_case_extract_(epi16, 7, _Vector, _DesiredType_)
            }
        }
        else if constexpr (_Is_epi8_v<_DesiredType_> || _Is_epu8_v<_DesiredType_>) {
           switch (_Where) {
                _Simd_stl_case_extract_(epi8, 0, _Vector, _DesiredType_)
                _Simd_stl_case_extract_(epi8, 1, _Vector, _DesiredType_)
                _Simd_stl_case_extract_(epi8, 2, _Vector, _DesiredType_)
                _Simd_stl_case_extract_(epi8, 3, _Vector, _DesiredType_)
                _Simd_stl_case_extract_(epi8, 4, _Vector, _DesiredType_)
                _Simd_stl_case_extract_(epi8, 5, _Vector, _DesiredType_)
                _Simd_stl_case_extract_(epi8, 6, _Vector, _DesiredType_)
                _Simd_stl_case_extract_(epi8, 7, _Vector, _DesiredType_)
                _Simd_stl_case_extract_(epi8, 8, _Vector, _DesiredType_)
                _Simd_stl_case_extract_(epi8, 9, _Vector, _DesiredType_)
                _Simd_stl_case_extract_(epi8, 10, _Vector, _DesiredType_)
                _Simd_stl_case_extract_(epi8, 11, _Vector, _DesiredType_)
                _Simd_stl_case_extract_(epi8, 12, _Vector, _DesiredType_)
                _Simd_stl_case_extract_(epi8, 13, _Vector, _DesiredType_)
                _Simd_stl_case_extract_(epi8, 14, _Vector, _DesiredType_)
                _Simd_stl_case_extract_(epi8, 15, _Vector, _DesiredType_)
            }
        }
    }
};

template <class _RegisterPolicy_>
class _SimdElementAccess<arch::CpuFeature::SSE42, _RegisterPolicy_>:
    public _SimdElementAccess<arch::CpuFeature::SSE41, _RegisterPolicy_>
{};


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
