#pragma once 

#include <src/simd_stl/numeric/SimdElementWise.h>
#include <simd_stl/numeric/BasicSimdMask.h>

#include <src/simd_stl/algorithm/AdvanceBytes.h>
#include <src/simd_stl/numeric/SimdBroadcast.h>

__SIMD_STL_NUMERIC_NAMESPACE_BEGIN

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_>
class _SimdMemoryAccess;

template <
    arch::CpuFeature	_SimdGeneration_,
    class				_RegisterPolicy_,
    typename            _DesiredType_,
    typename            _VectorType_>
simd_stl_always_inline _DesiredType_* _SimdCompressStoreUnaligned(
    _DesiredType_* _Where,
    type_traits::__deduce_simd_mask_type<_SimdGeneration_,
    _DesiredType_, _RegisterPolicy_>    _Mask,
    _VectorType_                        _Vector) noexcept;

template <
    arch::CpuFeature	_SimdGeneration_,
    class				_RegisterPolicy_,
    typename            _DesiredType_,
    typename            _VectorType_>
simd_stl_always_inline _DesiredType_* _SimdCompressStoreAligned(
    _DesiredType_*                          _Where,
    type_traits::__deduce_simd_mask_type<_SimdGeneration_,
        _DesiredType_, _RegisterPolicy_>    _Mask,
    const _VectorType_                      _Vector) noexcept;

#pragma region Sse2-Sse4.2 memory access 

template <>
class _SimdMemoryAccess<arch::CpuFeature::SSE2, numeric::xmm128> {
    static constexpr auto _Generation   = arch::CpuFeature::SSE2;
    using _RegisterPolicy               = numeric::xmm128;

    template <class _DesiredType_>
    using _Simd_mask_type = type_traits::__deduce_simd_mask_type<_Generation, _DesiredType_, _RegisterPolicy>;
public:
    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _LoadUpperHalf(const void* _Where) noexcept {
        return _IntrinBitcast<_VectorType_>(_mm_loadh_pd(
            _SimdBroadcastZeros<_Generation, _RegisterPolicy, __m128d>(),
            static_cast<const double*>(_Where)));
    }

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _LoadLowerHalf(const void* _Where) noexcept {
        return _IntrinBitcast<_VectorType_>(_mm_loadl_pd(
            _SimdBroadcastZeros<_Generation, _RegisterPolicy, __m128d>(),
            static_cast<const double*>(_Where)));
    }

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _NonTemporalLoad(const void* _Where) noexcept {
        return _LoadAligned<_VectorType_, void>(_Where);
    }

    template <typename _VectorType_>
    static simd_stl_always_inline void _NonTemporalStore(
        void*           _Where,
        _VectorType_    _Vector) noexcept 
    {
        _mm_stream_si128(static_cast<__m128i*>(_Where), _IntrinBitcast<__m128i>(_Vector));
    }

    static simd_stl_always_inline void _StreamingFence() noexcept {
        return _mm_sfence();
    }

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _LoadUnaligned(const void* _Where) noexcept {
        if      constexpr (std::is_same_v<_VectorType_, __m128i>)
            return _mm_loadu_si128(reinterpret_cast<const __m128i*>(_Where));

        else if constexpr (std::is_same_v<_VectorType_, __m128d>)
            return _mm_loadu_pd(reinterpret_cast<const double*>(_Where));

        else if constexpr (std::is_same_v<_VectorType_, __m128>)
            return _mm_loadu_ps(reinterpret_cast<const float*>(_Where));
    }

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _LoadAligned(const void* _Where) noexcept {
        if      constexpr (std::is_same_v<_VectorType_, __m128i>)
            return _mm_load_si128(reinterpret_cast<const __m128i*>(_Where));

        else if constexpr (std::is_same_v<_VectorType_, __m128d>)
            return _mm_load_pd(reinterpret_cast<const double*>(_Where));

        else if constexpr (std::is_same_v<_VectorType_, __m128>)
            return _mm_load_ps(reinterpret_cast<const float*>(_Where));
    }

    template <typename _VectorType_>
    static simd_stl_always_inline void _StoreUpperHalf(
        void*           _Where,
        _VectorType_    _Vector) noexcept
    {
        _mm_storeh_pd(reinterpret_cast<double*>(_Where), _IntrinBitcast<__m128d>(_Vector));
    }

    template <typename _VectorType_>
    static simd_stl_always_inline void _StoreLowerHalf(
        void*           _Where,
        _VectorType_    _Vector) noexcept
    {
        _mm_storel_epi64(reinterpret_cast<__m128i*>(_Where), _IntrinBitcast<__m128i>(_Vector));
    }

    template <typename _VectorType_>
    static simd_stl_always_inline void _StoreUnaligned(
        void*           _Where,
        _VectorType_    _Vector) noexcept
    {
        if      constexpr (std::is_same_v<_VectorType_, __m128i>)
            return _mm_storeu_si128(reinterpret_cast<__m128i*>(_Where), _Vector);

        else if constexpr (std::is_same_v<_VectorType_, __m128d>)
            return _mm_storeu_pd(reinterpret_cast<double*>(_Where), _Vector);

        else if constexpr (std::is_same_v<_VectorType_, __m128>)
            return _mm_storeu_ps(reinterpret_cast<float*>(_Where), _Vector);
    }

    template <typename _VectorType_>
    static simd_stl_always_inline void _StoreAligned(
        void*           _Where,
        _VectorType_    _Vector) noexcept
    {
        if      constexpr (std::is_same_v<_VectorType_, __m128i>)
            return _mm_store_si128(reinterpret_cast<__m128i*>(_Where), _Vector);

        else if constexpr (std::is_same_v<_VectorType_, __m128d>)
            return _mm_store_pd(reinterpret_cast<double*>(_Where), _Vector);
   
        else if constexpr (std::is_same_v<_VectorType_, __m128>)
            return _mm_store_ps(reinterpret_cast<float*>(_Where), _Vector);
    }


    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline void _MaskStoreUnaligned(
        _DesiredType_*                          _Where,
        const _Simd_mask_type<_DesiredType_>    _Mask,
        const _VectorType_                      _Vector) noexcept
    {
        _StoreUnaligned(_Where, _SimdBlend<_Generation, _RegisterPolicy, _DesiredType_>(
            _Vector, _LoadUnaligned<_VectorType_>(_Where), _Mask));
    }
    
    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline void _MaskStoreAligned(
        _DesiredType_*                          _Where,
        const _Simd_mask_type<_DesiredType_>    _Mask,
        const _VectorType_                      _Vector) noexcept
    {
        _StoreAligned(_Where, _SimdBlend<_Generation, _RegisterPolicy, _DesiredType_>(
            _Vector, _LoadAligned<_VectorType_>(_Where), _Mask));
    }

    template <
        typename _VectorType_,
        typename _DesiredType_>
    static simd_stl_always_inline _VectorType_ _MaskLoadUnaligned(
        const _DesiredType_*                    _Where,
        const _Simd_mask_type<_DesiredType_>    _Mask) noexcept
    {
        return _SimdBlend<_Generation, _RegisterPolicy, _DesiredType_>(
            _LoadUnaligned<_VectorType_>(_Where), 
            _SimdBroadcastZeros<_Generation, _RegisterPolicy, _VectorType_>(), _Mask);
    }

    template <
        typename _VectorType_,
        typename _DesiredType_>
    static simd_stl_always_inline _VectorType_ _MaskLoadAligned(
        const _DesiredType_*                    _Where,
        const _Simd_mask_type<_DesiredType_>    _Mask) noexcept
    {
        return _SimdBlend<_Generation, _RegisterPolicy, _DesiredType_>(
            _LoadAligned<_VectorType_>(_Where), 
            _SimdBroadcastZeros<_Generation, _RegisterPolicy, _VectorType_>(), _Mask);
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _DesiredType_* _CompressStoreLowerHalf(
        _DesiredType_*                  _Where,
        _Simd_mask_type<_DesiredType_>  _Mask,
        _VectorType_                    _Vector) noexcept
    {
        static_assert(sizeof(_DesiredType_) != 8);

        __m128i _Shuffle;

        if constexpr (sizeof(_DesiredType_) == 4)
            _Shuffle = _LoadLowerHalf<__m128i>(_Tables32BitSse._Shuffle[_Mask]);

        else if constexpr (sizeof(_DesiredType_) == 2)
            _Shuffle = _LoadLowerHalf<__m128i>(_Tables16BitSse._Shuffle[_Mask]);

        else if constexpr (sizeof(_DesiredType_) == 1)
            _Shuffle = _LoadLowerHalf<__m128i>(_Tables8BitSse._Shuffle[_Mask]);

        const auto _Destination = _mm_shuffle_epi8(_IntrinBitcast<__m128i>(_Vector), _Shuffle);
        _mm_storel_epi64(reinterpret_cast<__m128i*>(_Where), _Destination);

        if constexpr (sizeof(_DesiredType_) == 4)
            algorithm::AdvanceBytes(_Where, _Tables32BitSse._Size[_Mask]);

        else if constexpr (sizeof(_DesiredType_) == 2)
            algorithm::AdvanceBytes(_Where, _Tables16BitSse._Size[_Mask]);

        else if constexpr (sizeof(_DesiredType_) == 1)
            algorithm::AdvanceBytes(_Where, _Tables8BitSse._Size[_Mask]);

        return _Where;
    }
    
    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _DesiredType_* _CompressStoreUpperHalf(
        _DesiredType_*                  _Where,
        _Simd_mask_type<_DesiredType_>  _Mask,
        _VectorType_                    _Vector) noexcept
    {
        static_assert(sizeof(_DesiredType_) != 8);
        __m128i _Shuffle;

       if constexpr (sizeof(_DesiredType_) == 4)
           _Shuffle = _LoadUpperHalf<__m128i>(_Tables32BitSse._Shuffle[_Mask]);
        else if constexpr (sizeof(_DesiredType_) == 2)
           _Shuffle = _LoadUpperHalf<__m128i>(_Tables16BitSse._Shuffle[_Mask]);
        else if constexpr (sizeof(_DesiredType_) == 1)
           _Shuffle = _LoadUpperHalf<__m128i>(_Tables8BitSse._Shuffle[_Mask]);


        _mm_storeh_pd(reinterpret_cast<double*>(_Where), _IntrinBitcast<__m128d>(
            _mm_shuffle_epi8(_IntrinBitcast<__m128i>(_Vector), _Shuffle)));

        if constexpr (sizeof(_DesiredType_) == 4)
            algorithm::AdvanceBytes(_Where, _Tables32BitSse._Size[_Mask]);
        else if constexpr (sizeof(_DesiredType_) == 2)
            algorithm::AdvanceBytes(_Where, _Tables16BitSse._Size[_Mask]);
        else if constexpr (sizeof(_DesiredType_) == 1)
            algorithm::AdvanceBytes(_Where, _Tables8BitSse._Size[_Mask]);

        return _Where;
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _DesiredType_* _CompressStoreUnaligned(
        _DesiredType_*                      _Where,
        _Simd_mask_type<_DesiredType_>      _Mask,
        _VectorType_                        _Vector) noexcept
    {
        if      constexpr (sizeof(_DesiredType_) == 8) {
            _StoreUnaligned(_Where, _mm_shuffle_epi8(_IntrinBitcast<__m128i>(_Vector),
                _LoadUnaligned<__m128i>(_Tables64BitSse._Shuffle[_Mask])));

            algorithm::AdvanceBytes(_Where, _Tables64BitSse._Size[_Mask]);
        }
        else if constexpr (sizeof(_DesiredType_) == 4) {
            _StoreUnaligned(_Where, _mm_shuffle_epi8(_IntrinBitcast<__m128i>(_Vector),
                _LoadUnaligned<__m128i>(_Tables32BitSse._Shuffle[_Mask])));

            algorithm::AdvanceBytes(_Where, _Tables32BitSse._Size[_Mask]);
        }
        else if constexpr (sizeof(_DesiredType_) == 2) {
            _StoreUnaligned(_Where, _mm_shuffle_epi8(_IntrinBitcast<__m128i>(_Vector),
                _LoadUnaligned<__m128i>(_Tables16BitSse._Shuffle[_Mask])));

            algorithm::AdvanceBytes(_Where, _Tables16BitSse._Size[_Mask]);
        }
        else if constexpr (sizeof(_DesiredType_) == 1) {
            auto _Start = _Where;

            _Where = _CompressStoreLowerHalf(_Where, _Mask & 0xFF, _Vector);
            _Where = _CompressStoreLowerHalf(_Where, (_Mask >> 8) & 0xFF, _mm_movehl_ps(
                _IntrinBitcast<__m128>(_mm_slli_si128(_IntrinBitcast<__m128i>(_Vector), 8)),
                _IntrinBitcast<__m128>(_Vector)));

            _MaskStoreUnaligned(_Start, ~((1u << (_XmmWidth - (_Where - _Start))) - 1u), _Vector);
        }

        return _Where;
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _DesiredType_* _CompressStoreAligned(
        _DesiredType_*                      _Where,
        _Simd_mask_type<_DesiredType_>      _Mask,
        _VectorType_                        _Vector) noexcept
    {
        if      constexpr (sizeof(_DesiredType_) == 8) {
            _StoreAligned(_Where, _mm_shuffle_epi8(_IntrinBitcast<__m128i>(_Vector),
                _LoadUnaligned<__m128i>(_Tables64BitSse._Shuffle[_Mask])));

            algorithm::AdvanceBytes(_Where, _Tables64BitSse._Size[_Mask]);
        }
        else if constexpr (sizeof(_DesiredType_) == 4) {
            _StoreAligned(_Where, _mm_shuffle_epi8(_IntrinBitcast<__m128i>(_Vector),
                _LoadUnaligned<__m128i>(_Tables32BitSse._Shuffle[_Mask])));

            algorithm::AdvanceBytes(_Where, _Tables32BitSse._Size[_Mask]);
        }
        else if constexpr (sizeof(_DesiredType_) == 2) {
            _StoreAligned(_Where, _mm_shuffle_epi8(_IntrinBitcast<__m128i>(_Vector),
                _LoadUnaligned<__m128i>(_Tables16BitSse._Shuffle[_Mask])));

            algorithm::AdvanceBytes(_Where, _Tables16BitSse._Size[_Mask]);
        }
        else if constexpr (sizeof(_DesiredType_) == 1) {
            auto _Start = _Where;

            _Where = _CompressStoreLowerHalf(_Where, _Mask & 0xFF, _Vector);
            _Where = _CompressStoreLowerHalf(_Where, (_Mask >> 8) & 0xFF, _mm_movehl_ps(
                _IntrinBitcast<__m128>(_mm_slli_si128(_IntrinBitcast<__m128i>(_Vector), 8)),
                _IntrinBitcast<__m128>(_Vector)));

            _MaskStoreUnaligned(_Start, ~((1u << (_XmmWidth - (_Where - _Start))) - 1u), _Vector);
        }

        return _Where;
    }
};

template <>
class _SimdMemoryAccess<arch::CpuFeature::SSE3, numeric::xmm128> :
    public _SimdMemoryAccess<arch::CpuFeature::SSE2, numeric::xmm128>
{};

template <>
class _SimdMemoryAccess<arch::CpuFeature::SSSE3, numeric::xmm128> :
    public _SimdMemoryAccess<arch::CpuFeature::SSE3, numeric::xmm128>
{
    static constexpr auto _Generation   = arch::CpuFeature::SSE2;
    using _RegisterPolicy               = numeric::xmm128;

    template <class _DesiredType_>
    using _Simd_mask_type = type_traits::__deduce_simd_mask_type<_Generation, _DesiredType_, _RegisterPolicy>;
public:
    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _DesiredType_* _CompressStoreLowerHalf(
        _DesiredType_*                          _Where,
        const _Simd_mask_type<_DesiredType_>    _Mask,
        const _VectorType_                      _Vector) noexcept
    {
        static_assert(sizeof(_DesiredType_) != 8);

        __m128i _Shuffle;

        if constexpr (sizeof(_DesiredType_) == 4)
            _Shuffle = _LoadLowerHalf<__m128i>(_Tables32BitSse._Shuffle[_Mask]);

        else if constexpr (sizeof(_DesiredType_) == 2)
            _Shuffle = _LoadLowerHalf<__m128i>(_Tables16BitSse._Shuffle[_Mask]);

        else if constexpr (sizeof(_DesiredType_) == 1)
            _Shuffle = _LoadLowerHalf<__m128i>(_Tables8BitSse._Shuffle[_Mask]);

        const auto _Destination = _mm_shuffle_epi8(_IntrinBitcast<__m128i>(_Vector), _Shuffle);
        _mm_storel_epi64(reinterpret_cast<__m128i*>(_Where), _Destination);

        if constexpr (sizeof(_DesiredType_) == 4)
            algorithm::AdvanceBytes(_Where, _Tables32BitSse._Size[_Mask]);

        else if constexpr (sizeof(_DesiredType_) == 2)
            algorithm::AdvanceBytes(_Where, _Tables16BitSse._Size[_Mask]);

        else if constexpr (sizeof(_DesiredType_) == 1)
            algorithm::AdvanceBytes(_Where, _Tables8BitSse._Size[_Mask]);

        return _Where;
    }
    
    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _DesiredType_* _CompressStoreUpperHalf(
        _DesiredType_*                          _Where,
        const _Simd_mask_type<_DesiredType_>    _Mask,
        const _VectorType_                      _Vector) noexcept
    {
        static_assert(sizeof(_DesiredType_) != 8);
        __m128i _Shuffle;

       if constexpr (sizeof(_DesiredType_) == 4)
           _Shuffle = _LoadUpperHalf<__m128i>(_Tables32BitSse._Shuffle[_Mask]);
        else if constexpr (sizeof(_DesiredType_) == 2)
           _Shuffle = _LoadUpperHalf<__m128i>(_Tables16BitSse._Shuffle[_Mask]);
        else if constexpr (sizeof(_DesiredType_) == 1)
           _Shuffle = _LoadUpperHalf<__m128i>(_Tables8BitSse._Shuffle[_Mask]);


        _mm_storeh_pd(reinterpret_cast<double*>(_Where), _IntrinBitcast<__m128d>(
            _mm_shuffle_epi8(_IntrinBitcast<__m128i>(_Vector), _Shuffle)));

        if constexpr (sizeof(_DesiredType_) == 4)
            algorithm::AdvanceBytes(_Where, _Tables32BitSse._Size[_Mask]);
        else if constexpr (sizeof(_DesiredType_) == 2)
            algorithm::AdvanceBytes(_Where, _Tables16BitSse._Size[_Mask]);
        else if constexpr (sizeof(_DesiredType_) == 1)
            algorithm::AdvanceBytes(_Where, _Tables8BitSse._Size[_Mask]);

        return _Where;
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _DesiredType_* _CompressStoreUnaligned(
        _DesiredType_*                          _Where,
        const _Simd_mask_type<_DesiredType_>    _Mask,
        const _VectorType_                      _Vector) noexcept
    {
        if      constexpr (sizeof(_DesiredType_) == 8) {
            _StoreUnaligned(_Where, _mm_shuffle_epi8(_IntrinBitcast<__m128i>(_Vector),
                _LoadUnaligned<__m128i>(_Tables64BitSse._Shuffle[_Mask])));

            algorithm::AdvanceBytes(_Where, _Tables64BitSse._Size[_Mask]);
        }
        else if constexpr (sizeof(_DesiredType_) == 4) {
            _StoreUnaligned(_Where, _mm_shuffle_epi8(_IntrinBitcast<__m128i>(_Vector),
                _LoadUnaligned<__m128i>(_Tables32BitSse._Shuffle[_Mask])));

            algorithm::AdvanceBytes(_Where, _Tables32BitSse._Size[_Mask]);
        }
        else if constexpr (sizeof(_DesiredType_) == 2) {
            _StoreUnaligned(_Where, _mm_shuffle_epi8(_IntrinBitcast<__m128i>(_Vector),
                _LoadUnaligned<__m128i>(_Tables16BitSse._Shuffle[_Mask])));

            algorithm::AdvanceBytes(_Where, _Tables16BitSse._Size[_Mask]);
        }
        else if constexpr (sizeof(_DesiredType_) == 1) {
            auto _Start = _Where;

            _Where = _CompressStoreLowerHalf(_Where, _Mask & 0xFF, _Vector);
            _Where = _CompressStoreLowerHalf(_Where, (_Mask >> 8) & 0xFF, _mm_movehl_ps(
                _IntrinBitcast<__m128>(_mm_slli_si128(_IntrinBitcast<__m128i>(_Vector), 8)),
                _IntrinBitcast<__m128>(_Vector)));

            _MaskStoreUnaligned(_Start, ~((1u << (_XmmWidth - (_Where - _Start))) - 1u), _Vector);
        }

        return _Where;
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _DesiredType_* _CompressStoreAligned(
        _DesiredType_*                          _Where,
        const _Simd_mask_type<_DesiredType_>    _Mask,
        const _VectorType_                      _Vector) noexcept
    {
        if      constexpr (sizeof(_DesiredType_) == 8) {
            _StoreAligned(_Where, _mm_shuffle_epi8(_IntrinBitcast<__m128i>(_Vector),
                _LoadUnaligned<__m128i>(_Tables64BitSse._Shuffle[_Mask])));

            algorithm::AdvanceBytes(_Where, _Tables64BitSse._Size[_Mask]);
        }
        else if constexpr (sizeof(_DesiredType_) == 4) {
            _StoreAligned(_Where, _mm_shuffle_epi8(_IntrinBitcast<__m128i>(_Vector),
                _LoadUnaligned<__m128i>(_Tables32BitSse._Shuffle[_Mask])));

            algorithm::AdvanceBytes(_Where, _Tables32BitSse._Size[_Mask]);
        }
        else if constexpr (sizeof(_DesiredType_) == 2) {
            _StoreAligned(_Where, _mm_shuffle_epi8(_IntrinBitcast<__m128i>(_Vector),
                _LoadUnaligned<__m128i>(_Tables16BitSse._Shuffle[_Mask])));

            algorithm::AdvanceBytes(_Where, _Tables16BitSse._Size[_Mask]);
        }
        else if constexpr (sizeof(_DesiredType_) == 1) {
            auto _Start = _Where;

            _Where = _CompressStoreLowerHalf(_Where, _Mask & 0xFF, _Vector);
            _Where = _CompressStoreLowerHalf(_Where, (_Mask >> 8) & 0xFF, _mm_movehl_ps(
                _IntrinBitcast<__m128>(_mm_slli_si128(_IntrinBitcast<__m128i>(_Vector), 8)),
                _IntrinBitcast<__m128>(_Vector)));

            _MaskStoreUnaligned(_Start, ~((1u << (_XmmWidth - (_Where - _Start))) - 1u), _Vector);
        }

        return _Where;
    }
};

template <>
class _SimdMemoryAccess<arch::CpuFeature::SSE41, numeric::xmm128> :
    public _SimdMemoryAccess<arch::CpuFeature::SSSE3, numeric::xmm128>
{    
public:
    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _NonTemporalLoad(const void* where) noexcept {
        return _IntrinBitcast<_VectorType_>(_mm_stream_load_si128(reinterpret_cast<const __m128i*>(where)));
    }
};

template <>
class _SimdMemoryAccess<arch::CpuFeature::SSE42, numeric::xmm128> :
    public _SimdMemoryAccess<arch::CpuFeature::SSE41, numeric::xmm128>
{};

#pragma endregion

#pragma region Avx-Avx2 memory access

template <>
class _SimdMemoryAccess<arch::CpuFeature::AVX, numeric::ymm256>
{
    static constexpr auto _Generation   = arch::CpuFeature::AVX;
    using _RegisterPolicy               = numeric::ymm256;

    template <class _DesiredType_>
    using _Simd_mask_type = type_traits::__deduce_simd_mask_type<_Generation, _DesiredType_, _RegisterPolicy>;

    template <
        int32 _First_,
        int32 _Second_>
    static constexpr int32 _Max() noexcept {
        return (_First_ > _Second_) ? _First_ : _Second_;
    }
public:
    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _LoadUpperHalf(const void* _Where) noexcept {
        return _IntrinBitcast<_VectorType_>(_mm256_insertf128_si256(_mm256_setzero_si256(), 
            _mm_lddqu_si128(reinterpret_cast<const __m128i*>(_Where)), 1));
    }

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _LoadLowerHalf(const void* _Where) noexcept {
        return _IntrinBitcast<_VectorType_>(_mm_lddqu_si128(reinterpret_cast<const __m128i*>(_Where)));
    }

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _NonTemporalLoad(const void* _Where) noexcept {
        return _LoadAligned<_VectorType_, void>(_Where);
    }

    template <typename _VectorType_>
    static simd_stl_always_inline void _NonTemporalStore(
        void*           _Where,
        _VectorType_    _Vector) noexcept 
    {
        _mm256_stream_si256(reinterpret_cast<__m256i*>(_Where), _IntrinBitcast<__m256i>(_Vector));
    }

    static simd_stl_always_inline void _StreamingFence() noexcept {
        return _mm_sfence();
    }

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _LoadUnaligned(const void* _Where) noexcept {
        if      constexpr (std::is_same_v<_VectorType_, __m256i>)
            return _mm256_lddqu_si256(reinterpret_cast<const __m256i*>(_Where));

        else if constexpr (std::is_same_v<_VectorType_, __m256d>)
            return _mm256_loadu_pd(reinterpret_cast<const double*>(_Where));

        else if constexpr (std::is_same_v<_VectorType_, __m256>)
            return _mm256_loadu_ps(reinterpret_cast<const float*>(_Where));
    }

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _LoadAligned(const void* _Where) noexcept {
        if      constexpr (std::is_same_v<_VectorType_, __m256i>)
            return _mm256_load_si256(reinterpret_cast<const __m256i*>(_Where));

        else if constexpr (std::is_same_v<_VectorType_, __m256d>)
            return _mm256_load_pd(reinterpret_cast<const double*>(_Where));

        else if constexpr (std::is_same_v<_VectorType_, __m256>)
            return _mm256_load_ps(reinterpret_cast<const float*>(_Where));
    }

    template <typename _VectorType_>
    static simd_stl_always_inline void _StoreUpperHalf(
        void*           _Where,
        _VectorType_    _Vector) noexcept
    {
        _mm_storeu_si128(reinterpret_cast<__m128i*>(_Where), _mm256_extractf128_si256(_IntrinBitcast<__m256i>(_Vector), 1));
    }

    template <typename _VectorType_>
    static simd_stl_always_inline void _StoreLowerHalf(
        void*           _Where,
        _VectorType_    _Vector) noexcept
    {
        _mm_storeu_si128(reinterpret_cast<__m128i*>(_Where), _IntrinBitcast<__m128i>(_Vector));
    }

    template <typename _VectorType_>
    static simd_stl_always_inline void _StoreUnaligned(
        void*           _Where,
        _VectorType_    _Vector) noexcept
    {
        if      constexpr (std::is_same_v<_VectorType_, __m256i>)
            return _mm256_storeu_si256(reinterpret_cast<__m256i*>(_Where), _Vector);

        else if constexpr (std::is_same_v<_VectorType_, __m256d>)
            return _mm256_storeu_pd(reinterpret_cast<double*>(_Where), _Vector);

        else if constexpr (std::is_same_v<_VectorType_, __m256>)
            return _mm256_storeu_ps(reinterpret_cast<float*>(_Where), _Vector);
    }

    template <typename _VectorType_>
    static simd_stl_always_inline void _StoreAligned(
        void*           _Where,
        _VectorType_    _Vector) noexcept
    {
        if      constexpr (std::is_same_v<_VectorType_, __m256i>)
            return _mm256_store_si256(reinterpret_cast<__m256i*>(_Where), _Vector);

        else if constexpr (std::is_same_v<_VectorType_, __m256d>)
            return _mm256_store_pd(reinterpret_cast<double*>(_Where), _Vector);
   
        else if constexpr (std::is_same_v<_VectorType_, __m256>)
            return _mm256_store_ps(reinterpret_cast<float*>(_Where), _Vector);
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline void _MaskStoreUnaligned(
        _DesiredType_*                          _Where,
        const _Simd_mask_type<_DesiredType_>    _Mask,
        const _VectorType_                      _Vector) noexcept
    {
        if constexpr (sizeof(_DesiredType_) == 8) {
            _mm256_maskstore_pd(reinterpret_cast<double*>(_Where),
                _SimdToVector<_Generation, _RegisterPolicy, __m256i, _DesiredType_>(_Mask),
                _IntrinBitcast<__m256d>(_Vector));
        }
        else if constexpr (sizeof(_DesiredType_) == 4) {
            _mm256_maskstore_ps(reinterpret_cast<float*>(_Where),
                _SimdToVector<_Generation, _RegisterPolicy, __m256i, _DesiredType_>(_Mask),
                _IntrinBitcast<__m256>(_Vector));
        }
        else {
            _StoreUnaligned(_Where, _SimdBlend<_Generation, _RegisterPolicy, _DesiredType_>(
                _Vector, _LoadUnaligned<_VectorType_>(_Where), _Mask));
        }
    }
    
    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline void _MaskStoreAligned(
        _DesiredType_*                          _Where,
        const _Simd_mask_type<_DesiredType_>    _Mask,
        const _VectorType_                      _Vector) noexcept
    {
        if constexpr (sizeof(_DesiredType_) == 8) {
            _mm256_maskstore_pd(reinterpret_cast<double*>(_Where),
                _SimdToVector<_Generation, _RegisterPolicy, __m256i, _DesiredType_>(_Mask),
                _IntrinBitcast<__m256d>(_Vector));
        }
        else if constexpr (sizeof(_DesiredType_) == 4) {
            _mm256_maskstore_ps(reinterpret_cast<float*>(_Where),
                _SimdToVector<_Generation, _RegisterPolicy, __m256i, _DesiredType_>(_Mask),
                _IntrinBitcast<__m256>(_Vector));
        }
        else {
            _StoreAligned(_Where, _SimdBlend<_Generation, _RegisterPolicy, _DesiredType_>(
                _Vector, _LoadAligned<_VectorType_>(_Where), _Mask));
        }
    }

    template <
        typename _VectorType_,
        typename _DesiredType_>
    static simd_stl_always_inline _VectorType_ _MaskLoadUnaligned(
        const _DesiredType_*                    _Where,
        const _Simd_mask_type<_DesiredType_>    _Mask) noexcept
    {
        if constexpr (sizeof(_DesiredType_) == 8) {
            return _IntrinBitcast<_VectorType_>(_mm256_maskload_pd(
                reinterpret_cast<const double*>(_Where),
                _SimdToVector<_Generation, _RegisterPolicy, __m256i, _DesiredType_>(_Mask)));
        }
        else if constexpr (sizeof(_DesiredType_) == 4) {
            return _IntrinBitcast<_VectorType_>(_mm256_maskload_ps(
                reinterpret_cast<const float*>(_Where), 
                _SimdToVector<_Generation, _RegisterPolicy, __m256i, _DesiredType_>(_Mask)));
        }
        else {
            return _SimdBlend<_Generation, _RegisterPolicy, _DesiredType_>(
                _LoadUnaligned<_VectorType_>(_Where), 
                _SimdBroadcastZeros<_Generation, _RegisterPolicy, _VectorType_>(), _Mask);
        }
    }

    template <
        typename _VectorType_,
        typename _DesiredType_>
    static simd_stl_always_inline _VectorType_ _MaskLoadAligned(
        const _DesiredType_*                    _Where,
        const _Simd_mask_type<_DesiredType_>    _Mask) noexcept
    {
        if constexpr (sizeof(_DesiredType_) == 8) {
            return _IntrinBitcast<_VectorType_>(_mm256_maskload_pd(
                reinterpret_cast<const double*>(_Where),
                _SimdToVector<_Generation, _RegisterPolicy, __m256i, _DesiredType_>(_Mask)));
        }
        else if constexpr (sizeof(_DesiredType_) == 4) {
            return _IntrinBitcast<_VectorType_>(_mm256_maskload_ps(
                reinterpret_cast<const float*>(_Where),
                _SimdToVector<_Generation, _RegisterPolicy, __m256i, _DesiredType_>(_Mask)));
        }
        else {
            return _SimdBlend<_Generation, _RegisterPolicy, _DesiredType_>(
                _LoadAligned<_VectorType_>(_Where),
                _SimdBroadcastZeros<_Generation, _RegisterPolicy, _VectorType_>(), _Mask);
        }
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _DesiredType_* _CompressStoreLowerHalf(
        _DesiredType_*                  _Where,
        _Simd_mask_type<_DesiredType_>  _Mask,
        _VectorType_                    _Vector) noexcept
    {
        return _SimdCompressStoreUnaligned<arch::CpuFeature::SSE42, xmm128, _DesiredType_>(
            _Where, _Mask, _IntrinBitcast<__m128i>(_Vector));
    }
    
    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _DesiredType_* _CompressStoreUpperHalf(
        _DesiredType_*                  _Where,
        _Simd_mask_type<_DesiredType_>  _Mask,
        _VectorType_                    _Vector) noexcept
    {
        return _SimdCompressStoreUnaligned<arch::CpuFeature::SSE42, xmm128, _DesiredType_>(_Where, _Mask,
            _mm256_extractf128_si256(_IntrinBitcast<__m256i>(_Vector), 1));
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _DesiredType_* _CompressStoreUnaligned(
        _DesiredType_*                      _Where,
        _Simd_mask_type<_DesiredType_>      _Mask,
        _VectorType_                        _Vector) noexcept
    {
        using _MaskType = _Simd_mask_type<_DesiredType_>;
        using _HalfType = IntegerForSize<_Max<(sizeof(_DesiredType_) >> 1), 1>()>::Unsigned;

        constexpr auto _Maximum = math::MaximumIntegralLimit<_HalfType>();
        constexpr auto _Shift = (sizeof(_MaskType) << 2);

        _Where = _CompressStoreLowerHalf<_DesiredType_>(_Where, _Mask & _Maximum, _Vector);
        _Where = _CompressStoreUpperHalf<_DesiredType_>(_Where, (_Mask >> _Shift) & _Maximum, _Vector);

        return _Where;
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _DesiredType_* _CompressStoreAligned(
        _DesiredType_*                      _Where,
        _Simd_mask_type<_DesiredType_>      _Mask,
        _VectorType_                        _Vector) noexcept
    {
        return _CompressStoreUnaligned<_DesiredType_>(_Where, _Mask, _Vector);
    }
};

template <>
class _SimdMemoryAccess<arch::CpuFeature::AVX2, numeric::ymm256>:
    public _SimdMemoryAccess<arch::CpuFeature::AVX, numeric::ymm256>
{
    static constexpr auto _Generation   = arch::CpuFeature::AVX2;
    using _RegisterPolicy               = numeric::ymm256;

    template <class _DesiredType_>
    using _Simd_mask_type = type_traits::__deduce_simd_mask_type<_Generation, _DesiredType_, _RegisterPolicy>;

    template <
        int32 _First_,
        int32 _Second_>
    static constexpr int32 _Max() noexcept {
        return (_First_ > _Second_) ? _First_ : _Second_;
    }
public:
    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _NonTemporalLoad(const void* _Where) noexcept {
        return _IntrinBitcast<_VectorType_>(_mm256_stream_load_si256(reinterpret_cast<const __m256i*>(_Where)));
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _DesiredType_* _CompressStoreUnaligned(
        _DesiredType_*                      _Where,
        _Simd_mask_type<_DesiredType_>      _Mask,
        _VectorType_                        _Vector) noexcept
    {
        if constexpr (sizeof(_DesiredType_) == 8) {
            const auto _Shuffle = _mm256_cvtepu8_epi32(_mm_loadu_si64(_Tables64BitAvx._Shuffle[_Mask]));
            const auto _Destination = _mm256_permutevar8x32_epi32(_IntrinBitcast<__m256i>(_Vector), _Shuffle);

            _mm256_storeu_si256(reinterpret_cast<__m256i*>(_Where), _Destination);
            algorithm::AdvanceBytes(_Where, _Tables64BitAvx._Size[_Mask]);
        }
        else if constexpr (sizeof(_DesiredType_) == 4) {
            const auto _Shuffle = _mm256_cvtepu8_epi32(_mm_loadu_si64(_Tables32BitAvx._Shuffle[_Mask]));
            const auto _Destination = _mm256_permutevar8x32_epi32(_IntrinBitcast<__m256i>(_Vector), _Shuffle);

            _mm256_storeu_si256(reinterpret_cast<__m256i*>(_Where), _Destination);
            algorithm::AdvanceBytes(_Where, _Tables32BitAvx._Size[_Mask]);
        }
        else {
            constexpr auto _Length = sizeof(_VectorType_) / sizeof(_DesiredType_);
            _DesiredType_ _Source[_Length];

            _SimdStoreUnaligned<_Generation, _RegisterPolicy>(_Source, _Vector);

            for (auto _Index = 0; _Index < _Length; ++_Index)
                if ((_Mask >> _Index) & 1)
                    *_Where++ = _Source[_Index];
        }

        return _Where;
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _DesiredType_* _CompressStoreAligned(
        _DesiredType_*                      _Where,
        _Simd_mask_type<_DesiredType_>      _Mask,
        _VectorType_                        _Vector) noexcept
    {
        if constexpr (sizeof(_DesiredType_) == 8) {
            const auto _Shuffle = _mm256_cvtepu8_epi32(_mm_loadu_si64(_Tables64BitAvx._Shuffle[_Mask]));
            const auto _Destination = _mm256_permutevar8x32_epi32(_IntrinBitcast<__m256i>(_Vector), _Shuffle);

            _mm256_store_si256(reinterpret_cast<__m256i*>(_Where), _Destination);
            algorithm::AdvanceBytes(_Where, _Tables64BitAvx._Size[_Mask]);
        }
        else if constexpr (sizeof(_DesiredType_) == 4) {
            const auto _Shuffle = _mm256_cvtepu8_epi32(_mm_loadu_si64(_Tables32BitAvx._Shuffle[_Mask]));
            const auto _Destination = _mm256_permutevar8x32_epi32(_IntrinBitcast<__m256i>(_Vector), _Shuffle);

            _mm256_store_si256(reinterpret_cast<__m256i*>(_Where), _Destination);
            algorithm::AdvanceBytes(_Where, _Tables32BitAvx._Size[_Mask]);
        }
        else {
            constexpr auto _Length = sizeof(_VectorType_) / sizeof(_DesiredType_);
            _DesiredType_ _Source[_Length];

            _SimdStoreUnaligned<_Generation, _RegisterPolicy>(_Source, _Vector);

            for (auto _Index = 0; _Index < _Length; ++_Index)
                if ((_Mask >> _Index) & 1)
                    *_Where++ = _Source[_Index];
        }

        return _Where;
    }
};

#pragma endregion

#pragma region Avx512 memory access 

template <>
class _SimdMemoryAccess<arch::CpuFeature::AVX512F, zmm512>
{
    static constexpr auto _Generation   = arch::CpuFeature::AVX512F;
    using _RegisterPolicy               = zmm512;

    template <class _DesiredType_>
    using _Simd_mask_type = type_traits::__deduce_simd_mask_type<_Generation, _DesiredType_, _RegisterPolicy>;
public:
    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _LoadUpperHalf(const void* _Where) noexcept {
        return _IntrinBitcast<_VectorType_>(_mm256_lddqu_si256(reinterpret_cast<const __m256i*>(_Where)));
    }

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _LoadLowerHalf(const void* _Where) noexcept {
        return _IntrinBitcast<_VectorType_>(_mm512_inserti64x4(
            _mm512_setzero_si512(), _mm256_lddqu_si256(reinterpret_cast<const __m256i*>(_Where)), 1));
    }

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _NonTemporalLoad(const void* _Where) noexcept {
        return _IntrinBitcast<_VectorType_>(_mm512_stream_load_si512(_Where));
    }

    template <typename _VectorType_>
    static simd_stl_always_inline void _NonTemporalStore(
        void* _Where,
        _VectorType_    _Vector) noexcept
    {
        _mm512_stream_si512(_Where, _IntrinBitcast<__m512i>(_Vector));
    }

    static simd_stl_always_inline void _StreamingFence() noexcept {
        return _mm_sfence();
    }

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _LoadUnaligned(const void* _Where) noexcept {
        if      constexpr (std::is_same_v<_VectorType_, __m512i>)
            return _mm512_loadu_si512(_Where);

        else if constexpr (std::is_same_v<_VectorType_, __m512d>)
            return _mm512_loadu_pd(_Where);

        else if constexpr (std::is_same_v<_VectorType_, __m512>)
            return _mm512_loadu_ps(_Where);
    }

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _LoadAligned(const void* _Where) noexcept {
        if      constexpr (std::is_same_v<_VectorType_, __m512i>)
            return _mm512_load_si512(_Where);

        else if constexpr (std::is_same_v<_VectorType_, __m512d>)
            return _mm512_load_pd(_Where);

        else if constexpr (std::is_same_v<_VectorType_, __m512>)
            return _mm512_load_ps(_Where);
    }

    template <typename _VectorType_>
    static simd_stl_always_inline void _StoreUpperHalf(
        void* _Where,
        _VectorType_    _Vector) noexcept
    {
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(_Where), _mm512_extracti64x4_epi64(_IntrinBitcast<__m512i>(_Vector), 1));
    }

    template <typename _VectorType_>
    static simd_stl_always_inline void _StoreLowerHalf(
        void* _Where,
        _VectorType_    _Vector) noexcept
    {
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(_Where), _IntrinBitcast<__m256d>(_Vector));
    }

    template <typename _VectorType_>
    static simd_stl_always_inline void _StoreUnaligned(
        void*           _Where,
        _VectorType_    _Vector) noexcept
    {
        if      constexpr (std::is_same_v<_VectorType_, __m512i>)
            return _mm512_storeu_si512(_Where, _Vector);

        else if constexpr (std::is_same_v<_VectorType_, __m512d>)
            return _mm512_storeu_pd(_Where, _Vector);

        else if constexpr (std::is_same_v<_VectorType_, __m512>)
            return _mm512_storeu_ps(_Where, _Vector);
    }

    template <typename _VectorType_>
    static simd_stl_always_inline void _StoreAligned(
        void*           _Where,
        _VectorType_    _Vector) noexcept
    {
        if      constexpr (std::is_same_v<_VectorType_, __m512i>)
            return _mm512_store_si512(_Where, _Vector);

        else if constexpr (std::is_same_v<_VectorType_, __m512d>)
            return _mm512_store_pd(_Where, _Vector);
   
        else if constexpr (std::is_same_v<_VectorType_, __m512>)
            return _mm512_store_ps(_Where, _Vector);
    }

     template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline void _MaskStoreUnaligned(
        _DesiredType_*                          _Where,
        const _Simd_mask_type<_DesiredType_>    _Mask,
        const _VectorType_                      _Vector) noexcept
    {
        if constexpr (sizeof(_DesiredType_) == 8)
            _mm512_mask_storeu_epi64(_Where, _Mask, _IntrinBitcast<__m512i>(_Vector));
        
        else if constexpr (sizeof(_DesiredType_) == 4)
            _mm512_mask_storeu_epi32(_Where, _Mask, _IntrinBitcast<__m512i>(_Vector));
        
        else
            _StoreUnaligned(_Where, _SimdBlend<_Generation, _RegisterPolicy, _DesiredType_>(
                _LoadUnaligned<_VectorType_>(_Where), _Vector, _Mask));
    }
    
    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline void _MaskStoreAligned(
        _DesiredType_*                          _Where,
        const _Simd_mask_type<_DesiredType_>    _Mask,
        const _VectorType_                      _Vector) noexcept
    {
        if constexpr (sizeof(_DesiredType_) == 8)
            _mm512_mask_store_epi64(_Where, _Mask, _IntrinBitcast<__m512i>(_Vector));

        else if constexpr (sizeof(_DesiredType_) == 4)
            _mm512_mask_store_epi32(_Where, _Mask, _IntrinBitcast<__m512i>(_Vector));

        else
            _StoreAligned(_Where, _SimdBlend<_Generation, _RegisterPolicy, _DesiredType_>(
                _LoadAligned<_VectorType_>(_Where), _Vector, _Mask));
    }

    template <
        typename _VectorType_,
        typename _DesiredType_>
    static simd_stl_always_inline _VectorType_ _MaskLoadUnaligned(
        const _DesiredType_*                    _Where,
        const _Simd_mask_type<_DesiredType_>    _Mask) noexcept
    {
        if constexpr (sizeof(_DesiredType_) == 8)
            return _IntrinBitcast<_VectorType_>(_mm512_mask_loadu_epi64(_mm512_setzero_si512(), _Mask, _Where));
        
        else if constexpr (sizeof(_DesiredType_) == 4)
            return _IntrinBitcast<_VectorType_>(_mm512_mask_loadu_epi32(_mm512_setzero_si512(), _Mask, _Where));
        
        else
            return _SimdBlend<_Generation, _RegisterPolicy, _DesiredType_>(
                _SimdBroadcastZeros<_Generation, _RegisterPolicy, _VectorType_>(),
                _LoadUnaligned<_VectorType_>(_Where), _Mask);
    }

    template <
        typename _VectorType_,
        typename _DesiredType_>
    static simd_stl_always_inline _VectorType_ _MaskLoadAligned(
        const _DesiredType_*                    _Where,
        const _Simd_mask_type<_DesiredType_>    _Mask) noexcept
    {
        if constexpr (sizeof(_DesiredType_) == 8)
            return _IntrinBitcast<_VectorType_>(_mm512_mask_load_epi64(_mm512_setzero_si512(), _Mask, _Where));
        
        else if constexpr (sizeof(_DesiredType_) == 4)
            return _IntrinBitcast<_VectorType_>(_mm512_mask_load_epi32(_mm512_setzero_si512(), _Mask, _Where));
        
        else
            return _SimdBlend<_Generation, _RegisterPolicy, _DesiredType_>(
                _SimdBroadcastZeros<_Generation, _RegisterPolicy, _VectorType_>(),
                _LoadAligned<_VectorType_>(_Where), _Mask);
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _DesiredType_* _CompressStoreLowerHalf(
        _DesiredType_*                  _Where,
        _Simd_mask_type<_DesiredType_>  _Mask,
        _VectorType_                    _Vector) noexcept
    {
        if constexpr (sizeof(_DesiredType_) == 8) {
            _mm512_mask_compressstoreu_epi64(_Where, _Mask & 0xFu, _IntrinBitcast<__m512i>(_Vector));
            algorithm::AdvanceBytes(_Where, (math::PopulationCount(_Mask) << 3));
        }
        else if constexpr (sizeof(_DesiredType_) == 4) {
            _mm512_mask_compressstoreu_epi32(_Where, _Mask & 0xFFu, _IntrinBitcast<__m512i>(_Vector));
            algorithm::AdvanceBytes(_Where, (math::PopulationCount(_Mask) << 2));
        }
        else {
            constexpr auto _Length = sizeof(__m256i) / sizeof(_DesiredType_);
            _DesiredType_ _Source[_Length];

            _SimdStoreUnaligned<arch::CpuFeature::AVX2, _RegisterPolicy>(_Source, _IntrinBitcast<__m256i>(_Vector));

            for (auto _Index = 0; _Index < _Length; ++_Index)
                if ((_Mask >> _Index) & 1)
                    *_Where++ = _Source[_Index];
        }

        return _Where;
    }
    
    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _DesiredType_* _CompressStoreUpperHalf(
        _DesiredType_*                  _Where,
        _Simd_mask_type<_DesiredType_>  _Mask,
        _VectorType_                    _Vector) noexcept
    {
        if constexpr (sizeof(_DesiredType_) == 8) {
            _mm512_mask_compress_epi64(_mm512_loadu_si512(_Where), (_Mask >> 4) & 0xFu, _IntrinBitcast<__m512i>(_Vector));
            algorithm::AdvanceBytes(_Where, (math::PopulationCount(_Mask) << 3));
        }
        else if constexpr (sizeof(_DesiredType_) == 4) {
            _mm512_mask_compressstoreu_epi32(_Where, (_Mask >> 8) & 0xFFu, _IntrinBitcast<__m512i>(_Vector));
            algorithm::AdvanceBytes(_Where, (math::PopulationCount(_Mask) << 2));
        }
        else {

        }

        return _Where;
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _DesiredType_* _CompressStoreUnaligned(
        _DesiredType_*                      _Where,
        _Simd_mask_type<_DesiredType_>      _Mask,
        _VectorType_                        _Vector) noexcept
    {
        if constexpr (sizeof(_DesiredType_) == 8) {
            _mm512_mask_compressstoreu_epi64(_Where, _Mask, _IntrinBitcast<__m512i>(_Vector));
            algorithm::AdvanceBytes(_Where, (math::PopulationCount(_Mask) << 3));
        }
        else if constexpr (sizeof(_DesiredType_) == 4) {
            _mm512_mask_compressstoreu_epi32(_Where, _Mask, _IntrinBitcast<__m512i>(_Vector));
            algorithm::AdvanceBytes(_Where, (math::PopulationCount(_Mask) << 2));
        }
        else {
            constexpr auto _Length = sizeof(__m256i) / sizeof(_DesiredType_);
            _DesiredType_ _Source[_Length];

            _SimdStoreUnaligned<arch::CpuFeature::AVX2, _RegisterPolicy>(_Source,
                _mm512_extracti64x4_epi64(_IntrinBitcast<__m512i>(_Vector), 1));

            for (auto _Index = 0; _Index < _Length; ++_Index)
                if ((_Mask >> _Index) & 1)
                    *_Where++ = _Source[_Index];
        }

        return _Where;
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _DesiredType_* _CompressStoreAligned(
        _DesiredType_*                      _Where,
        _Simd_mask_type<_DesiredType_>      _Mask,
        _VectorType_                        _Vector) noexcept
    {
        return _CompressStoreUnaligned<_DesiredType_>(_Where, _Mask, _Vector);
    }
};

template <>
class _SimdMemoryAccess<arch::CpuFeature::AVX512BW, zmm512> :
    public _SimdMemoryAccess<arch::CpuFeature::AVX512F, zmm512>
{};

template <>
class _SimdMemoryAccess<arch::CpuFeature::AVX512DQ, zmm512> :
    public _SimdMemoryAccess<arch::CpuFeature::AVX512BW, zmm512>
{};

template <>
class _SimdMemoryAccess<arch::CpuFeature::AVX512VL, zmm512> :
    public _SimdMemoryAccess<arch::CpuFeature::AVX512DQ, zmm512>
{};

#pragma endregion

template <
    arch::CpuFeature	_SimdGeneration_,
    class				_RegisterPolicy_,
    class				_VectorType_>
simd_stl_nodiscard simd_stl_always_inline _VectorType_ _SimdLoadUnaligned(const void* _Where) noexcept {
    _VerifyRegisterPolicy(_SimdGeneration_, _RegisterPolicy_);
    return _SimdMemoryAccess<_SimdGeneration_, _RegisterPolicy_>::template _LoadUnaligned<_VectorType_>(_Where);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    class				_RegisterPolicy_,
    class				_VectorType_>
simd_stl_nodiscard simd_stl_always_inline _VectorType_ _SimdLoadAligned(const void* _Where) noexcept {
    _VerifyRegisterPolicy(_SimdGeneration_, _RegisterPolicy_);
    return _SimdMemoryAccess<_SimdGeneration_, _RegisterPolicy_>::template _LoadAligned<_VectorType_>(_Where);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    class				_RegisterPolicy_,
    class				_VectorType_>
simd_stl_nodiscard simd_stl_always_inline _VectorType_ _SimdNonTemporalLoad(const void* _Where) noexcept {
    _VerifyRegisterPolicy(_SimdGeneration_, _RegisterPolicy_);
    return _SimdMemoryAccess<_SimdGeneration_, _RegisterPolicy_>::template _NonTemporalLoad<_VectorType_>(_Where);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    class				_RegisterPolicy_,
    class				_VectorType_>
simd_stl_nodiscard simd_stl_always_inline _VectorType_ _SimdLoadUpperHalf(const void* _Where) noexcept {
    _VerifyRegisterPolicy(_SimdGeneration_, _RegisterPolicy_);
    return _SimdMemoryAccess<_SimdGeneration_, _RegisterPolicy_>::template _LoadUpperHalf<_VectorType_>(_Where);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    class				_RegisterPolicy_,
    class				_VectorType_>
simd_stl_nodiscard simd_stl_always_inline _VectorType_ _SimdLoadLowerHalf(const void* _Where) noexcept {
    _VerifyRegisterPolicy(_SimdGeneration_, _RegisterPolicy_);
    return _SimdMemoryAccess<_SimdGeneration_, _RegisterPolicy_>::template _LoadLowerHalf<_VectorType_>(_Where);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    class				_RegisterPolicy_,
    typename            _VectorType_,
    typename            _DesiredType_>
simd_stl_always_inline _VectorType_ _SimdMaskLoadUnaligned(
    const _DesiredType_*                                    _Where,
    const type_traits::__deduce_simd_mask_type<
        _SimdGeneration_, _DesiredType_, _RegisterPolicy_>  _Mask) noexcept
{
    _VerifyRegisterPolicy(_SimdGeneration_, _RegisterPolicy_);
    return _SimdMemoryAccess<_SimdGeneration_, _RegisterPolicy_>::template _MaskLoadUnaligned<_VectorType_, _DesiredType_>(_Where, _Mask);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    class				_RegisterPolicy_,
    typename            _VectorType_,
    typename            _DesiredType_>
simd_stl_always_inline _VectorType_ _SimdMaskLoadAligned(
    const _DesiredType_*                                    _Where,
    const type_traits::__deduce_simd_mask_type<
        _SimdGeneration_, _DesiredType_, _RegisterPolicy_>  _Mask) noexcept
{
    _VerifyRegisterPolicy(_SimdGeneration_, _RegisterPolicy_);
    return _SimdMemoryAccess<_SimdGeneration_, _RegisterPolicy_>::template _MaskLoadAligned<_VectorType_, _DesiredType_>(_Where, _Mask);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    class				_RegisterPolicy_,
    class				_VectorType_>
simd_stl_nodiscard simd_stl_always_inline void _SimdStoreUnaligned(
    void*               _Where,
    const _VectorType_  _Vector) noexcept
{
    _VerifyRegisterPolicy(_SimdGeneration_, _RegisterPolicy_);
    _SimdMemoryAccess<_SimdGeneration_, _RegisterPolicy_>::template _StoreUnaligned(_Where, _Vector);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    class				_RegisterPolicy_,
    class				_VectorType_>
simd_stl_nodiscard simd_stl_always_inline void _SimdStoreAligned(
    void*               _Where,
    const _VectorType_  _Vector) noexcept
{
    _VerifyRegisterPolicy(_SimdGeneration_, _RegisterPolicy_);
    _SimdMemoryAccess<_SimdGeneration_, _RegisterPolicy_>::template _StoreAligned(_Where, _Vector);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    class				_RegisterPolicy_,
    class				_VectorType_>
simd_stl_nodiscard simd_stl_always_inline void _SimdStoreUpperHalf(
    void*               _Where,
    const _VectorType_  _Vector) noexcept
{
    _VerifyRegisterPolicy(_SimdGeneration_, _RegisterPolicy_);
    _SimdMemoryAccess<_SimdGeneration_, _RegisterPolicy_>::template _StoreUpperHalf(_Where, _Vector);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    class				_RegisterPolicy_,
    class				_VectorType_>
simd_stl_nodiscard simd_stl_always_inline void _SimdStoreLowerHalf(
    void*               _Where,
    const _VectorType_  _Vector) noexcept
{
    _VerifyRegisterPolicy(_SimdGeneration_, _RegisterPolicy_);
    _SimdMemoryAccess<_SimdGeneration_, _RegisterPolicy_>::template _StoreLowerHalf(_Where, _Vector);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    class				_RegisterPolicy_,
    class				_VectorType_>
simd_stl_nodiscard simd_stl_always_inline void _SimdNonTemporalStore(
    void*               _Where,
    const _VectorType_  _Vector) noexcept
{
    _VerifyRegisterPolicy(_SimdGeneration_, _RegisterPolicy_);
    _SimdMemoryAccess<_SimdGeneration_, _RegisterPolicy_>::template _NonTemporalStore(_Where, _Vector);
}

template <arch::CpuFeature _SimdGeneration_>
simd_stl_nodiscard simd_stl_always_inline void _SimdStreamingFence() noexcept {
    _SimdMemoryAccess<_SimdGeneration_, _DefaultRegisterPolicy<_SimdGeneration_>>::_StreamingFence();
}

template <
    arch::CpuFeature	_SimdGeneration_,
    class				_RegisterPolicy_,
    class               _DesiredType_,
    class               _VectorType_>
simd_stl_always_inline void _SimdMaskStoreUnaligned(
    _DesiredType_*                                          _Where,
    const type_traits::__deduce_simd_mask_type<
        _SimdGeneration_, _DesiredType_, _RegisterPolicy_>  _Mask,
    const _VectorType_                                      _Vector) noexcept
{
    _VerifyRegisterPolicy(_SimdGeneration_, _RegisterPolicy_);
    _SimdMemoryAccess<_SimdGeneration_, _RegisterPolicy_>::template _MaskStoreUnaligned(_Where, _Mask, _Vector);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    class				_RegisterPolicy_,
    class               _DesiredType_,
    class               _VectorType_>
simd_stl_always_inline void _SimdMaskStoreAligned(
    _DesiredType_*                                          _Where,
    const type_traits::__deduce_simd_mask_type<
        _SimdGeneration_, _DesiredType_, _RegisterPolicy_>  _Mask,
    const _VectorType_                                      _Vector) noexcept
{
    _VerifyRegisterPolicy(_SimdGeneration_, _RegisterPolicy_);
    _SimdMemoryAccess<_SimdGeneration_, _RegisterPolicy_>::template _MaskStoreAligned(_Where, _Mask, _Vector);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    class				_RegisterPolicy_,
    typename            _DesiredType_,
    typename            _VectorType_>
simd_stl_always_inline _DesiredType_* _SimdCompressStoreUnaligned(
    _DesiredType_*                          _Where,
    type_traits::__deduce_simd_mask_type<_SimdGeneration_,
        _DesiredType_, _RegisterPolicy_>    _Mask,
    const _VectorType_                      _Vector) noexcept
{
    _VerifyRegisterPolicy(_SimdGeneration_, _RegisterPolicy_);
    return _SimdMemoryAccess<_SimdGeneration_, _RegisterPolicy_>::template _CompressStoreUnaligned(_Where, _Mask, _Vector);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    class				_RegisterPolicy_,
    typename            _DesiredType_,
    typename            _VectorType_>
simd_stl_always_inline _DesiredType_* _SimdCompressStoreAligned(
    _DesiredType_*                          _Where,
    type_traits::__deduce_simd_mask_type<_SimdGeneration_,
        _DesiredType_, _RegisterPolicy_>    _Mask,
    const _VectorType_                      _Vector) noexcept
{
    _VerifyRegisterPolicy(_SimdGeneration_, _RegisterPolicy_);
    return _SimdMemoryAccess<_SimdGeneration_, _RegisterPolicy_>::template _CompressStoreAligned(_Where, _Mask, _Vector);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    class				_RegisterPolicy_,
    typename            _DesiredType_,
    typename            _VectorType_>
static simd_stl_always_inline _DesiredType_* _SimdCompressStoreLowerHalf(
    _DesiredType_*                          _Where,
    type_traits::__deduce_simd_mask_type<_SimdGeneration_,
        _DesiredType_, _RegisterPolicy_>    _Mask,
    const _VectorType_                      _Vector) noexcept
{
    _VerifyRegisterPolicy(_SimdGeneration_, _RegisterPolicy_);
    return _SimdMemoryAccess<_SimdGeneration_, _RegisterPolicy_>::template _CompressStoreLowerHalf(_Where, _Mask, _Vector);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    class				_RegisterPolicy_,
    typename            _DesiredType_,
    typename            _VectorType_>
static simd_stl_always_inline _DesiredType_* _SimdCompressStoreUpperHalf(
    _DesiredType_*                      _Where,
    type_traits::__deduce_simd_mask_type<_SimdGeneration_,
    _DesiredType_, _RegisterPolicy_>    _Mask,
    const _VectorType_                  _Vector) noexcept
{
    _VerifyRegisterPolicy(_SimdGeneration_, _RegisterPolicy_);
    return _SimdMemoryAccess<_SimdGeneration_, _RegisterPolicy_>::template _CompressStoreUpperHalf(_Where, _Mask, _Vector);
}

__SIMD_STL_NUMERIC_NAMESPACE_END
