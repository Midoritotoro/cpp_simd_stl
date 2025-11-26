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
        return _SimdBlend<_Generation, _RegisterPolicy, _DesiredType_>(
            _SimdBroadcastZeros<_Generation, _RegisterPolicy, _VectorType_>(),
            _LoadAligned<_VectorType_>(_Where), _Mask);
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _DesiredType_* _CompressStoreUnaligned(
        _DesiredType_*                  _Where,
        _Simd_mask_type<_DesiredType_>  _Mask,
        const _VectorType_              _Vector) noexcept
    {}

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _DesiredType_* _CompressStoreAligned(
        _DesiredType_*                  _Where,
        _Simd_mask_type<_DesiredType_>  _Mask,
        const _VectorType_              _Vector) noexcept
    {}

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _DesiredType_* _CompressStoreLowerHalf(
        _DesiredType_* _Where,
        const _Simd_mask_type<_DesiredType_>    _Mask,
        const _VectorType_                      _Vector) noexcept
    {}

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _DesiredType_* _CompressStoreUpperHalf(
        _DesiredType_* _Where,
        const _Simd_mask_type<_DesiredType_>    _Mask,
        const _VectorType_                      _Vector) noexcept
    {}
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
