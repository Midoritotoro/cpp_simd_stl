#pragma once

#include <simd_stl/numeric/Simd.h>
#include <src/simd_stl/numeric/CachePrefetcher.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <class _Type_>
simd_stl_declare_const_function simd_stl_always_inline const void* __find_scalar(
    const void* __first,
    const void* __last,
    _Type_      __value) noexcept
{
    auto _Current = static_cast<const _Type_*>(__first);

    while (_Current != _Last && *_Current != _Value)
        ++_Current;

    return _Current;
}

template <
    class _Simd_,
    class _CachePrefetcher_>
simd_stl_declare_const_function simd_stl_always_inline const void* __find_vectorized_internal(
    const void*                 __first,
    const void*                 __last,
    typename _Simd_::value_type __value,
    _CachePrefetcher_&&         __prefetcher) noexcept
{
    numeric::zero_upper_at_exit_guard<_Simd_::_Generation> _Guard;

    constexpr auto _Is_masked_memory_access_supported = _Simd_::template is_native_mask_store_supported_v<> &&
        _Simd_::template is_native_mask_load_supported_v<>;

    const auto _Size        = __byte_length(__first, _Last);
    const auto _AlignedSize = _Size & (~(sizeof(_Simd_) - 1));

    auto _Comparand = _Simd_(_Value);

    const void* _StopAt = __first;
    __advance_bytes(_StopAt, _AlignedSize);

    do {
        _Prefetcher(reinterpret_cast<const char*>(__first) + (sizeof(_Simd_)));

        const auto _Loaded  = _Simd_::load(__first);
        const auto _Mask    = _Comparand.mask_compare(_Loaded, type_traits::equal_to<>{});

        if (_Mask.anyOf())
            return static_cast<const typename _Simd_::value_type*>(__first) + _Mask.countTrailingZeroBits();

        __advance_bytes(__first, sizeof(_Simd_));
    } while (__first != _StopAt);

    if constexpr (_Is_masked_memory_access_supported) {
        if (_TailSize != 0) {
            const auto _TailMask    = _Simd_::make_tail_mask(_TailSize);
            const auto _Loaded      = _Simd_::mask_load(__first, _TailMask);

            const auto _Compared = _Comparand.native_compare(_Loaded, type_traits::equal_to<>{}) & _TailMask;
            const auto _Mask = numeric::simd_mask<_Simd_::_Generation,
                typename _Simd_::value_type>(numeric::_SimdToNativeMask<_Simd_::_Generation,
                typename _Simd_::policy_type, std::remove_cv_t<decltype(_Compared)>>(_Compared));

            if (_Mask.anyOf())
                return static_cast<const typename _Simd_::value_type*>(__first) + _Mask.countTrailingZeroBits();
        }
    }
    else {
        if (_TailSize != 0)
            _Last = _FindScalar(__first, _Last, _Value);
    }

    return _Last;
}

template <class _Type_>
simd_stl_declare_const_function _Type_* simd_stl_stdcall __find_vectorized(
    const void* __first,
    const void* __last,
    _Type_      __value) noexcept
{
    const auto _Size = __byte_length(__first, _Last);

    if (_Size > numeric::__zmm_threshold<_Type_>) {
        if constexpr (sizeof(_Type_) <= 2) {
            using _SimdType_ = simd_stl::numeric::simd512_avx512bw<_Type_>;

            if (const auto _AlignedSize = _Size & (~(sizeof(_SimdType_) - 1));
                _AlignedSize != 0 && arch::ProcessorFeatures::AVX512BW()
            )
                return const_cast<_Type_*>(static_cast<const volatile _Type_*>(
                    _FindVectorizedInternal<_SimdType_>(
                        __first, _Last, _AlignedSize,
                        _Size & (sizeof(_SimdType_) - sizeof(typename _SimdType_::value_type)),
                        _Value,
                        numeric::__cache_prefetcher<numeric::__prefetch_hint::NTA>{}
                    )
                ));
        } else {
            using _SimdType_ = simd_stl::numeric::simd512_avx512f<_Type_>;

            if (const auto _AlignedSize = _Size & (~(sizeof(_SimdType_) - 1));
                _AlignedSize != 0 && arch::ProcessorFeatures::AVX512F()
            )
                return const_cast<_Type_*>(static_cast<const volatile _Type_*>(
                    _FindVectorizedInternal<_SimdType_>(
                        __first, _Last, _AlignedSize,
                        _Size & (sizeof(_SimdType_) - sizeof(typename _SimdType_::value_type)),
                        _Value,
                        numeric::__cache_prefetcher<numeric::__prefetch_hint::NTA>{}
                    )
                ));
        }
    } else {
        if constexpr (sizeof(_Type_) <= 2) {
            using _SimdType_ = simd_stl::numeric::simd256_avx512vlbw<_Type_>;

            if (const auto _AlignedSize = _Size & (~(sizeof(_SimdType_) - 1));
                _AlignedSize != 0 && arch::ProcessorFeatures::AVX512BW() && arch::ProcessorFeatures::AVX512VL()
            )
                return const_cast<_Type_*>(static_cast<const volatile _Type_*>(
                    _FindVectorizedInternal<_SimdType_>(
                        __first, _Last, _AlignedSize,
                        _Size & (sizeof(_SimdType_) - sizeof(typename _SimdType_::value_type)),
                        _Value,
                        numeric::__cache_prefetcher<numeric::__prefetch_hint::NTA>{}
                    )
                ));
        } else {
            using _SimdType_ = simd_stl::numeric::simd256_avx512vlf<_Type_>;

            if (const auto _AlignedSize = _Size & (~(sizeof(_SimdType_) - 1));
                _AlignedSize != 0 && arch::ProcessorFeatures::AVX512F() && arch::ProcessorFeatures::AVX512VL()
            )
                return const_cast<_Type_*>(static_cast<const volatile _Type_*>(
                    _FindVectorizedInternal<_SimdType_>(
                        __first, _Last, _AlignedSize,
                        _Size & (sizeof(_SimdType_) - sizeof(typename _SimdType_::value_type)),
                        _Value,
                        numeric::__cache_prefetcher<numeric::__prefetch_hint::NTA>{}
                    )
                ));
        }
    }

    if (arch::ProcessorFeatures::AVX2()) {
        using _SimdType_ = simd_stl::numeric::simd256_avx2<_Type_>;

        if (const auto _AlignedSize = _Size & (~(sizeof(_SimdType_) - 1));
            _AlignedSize != 0 && arch::ProcessorFeatures::AVX2()
        )
            return const_cast<_Type_*>(static_cast<const volatile _Type_*>(
                _FindVectorizedInternal<_SimdType_>(
                    __first, _Last, _AlignedSize,
                    _Size & (sizeof(_SimdType_) - sizeof(typename _SimdType_::value_type)),
                    _Value,
                    numeric::__cache_prefetcher<numeric::__prefetch_hint::NTA>{}
                )
            ));
    } else if (arch::ProcessorFeatures::SSE2()) {
        using _SimdType_ = simd_stl::numeric::simd128_sse2<_Type_>;

        if (const auto _AlignedSize = _Size & (~(sizeof(_SimdType_) - 1));
            _AlignedSize != 0 && arch::ProcessorFeatures::SSE2()
        )
            return const_cast<_Type_*>(static_cast<const volatile _Type_*>(
                _FindVectorizedInternal<_SimdType_>(
                    __first, _Last, _AlignedSize,
                    _Size & (sizeof(_SimdType_) - sizeof(typename _SimdType_::value_type)),
                    _Value,
                    numeric::__cache_prefetcher<numeric::__prefetch_hint::NTA>{}
                )
            ));
    }

    return const_cast<_Type_*>(static_cast<const volatile _Type_*>(
        _FindScalar(__first, _Last, _Value)
    ));
}

__SIMD_STL_ALGORITHM_NAMESPACE_END