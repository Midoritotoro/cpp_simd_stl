#pragma once

#include <simd_stl/numeric/Simd.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <class _Type_>
simd_stl_declare_const_function simd_stl_always_inline const void* _FindScalar(
    const void* _First,
    const void* _Last,
    _Type_      _Value) noexcept
{
    auto _Current = static_cast<const _Type_*>(_First);

    while (_Current != _Last && *_Current != _Value)
        ++_Current;

    return (_Current == _Last) ? _Last : _Current;
}

template <class _Simd_>
simd_stl_declare_const_function simd_stl_always_inline const void* _FindVectorizedInternal(
    const void*                 _First,
    const void*                 _Last,
    typename _Simd_::value_type _Value) noexcept
{
    numeric::zero_upper_at_exit_guard<_Simd_::_Generation> _Guard;

    constexpr auto _Is_masked_memory_access_supported = _Simd_::template is_native_mask_store_supported_v<> &&
        _Simd_::template is_native_mask_load_supported_v<>;

    const auto _Size        = ByteLength(_First, _Last);
    const auto _AlignedSize = _Size & (~(sizeof(_Simd_) - 1));

    auto _Comparand = _Simd_(_Value);

    if (_AlignedSize != 0) {
        const void* _StopAt = _First;
        AdvanceBytes(_StopAt, _AlignedSize);

        do {
            const auto _Loaded  = _Simd_::loadUnaligned(_First);
            const auto _Mask    = _Comparand.maskEqual(_Loaded);

            if (_Mask.anyOf())
                return static_cast<const typename _Simd_::value_type*>(_First) + _Mask.countTrailingZeroBits();

            AdvanceBytes(_First, sizeof(_Simd_));
        } while (_First != _StopAt);
    }

    if constexpr (_Is_masked_memory_access_supported) {
        const auto _TailSize = _Size & (sizeof(_Simd_) - sizeof(typename _Simd_::value_type));

        if (_TailSize != 0) {
            const auto _TailMask    = _Simd_::makeTailMask(_TailSize);
            const auto _Loaded      = _Simd_::maskLoadUnaligned(_First, _TailMask);

            const auto _Compared = _Comparand.nativeEqual(_Loaded) & _TailMask;
            const auto _Mask = numeric::basic_simd_mask<_Simd_::_Generation,
                typename _Simd_::value_type>(numeric::_SimdToNativeMask<_Simd_::_Generation,
                typename _Simd_::policy_type, std::remove_cv_t<decltype(_Compared)>>(_Compared));

            if (_Mask.anyOf())
                return static_cast<const typename _Simd_::value_type*>(_First) + _Mask.countTrailingZeroBits();
        }
    }
    else {
        if (_First != _Last)
            _Last = _FindScalar(_First, _Last, _Value);
    }

    return _Last;
}

template <class _Type_>
simd_stl_declare_const_function _Type_* simd_stl_stdcall _FindVectorized(
    const void* _First,
    const void* _Last,
    _Type_      _Value) noexcept
{
    if constexpr (sizeof(_Type_) <= 2) {
        if (arch::ProcessorFeatures::AVX512BW())
            return const_cast<_Type_*>(static_cast<const volatile _Type_*>(
                _FindVectorizedInternal<simd_stl::numeric::simd256_avx512vlbw<_Type_>>(_First, _Last, _Value)));
    }
    else {
        if (arch::ProcessorFeatures::AVX512F())
            return const_cast<_Type_*>(static_cast<const volatile _Type_*>(
                _FindVectorizedInternal<simd_stl::numeric::simd256_avx512vlf<_Type_>>(_First, _Last, _Value)));
    }

    if (arch::ProcessorFeatures::AVX2())
        return const_cast<_Type_*>(static_cast<const volatile _Type_*>(
            _FindVectorizedInternal<simd_stl::numeric::simd256_avx2<_Type_>>(_First, _Last, _Value)));
    else if (arch::ProcessorFeatures::SSE2())
        return const_cast<_Type_*>(static_cast<const volatile _Type_*>(
            _FindVectorizedInternal<simd_stl::numeric::simd128_sse2<_Type_>>(_First, _Last, _Value)));


    return const_cast<_Type_*>(static_cast<const volatile _Type_*>(_FindScalar(_First, _Last, _Value)));
}

__SIMD_STL_ALGORITHM_NAMESPACE_END