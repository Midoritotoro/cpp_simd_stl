#pragma once

#include <simd_stl/numeric/Simd.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <class _Type_>
simd_stl_declare_const_function simd_stl_always_inline std::pair<_Type_, _Type_> _MinmaxScalar(
    const void* _First,
    const void* _Last) noexcept
{
    const _Type_* _FirstCasted = static_cast<const _Type_*>(_First);

    auto _Max = _FirstCasted;
    auto _Min = _FirstCasted;

    for (; ++_FirstCasted != _Last; ) {
        if (*_FirstCasted > *_Max)
            _Max = _FirstCasted;
        if (*_FirstCasted < *_Min)
            _Min = _FirstCasted;
    }

    return { *_Min, *_Max };
}

template <class _Simd_>
simd_stl_declare_const_function simd_stl_always_inline std::pair<typename _Simd_::value_type, typename _Simd_::value_type> _MinmaxVectorizedInternal(
    const void* _First,
    const void* _Last) noexcept
{
    numeric::zero_upper_at_exit_guard<_Simd_::__generation> _Guard;

    constexpr auto _Is_masked_memory_access_supported = _Simd_::template is_native_mask_store_supported_v<> &&
        _Simd_::template is_native_mask_load_supported_v<>;

    const auto _Size = __byte_length(_First, _Last);
    const auto _AlignedSize = _Size & (~(sizeof(_Simd_) - 1));

    auto _MaximumValues = _Simd_();
    auto _MinimumValues = _Simd_();

    _MaximumValues.broadcastZeros();
    _MinimumValues.broadcastZeros();

    const auto _HasAlignedPart = (_AlignedSize != 0);

    if (_HasAlignedPart) {
        const void* _StopAt = _First;
        __advance_bytes(_StopAt, _AlignedSize);

        _MaximumValues = _Simd_::loadUnaligned(_First);
        __advance_bytes(_First, sizeof(_Simd_));

        while (_First != _StopAt) {
            const auto _Loaded = _Simd_::loadUnaligned(_First);

            _MaximumValues = _MaximumValues.verticalMax(_Loaded);
            _MinimumValues = _MinimumValues.verticalMin(_Loaded);

            __advance_bytes(_First, sizeof(_Simd_));
        }
    }

    const auto _TailSize = _Size & (sizeof(_Simd_) - sizeof(typename _Simd_::value_type));
    
    if (_TailSize != 0) {
        if constexpr (_Is_masked_memory_access_supported) {
            const auto _TailMask = _Simd_::makeTailMask(_TailSize);
            const auto _Loaded = _Simd_::maskLoadUnaligned(_First, _TailMask);

            if (_HasAlignedPart) {
                _MaximumValues = _MaximumValues.verticalMax(_Loaded);
                _MinimumValues = _MinimumValues.verticalMin(_Loaded);
            }
            else {
                _MaximumValues = _Loaded;
                _MinimumValues = _Loaded;
            }
        }
        else {
            if (_HasAlignedPart) {
                const auto _Minmax = _MinmaxScalar<typename _Simd_::value_type>(_First, _Last);

                const auto _HorizontalMax = _MaximumValues.horizontalMax();
                const auto _HorizontalMin = _MinimumValues.horizontalMin();
                
                return std::pair { 
                    std::min(_Minmax.first, _MinimumValues.horizontalMin()),
                    std::max(_Minmax.second, _MaximumValues.horizontalMax())
                };
            }
            else {
                return _MinmaxScalar<typename _Simd_::value_type>(_First, _Last);
            }
        }
    }

    return { _MinimumValues.horizontalMin(), _MaximumValues.horizontalMax() };
}

template <class _Type_>
simd_stl_declare_const_function std::pair<_Type_, _Type_> simd_stl_stdcall _MinmaxVectorized(
    const void* _First,
    const void* _Last) noexcept
{
    if constexpr (sizeof(_Type_) <= 2) {
        if (arch::ProcessorFeatures::AVX512VL() && arch::ProcessorFeatures::AVX512BW())
            return _MinmaxVectorizedInternal<numeric::simd256_avx512vlbw<_Type_>>(_First, _Last);
    }
    else {
        if (arch::ProcessorFeatures::AVX512VL() && arch::ProcessorFeatures::AVX512F())
            return _MinmaxVectorizedInternal<numeric::simd256_avx512vlf<_Type_>>(_First, _Last);
    }

    if (arch::ProcessorFeatures::AVX2())
        return _MinmaxVectorizedInternal<numeric::simd256_avx2<_Type_>>(_First, _Last);
    else if (arch::ProcessorFeatures::SSE41())
        return _MinmaxVectorizedInternal<numeric::simd128_sse41<_Type_>>(_First, _Last);
    else if (arch::ProcessorFeatures::SSSE3())
        return _MinmaxVectorizedInternal<numeric::simd128_ssse3<_Type_>>(_First, _Last);
    else if (arch::ProcessorFeatures::SSE2())
        return _MinmaxVectorizedInternal<numeric::simd128_sse2<_Type_>>(_First, _Last);

    return _MinmaxScalar<_Type_>(_First, _Last);
}

__SIMD_STL_ALGORITHM_NAMESPACE_END