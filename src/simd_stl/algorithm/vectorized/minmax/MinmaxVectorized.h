#pragma once

#include <simd_stl/numeric/BasicSimd.h>


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

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Type_>
simd_stl_declare_const_function simd_stl_always_inline std::pair<_Type_, _Type_> _MinmaxVectorizedInternal(
    const void* _First,
    const void* _Last) noexcept
{
    using _SimdType_ = numeric::simd<_SimdGeneration_, _Type_>;
    numeric::zero_upper_at_exit_guard<_SimdGeneration_> _Guard;

    constexpr auto _Is_masked_memory_access_supported = _SimdType_::template is_native_mask_store_supported_v<> &&
        _SimdType_::template is_native_mask_load_supported_v<>;

    const auto _Size = ByteLength(_First, _Last);
    const auto _AlignedSize = _Size & (~(sizeof(_SimdType_) - 1));

    auto _MaximumValues = _SimdType_();
    auto _MinimumValues = _SimdType_();

    _MaximumValues.broadcastZeros();
    _MinimumValues.broadcastZeros();

    const auto _HasAlignedPart = (_AlignedSize != 0);

    if (_HasAlignedPart) {
        const void* _StopAt = _First;
        AdvanceBytes(_StopAt, _AlignedSize);

        _MaximumValues = _SimdType_::loadUnaligned(_First);
        AdvanceBytes(_First, sizeof(_SimdType_));

        while (_First != _StopAt) {
            const auto _Loaded = _SimdType_::loadUnaligned(_First);
            AdvanceBytes(_First, sizeof(_SimdType_));

            _MaximumValues = _MaximumValues.verticalMax(_Loaded);
            _MinimumValues = _MinimumValues.verticalMin(_Loaded);
        }
    }

    const auto _TailSize = _Size & (sizeof(_SimdType_) - sizeof(_Type_));
    
    if (_TailSize != 0) {
        if constexpr (_Is_masked_memory_access_supported) {
            const auto _TailMask = _SimdType_::makeTailMask(_TailSize);
            const auto _Loaded = _SimdType_::maskLoadUnaligned(_First, _TailMask);

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
                const auto _Minmax = _MinmaxScalar<_Type_>(_First, _Last);

                const auto _HorizontalMax = _MaximumValues.horizontalMax();
                const auto _HorizontalMin = _MinimumValues.horizontalMin();
                
                return std::pair { 
                    std::min(_Minmax.first, _MinimumValues.horizontalMin()),
                    std::max(_Minmax.second, _MaximumValues.horizontalMax())
                };
            }
            else {
                return _MinmaxScalar<_Type_>(_First, _Last);
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
        if (arch::ProcessorFeatures::AVX512BW())
            return _MinmaxVectorizedInternal<arch::CpuFeature::AVX512BW, _Type_>(_First, _Last);
    }
    else {
        if (arch::ProcessorFeatures::AVX512F())
            return _MinmaxVectorizedInternal<arch::CpuFeature::AVX512F, _Type_>(_First, _Last);
    }

    if (arch::ProcessorFeatures::AVX2())
        return _MinmaxVectorizedInternal<arch::CpuFeature::AVX2, _Type_>(_First, _Last);
    else if (arch::ProcessorFeatures::SSE41())
        return _MinmaxVectorizedInternal<arch::CpuFeature::SSE41, _Type_>(_First, _Last);
    else if (arch::ProcessorFeatures::SSSE3())
        return _MinmaxVectorizedInternal<arch::CpuFeature::SSSE3, _Type_>(_First, _Last);
    else if (arch::ProcessorFeatures::SSE2())
        return _MinmaxVectorizedInternal<arch::CpuFeature::SSE2, _Type_>(_First, _Last);

    return _MinmaxScalar<_Type_>(_First, _Last);
}

__SIMD_STL_ALGORITHM_NAMESPACE_END