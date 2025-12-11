#pragma once

#include <simd_stl/SimdStlNamespace.h>

#include <simd_stl/compatibility/Inline.h>
#include <simd_stl/compatibility/FunctionAttributes.h>

#include <simd_stl/compatibility/SimdCompatibility.h>
#include <simd_stl/arch/ProcessorFeatures.h>

#include <src/simd_stl/algorithm/AdvanceBytes.h>

#include <simd_stl/math/BitMath.h>
#include <simd_stl/numeric/BasicSimd.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <class _Type_>
simd_stl_declare_const_function simd_stl_always_inline const void* _RemoveScalar(
    void*       _First,
    const void* _Current,
    const void* _Last,
    _Type_      _Value) noexcept
{
    auto _CurrentCasted  = static_cast<const _Type_*>(_Current);
    auto _FirstCasted    = static_cast<_Type_*>(_First);

    for (; _CurrentCasted != _Last; ++_CurrentCasted) {
        const auto _CurrentValue = *_CurrentCasted;

        if (_CurrentValue != _Value)
            *_FirstCasted++ = _CurrentValue;
    }

    return _FirstCasted;
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Type_>
simd_stl_always_inline const void* _RemoveVectorizedInternal(
    void*       _First,
    const void* _Last,
    _Type_      _Value) noexcept
{
    using _SimdType_ = numeric::basic_simd<_SimdGeneration_, _Type_>;
    const auto _AlignedSize  = ByteLength(_First, _Last) & (~(sizeof(_SimdType_) - 1));

    void* _Current = _First;

    if (_AlignedSize != 0) {
        const auto _Comparand = _SimdType_(_Value);

        const void* _StopAt = _First;
        AdvanceBytes(_StopAt, _AlignedSize);

        do {
            const auto _Loaded = _SimdType_::loadUnaligned(_Current);
            const auto _Mask = _Comparand.maskEqual(_Loaded);

            _First = _Loaded.compressStoreUnaligned(_First, _Mask.unwrap());
            AdvanceBytes(_Current, sizeof(_SimdType_));
        } while (_Current != _StopAt);
    }

    return (_Current == _Last) ? _First : _RemoveScalar<_Type_>(_First, _Current, _Last, _Value);
}

template <class _Type_>
simd_stl_declare_const_function simd_stl_always_inline _Type_* _RemoveVectorized(
    void*       _First,
    const void* _Last,
    _Type_      _Value) noexcept
{
    if constexpr (sizeof(_Type_) <= 2) {
        if (arch::ProcessorFeatures::AVX512BW())
            return const_cast<_Type_*>(static_cast<const _Type_*>(
                _RemoveVectorizedInternal<arch::CpuFeature::AVX512BW, _Type_>(_First, _Last, _Value)));
    }
    else {
        if (arch::ProcessorFeatures::AVX512F())
            return const_cast<_Type_*>(static_cast<const _Type_*>(
                _RemoveVectorizedInternal<arch::CpuFeature::AVX512F, _Type_>(_First, _Last, _Value)));
    }

    if (arch::ProcessorFeatures::AVX2())
        return const_cast<_Type_*>(static_cast<const _Type_*>(
            _RemoveVectorizedInternal<arch::CpuFeature::AVX2, _Type_>(_First, _Last, _Value)));
    else if (arch::ProcessorFeatures::SSSE3())
        return const_cast<_Type_*>(static_cast<const _Type_*>(
            _RemoveVectorizedInternal<arch::CpuFeature::SSSE3, _Type_>(_First, _Last, _Value)));

    return const_cast<_Type_*>(static_cast<const _Type_*>(_RemoveScalar<_Type_>(_First, _First, _Last, _Value)));
}


__SIMD_STL_ALGORITHM_NAMESPACE_END
