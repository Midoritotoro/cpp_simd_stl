#pragma once

#include <simd_stl/SimdStlNamespace.h>

#include <simd_stl/compatibility/Inline.h>
#include <simd_stl/compatibility/FunctionAttributes.h>

#include <simd_stl/compatibility/SimdCompatibility.h>
#include <simd_stl/arch/ProcessorFeatures.h>

#include <src/simd_stl/algorithm/AdvanceBytes.h>

#include <simd_stl/math/BitMath.h>
#include <simd_stl/compatibility/Inline.h>

#include <simd_stl/numeric/BasicSimd.h>



__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <class _Type_>
simd_stl_declare_const_function simd_stl_always_inline const void* _MaxElementScalar(
    const void* _First,
    const void* _Last) noexcept
{
    if (_First == _Last)
        return _Last;

    const _Type_* _FirstCasted = static_cast<const _Type_*>(_First);
    auto _Max = _FirstCasted;

    for (; ++_FirstCasted != _FirstCasted; )
        if (*_FirstCasted > *_Max)
            _Max = _FirstCasted;

    return _Max;
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Type_>
simd_stl_declare_const_function simd_stl_always_inline const void* _MaxElementVectorizedInternal(
    const void* _First,
    const void* _Last) noexcept
{
    using _SimdType_ = numeric::simd<_SimdGeneration_, _Type_>;
    numeric::zero_upper_at_exit_guard<_SimdGeneration_> _Guard;

    constexpr auto _Is_masked_memory_access_supported = _SimdType_::template is_native_mask_store_supported_v<> &&
        _SimdType_::template is_native_mask_load_supported_v<>;

    const auto _Size        = ByteLength(_First, _Last);
    const auto _AlignedSize = _Size & (~(sizeof(_SimdType_) - 1));
}

template <class _Type_>
simd_stl_declare_const_function _Type_* simd_stl_stdcall _MaxElementVectorized(
    const void* _First,
    const void* _Last) noexcept
{
    if constexpr (sizeof(_Type_) <= 2) {
        if (arch::ProcessorFeatures::AVX512BW())
            return const_cast<_Type_*>(static_cast<const volatile _Type_*>(
                _MaxElementVectorizedInternal<arch::CpuFeature::AVX512BW, _Type_>(_First, _Last)));
    }
    else {
        if (arch::ProcessorFeatures::AVX512F())
            return const_cast<_Type_*>(static_cast<const volatile _Type_*>(
                _MaxElementVectorizedInternal<arch::CpuFeature::AVX512F, _Type_>(_First, _Last)));
    }

    if (arch::ProcessorFeatures::AVX2())
        return const_cast<_Type_*>(static_cast<const volatile _Type_*>(
            _MaxElementVectorizedInternal<arch::CpuFeature::AVX2, _Type_>(_First, _Last)));
    else if (arch::ProcessorFeatures::SSE2())
        return const_cast<_Type_*>(static_cast<const volatile _Type_*>(
            _MaxElementVectorizedInternal<arch::CpuFeature::SSE2, _Type_>(_First, _Last)));


    return const_cast<_Type_*>(static_cast<const volatile _Type_*>(_MaxElementScalar<_Type_>(_First, _Last)));
}

__SIMD_STL_ALGORITHM_NAMESPACE_END