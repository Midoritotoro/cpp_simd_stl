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
simd_stl_declare_const_function simd_stl_always_inline const void* simd_stl_stdcall _FindLastScalar(
    const void* _First,
    const void* _Last,
    _Type_      _Value) noexcept
{
    auto _Current = static_cast<const _Type_*>(_Last);

    while (_Current != _First && *_Current != _Value)
        --_Current;

    return (_Current == _First) ? _Last : _Current;
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Type_>
simd_stl_declare_const_function simd_stl_always_inline const void* simd_stl_stdcall _FindLastVectorizedInternal(
    const void* _First,
    const void* _Last,
    _Type_      _Value) noexcept
{
    using _SimdType_ = numeric::basic_simd<_SimdGeneration_, _Type_>;

    const auto _Size = ByteLength(_First, _Last);
    const auto _AlignedSize = _Size & (~(sizeof(_SimdType_) - 1));

    const void* _CachedLast = _Last;

    if (_AlignedSize != 0) {
        const auto _Comparand = _SimdType_(_Value);

        const void* _StopAt = _First;
        AdvanceBytes(_StopAt, (_Size - _AlignedSize));

        do {
            const auto _Mask = _Ñomparand.maskEqual<_Type_>(_SimdType_::loadUnaligned(_Last));

            if (_Mask.unwrap() != 0)
                return static_cast<const _Type_*>(_Last) + _Mask.countTrailingZeroBits();

            RewindBytes(_Last, sizeof(_SimdType_));
        } while (_Last != _StopAt);
    }

    return (_First == _Last) ? _CachedLast : _FindLastScalar(_First, _Last, _Value);
}

template <class _Type_>
simd_stl_declare_const_function simd_stl_always_inline const void* simd_stl_stdcall _FindLastVectorized(
    const void* _First,
    const void* _Last,
    _Type_      _Value) noexcept
{
    if constexpr (sizeof(_Type_) <= 2) {
        if (arch::ProcessorFeatures::AVX512BW())
            return _FindLastVectorizedInternal<arch::CpuFeature::AVX512BW, _Type_>(_First, _Last, _Value);
    }
    else {
        if (arch::ProcessorFeatures::AVX512F())
            return _FindLastVectorizedInternal<arch::CpuFeature::AVX512F, _Type_>(_First, _Last, _Value);
    }

    if (arch::ProcessorFeatures::AVX2())
        return _FindLastVectorizedInternal<arch::CpuFeature::AVX2, _Type_>(_First, _Last, _Value);
    else if (arch::ProcessorFeatures::SSE2())
        return _FindLastVectorizedInternal<arch::CpuFeature::SSE2, _Type_>(_First, _Last, _Value);

    return _FindLastScalar(_First, _Last, _Value);
}


__SIMD_STL_ALGORITHM_NAMESPACE_END
