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
simd_stl_declare_const_function simd_stl_always_inline bool simd_stl_stdcall _ContainsScalar(
    const void* _FirstPointer,
    const void* _LastPointer,
    _Type_      _Value) noexcept
{
    auto _Current = static_cast<const _Type_*>(_FirstPointer);

    while (_Current != _LastPointer) {
        if (*_Current == _Value)
            return true;

        ++_Current;
    }

    return false;
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Type_>
simd_stl_declare_const_function simd_stl_always_inline bool simd_stl_stdcall _ContainsVectorizedInternal(
    const void* _FirstPointer,
    const void* _LastPointer,
    _Type_      _Value) noexcept
{
    using _SimdType_ = numeric::basic_simd<_SimdGeneration_, _Type_>;

    const auto _Size = ByteLength(_FirstPointer, _LastPointer);
    const auto _AlignedSize = _Size & (~(sizeof(_SimdType_) - 1));

    if (_AlignedSize != 0) {
        const auto _Comparand = _SimdType_(_Value);

        const void* _StopAt = _FirstPointer;
        AdvanceBytes(_StopAt, _AlignedSize);

        do {
            const auto _Mask = _Comparand.maskEqual<_Type_>(_SimdType_::loadUnaligned(_FirstPointer));

            if (_Mask.unwrap() != 0)
                return true;

            AdvanceBytes(_FirstPointer, sizeof(_SimdType_));
        } while (_FirstPointer != _StopAt);
    }

    return (_FirstPointer == _LastPointer) ? false : _ContainsScalar(_FirstPointer, _LastPointer, _Value);
}

template <class _Type_>
simd_stl_declare_const_function simd_stl_always_inline bool simd_stl_stdcall _ContainsVectorized(
    const void* _FirstPointer,
    const void* _LastPointer,
    _Type_      _Value) noexcept
{
    if constexpr (sizeof(_Type_) <= 2) {
        if (arch::ProcessorFeatures::AVX512BW())
            return _ContainsVectorizedInternal<arch::CpuFeature::AVX512BW, _Type_>(_FirstPointer, _LastPointer, _Value);
    }
    else {
        if (arch::ProcessorFeatures::AVX512F())
            return _ContainsVectorizedInternal<arch::CpuFeature::AVX512F, _Type_>(_FirstPointer, _LastPointer, _Value);
    }

    if (arch::ProcessorFeatures::AVX2())
         return _ContainsVectorizedInternal<arch::CpuFeature::AVX2, _Type_>(_FirstPointer, _LastPointer, _Value);
    else if (arch::ProcessorFeatures::SSE2())
        return _ContainsVectorizedInternal<arch::CpuFeature::SSE2, _Type_>(_FirstPointer, _LastPointer, _Value);

    return _ContainsScalar(_FirstPointer, _LastPointer, _Value);
}


__SIMD_STL_ALGORITHM_NAMESPACE_END
