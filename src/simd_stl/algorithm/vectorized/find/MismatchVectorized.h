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

template <typename _Type_> 
simd_stl_declare_const_function simd_stl_always_inline sizetype simd_stl_stdcall _MismatchScalar(
    const void* const   _First,
    const void* const   _Second,
    sizetype&           _MismatchPosition,
    sizetype            _Size) noexcept
{
    const _Type_* const _FirstPointer    = static_cast<const _Type_* const>(_First);
    const _Type_* const _SecondPointer   = static_cast<const _Type_* const>(_Second);

    _MismatchPosition /= sizeof(_Type_);

    for (; _MismatchPosition < (_Size / sizeof(_Type_)); ++_MismatchPosition)
        if (*(_FirstPointer + _MismatchPosition) != *(_SecondPointer + _MismatchPosition))
            return _MismatchPosition;
   
    return _MismatchPosition;
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Type_>
simd_stl_declare_const_function simd_stl_always_inline sizetype _MismatchVectorizedInternal(
    const void* const   _First,
    const void* const   _Second,
    const sizetype      _Size) noexcept
{
    using _SimdType_ = numeric::basic_simd<_SimdGeneration_, _Type_>;
    numeric::zero_upper_at_exit_guard<_SimdGeneration_> _Guard;

    const auto _Bytes        = _Size * sizeof(_Type_);

    auto _AlignedSize        = (_Bytes & (~sizetype(sizeof(_SimdType_) - 1)));
    auto _MismatchPosition   = sizetype(0);

    for (; _MismatchPosition < _AlignedSize; _MismatchPosition += sizeof(_SimdType_)) {
        const auto _LoadedFirst  = _SimdType_::loadUnaligned(static_cast<const char* const>(_First) + _MismatchPosition);
        const auto _LoadedSecond = _SimdType_::loadUnaligned(static_cast<const char* const>(_Second) + _MismatchPosition);

        const auto _Mask = _LoadedFirst.maskEqual(_LoadedSecond);

        if (_Mask.allOf() == false)
            return (_MismatchPosition / sizeof(_Type_)) + _Mask.countTrailingOneBits();
    }

    return (_MismatchPosition == _Bytes)
        ? (_MismatchPosition / sizeof(_Type_))
        : _MismatchScalar<_Type_>(_First, _Second, _MismatchPosition, _Bytes);
}

template <typename _Type_>
simd_stl_declare_const_function sizetype simd_stl_stdcall _MismatchVectorized(
    const void* const   _First,
    const void* const   _Second,
    const sizetype      _Size) noexcept
{
    if constexpr (sizeof(_Type_) <= 2) {
        if (arch::ProcessorFeatures::AVX512BW())
            return _MismatchVectorizedInternal<arch::CpuFeature::AVX512BW, _Type_>(_First, _Second, _Size);
    }
    else {
        if (arch::ProcessorFeatures::AVX512F())
            return _MismatchVectorizedInternal<arch::CpuFeature::AVX512F, _Type_>(_First, _Second, _Size);
    }

    if (arch::ProcessorFeatures::AVX2())
        return _MismatchVectorizedInternal<arch::CpuFeature::AVX2, _Type_>(_First, _Second, _Size);
    else if (arch::ProcessorFeatures::SSE2())
        return _MismatchVectorizedInternal<arch::CpuFeature::SSE2, _Type_>(_First, _Second, _Size);

    auto _MismatchPosition = sizetype(0);
    return _MismatchScalar<_Type_>(_First, _Second, _MismatchPosition, _Size);
}

__SIMD_STL_ALGORITHM_NAMESPACE_END
