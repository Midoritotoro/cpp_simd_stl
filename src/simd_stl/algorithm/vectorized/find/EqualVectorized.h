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
simd_stl_declare_const_function simd_stl_always_inline bool simd_stl_stdcall _EqualScalar(
    const void* _First,
    const void* _Second,
    sizetype&   _ByteOffset,
    sizetype    _Size) noexcept
{
    const _Type_* _FirstPointer    = static_cast<const _Type_*>(_First);
    const _Type_* _SecondPointer   = static_cast<const _Type_*>(_Second);

    _ByteOffset /= sizeof(_Type_);

    for (; _ByteOffset < (_Size / sizeof(_Type_)); ++_ByteOffset)
        if (*(_FirstPointer + _ByteOffset) != *(_SecondPointer + _ByteOffset))
            return false;
   
    return true;
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Type_>
simd_stl_declare_const_function simd_stl_always_inline bool simd_stl_stdcall _EqualVectorizedInternal(
    const void*     _First,
    const void*     _Second,
    const sizetype  _Size) noexcept
{
    using _SimdType_ = numeric::basic_simd<_SimdGeneration_, _Type_>;
    numeric::zero_upper_at_exit_guard<_SimdGeneration_> _Guard;

    const auto _Bytes   = _Size * sizeof(_Type_);

    auto _AlignedSize   = (_Bytes & (~(sizeof(_SimdType_) - 1)));
    auto _Offset        = sizetype(0);

    for (; _Offset < _AlignedSize; _Offset += sizeof(_SimdType_)) {
        const auto _LoadedFirst  = _SimdType_::loadUnaligned(static_cast<const char*>(_First) + _Offset);
        const auto _LoadedSecond = _SimdType_::loadUnaligned(static_cast<const char*>(_Second) + _Offset);

        const auto _ComparedMask = _LoadedFirst.maskEqual(_LoadedSecond);

        if (_ComparedMask.allOf() == false)
            return false;
    }

    return (_Offset == _Bytes) ? true : _EqualScalar<_Type_>(_First, _Second, _Offset, _Bytes);
}


template <typename _Type_>
simd_stl_declare_const_function bool simd_stl_stdcall _EqualVectorized(
    const void*     _First,
    const void*     _Second,
    const sizetype  _Size) noexcept
{
    if constexpr (sizeof(_Type_) <= 2) {
        if (arch::ProcessorFeatures::AVX512BW())
            return _EqualVectorizedInternal<arch::CpuFeature::AVX512BW, _Type_>(_First, _Second, _Size);
    }
    else {
        if (arch::ProcessorFeatures::AVX512F())
            return _EqualVectorizedInternal<arch::CpuFeature::AVX512F, _Type_>(_First, _Second, _Size);
    }

    if (arch::ProcessorFeatures::AVX2())
        return _EqualVectorizedInternal<arch::CpuFeature::AVX2, _Type_>(_First, _Second, _Size);
    else if (arch::ProcessorFeatures::SSE2())
        return _EqualVectorizedInternal<arch::CpuFeature::SSE2, _Type_>(_First, _Second, _Size);

    auto _Offset = sizetype(0);
    return _EqualScalar<_Type_>(_First, _Second, _Offset, _Size * sizeof(_Type_));
}

__SIMD_STL_ALGORITHM_NAMESPACE_END
