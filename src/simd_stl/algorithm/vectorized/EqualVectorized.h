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
simd_stl_declare_const_function simd_stl_always_inline bool EqualScalar(
    const void* const   firstRangeBegin,
    const void* const   secondRangeBegin,
    sizetype&           byteOffset,
    sizetype            size) noexcept
{
    const _Type_* const firstPointer    = static_cast<const _Type_* const>(firstRangeBegin);
    const _Type_* const secondPointer   = static_cast<const _Type_* const>(secondRangeBegin);

    byteOffset /= sizeof(_Type_);

    for (; byteOffset < (size / sizeof(_Type_)); ++byteOffset)
        if (*(firstPointer + byteOffset) != *(secondPointer + byteOffset))
            return false;
   
    return true;
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Type_>
simd_stl_declare_const_function simd_stl_always_inline bool EqualVectorizedInternal(
    const void* const   firstRangeBegin,
    const void* const   secondRangeBegin,
    const sizetype      size) noexcept
{
    using _SimdType_ = numeric::basic_simd<_SimdGeneration_, _Type_>;

    const auto bytes        = size * sizeof(_Type_);

    auto alignedSize        = (bytes & (~sizetype(_SimdType_::width() - 1)));
    auto byteOffset         = sizetype(0);

    if (alignedSize != 0) {
        for (; byteOffset < alignedSize; byteOffset += _SimdType_::width()) {
            const auto loadedFirst  = _SimdType_::loadUnaligned(static_cast<const char* const>(firstRangeBegin) + byteOffset);
            const auto loadedSecond = _SimdType_::loadUnaligned(static_cast<const char* const>(secondRangeBegin) + byteOffset);

            const auto comparedMask = loadedFirst.maskEqual(loadedSecond);

            if (comparedMask.allOf() == false)
                return false;
        }
    }

    return (byteOffset == bytes)
        ? true
        : EqualScalar<_Type_>(firstRangeBegin, secondRangeBegin, byteOffset, bytes);
}


template <typename _Type_>
simd_stl_declare_const_function simd_stl_always_inline bool EqualVectorized(
    const void* const   firstRangeBegin,
    const void* const   secondRangeBegin,
    const sizetype      size) noexcept
{
    
    if (arch::ProcessorFeatures::SSE2())
        return EqualVectorizedInternal<arch::CpuFeature::SSE2, _Type_>(firstRangeBegin, secondRangeBegin, size);

    auto byteOffset = sizetype(0);
    return EqualScalar<_Type_>(firstRangeBegin, secondRangeBegin, byteOffset, size * sizeof(_Type_));
}

__SIMD_STL_ALGORITHM_NAMESPACE_END
