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
simd_stl_declare_const_function simd_stl_always_inline sizetype MismatchScalar(
    const void* const   firstRangeBegin,
    const void* const   secondRangeBegin,
    sizetype&           mismatchPosition,
    sizetype            size) noexcept
{
    const _Type_* const firstPointer    = static_cast<const _Type_* const>(firstRangeBegin);
    const _Type_* const secondPointer   = static_cast<const _Type_* const>(secondRangeBegin);

    mismatchPosition /= sizeof(_Type_);

    for (; mismatchPosition < (size / sizeof(_Type_)); ++mismatchPosition)
        if (*(firstPointer + mismatchPosition) != *(secondPointer + mismatchPosition))
            return mismatchPosition;
   
    return mismatchPosition;
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Type_>
simd_stl_declare_const_function simd_stl_always_inline sizetype MismatchVectorizedInternal(
    const void* const   firstRangeBegin,
    const void* const   secondRangeBegin,
    const sizetype      size) noexcept
{
    using _SimdType_ = numeric::basic_simd<_SimdGeneration_, _Type_>;

    auto alignedSize        = (size & (~sizetype(_SimdType_::width() - 1)));
    auto mismatchPosition   = sizetype(0);

    if (alignedSize != 0) {
        for (; mismatchPosition < alignedSize; mismatchPosition += _SimdType_::width()) {
            const auto loadedFirst  = _SimdType_::loadUnaligned(
                static_cast<const char* const>(firstRangeBegin) + mismatchPosition);

            const auto loadedSecond = _SimdType_::loadUnaligned(
                static_cast<const char* const>(secondRangeBegin) + mismatchPosition);

            const auto comparedMask = loadedFirst.maskEqual(loadedSecond);

            if ((~comparedMask.unwrap()) != 0) {
                mismatchPosition += comparedMask.countTrailingZeroBits();
                return mismatchPosition / sizeof(_Type_);
            }
        }
    }

    return (mismatchPosition == size)
        ? mismatchPosition
        : MismatchScalar<_Type_>(firstRangeBegin, secondRangeBegin, mismatchPosition, size - mismatchPosition);
}


template <typename _Type_>
simd_stl_declare_const_function simd_stl_always_inline sizetype MismatchVectorized(
    const void* const   firstRangeBegin,
    const void* const   secondRangeBegin,
    const sizetype      size) noexcept
{
    
    if (arch::ProcessorFeatures::SSE2())
        return MismatchVectorizedInternal<arch::CpuFeature::SSE2, _Type_>(firstRangeBegin, secondRangeBegin, size);
}

__SIMD_STL_ALGORITHM_NAMESPACE_END
