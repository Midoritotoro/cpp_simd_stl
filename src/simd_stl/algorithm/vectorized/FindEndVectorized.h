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
simd_stl_declare_const_function simd_stl_always_inline const void* FindEndScalar(
    const void*     firstPointer,
    const sizetype  firstRangeLength,
    const void*     secondPointer,
    const sizetype  secondRangeLength) noexcept
{
    const _Type_* haystack = static_cast<const _Type_*>(firstPointer);
    const _Type_* needle   = static_cast<const _Type_*>(secondPointer);

    if (secondRangeLength == 0)
        return haystack + firstRangeLength;

    if (firstRangeLength < secondRangeLength)
        return nullptr;

    for (const _Type_* iterator = haystack + (firstRangeLength - secondRangeLength);
        iterator >= haystack; --iterator)
    {
        bool match = true;
        for (sizetype j = 0; j < secondRangeLength; ++j) {
            if (iterator[j] != needle[j]) {
                match = false;
                break;
            }
        }

        if (match)
            return iterator;

        if (iterator == haystack)
            break;
    }

    return nullptr;
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Type_,
    sizetype            _NeedleLength_,
    typename            _MemCmpLike_>
simd_stl_declare_const_function simd_stl_always_inline const _Type_* FindEndVectorizedInternalFixedSize(
    const void*     firstPointer,
    const sizetype  firstRangeLength,
    const void*     secondPointer,
    _MemCmpLike_    memcmpLike) noexcept
{
    using _SimdType_ = numeric::basic_simd<_SimdGeneration_, _Type_>;
     
    if constexpr ((_NeedleLength_ * sizeof(_Type_)) > sizeof(_SimdType_))
        return FindEndScalar<_Type_>(
            firstPointer, firstRangeLength, secondPointer, _NeedleLength_);

    const auto firstRangeByteLength = firstRangeLength * sizeof(_Type_);
    const auto alignedFirstRangeByteLength = (firstRangeByteLength & (~(_SimdType_::width() - 1)));

    const auto needleBeginComparand = _SimdType_(static_cast<const _Type_*>(secondPointer)[0]);
    const auto needleEndComparand   = _SimdType_(static_cast<const _Type_*>(secondPointer)[_NeedleLength_ - 1]);

    const void* firstRangeEnd           = static_cast<const char*>(firstPointer) + firstRangeByteLength;
    const void* firstRangeAlignedLimit  = firstRangeEnd;

    RewindBytes(firstRangeAlignedLimit, alignedFirstRangeByteLength);

    if (alignedFirstRangeByteLength != 0) {
        do {
            const auto firstLoaded  = _SimdType_::loadUnaligned(firstRangeEnd);
            const auto secondLoaded = _SimdType_::loadUnaligned(static_cast<const _Type_*>(firstRangeEnd) - _NeedleLength);

            const auto needleEndMask = needleEndComparand.maskEqual(firstLoaded);
            const auto needleBeginMask = needleBeginComparand.maskEqual(secondLoaded);

            auto mask = needleEndMask & needleBeginMask;

            // needle_needle_needle
            // needle 

            // eeeeeeeeeeeeeeee & le_needle_needle
            // 0101100100110001

            // nnnnnnnnnnnnnnnn & needle_needle_ne
            // 1000000100000010

            // 0101100100110001 & 
            // 1000000100000010
            // 0000000100110000

            while (mask.anyOf()) {
                const auto trailingZeros = mask.countTrailingZeroBits();

                const auto mainRangeBegin = static_cast<const char*>(firstRangeEnd) - trailingZeros - sizeof(_Type_) - (_NeedleLength_ * sizeof(_Type_));

                if (memcmpLike(mainRangeBegin, static_cast<const char*>(secondPointer) + sizeof(_Type_)))
                    return reinterpret_cast<const _Type_*>(static_cast<const char*>(firstRangeEnd) - trailingZeros - (_NeedleLength_ * sizeof(_Type_)));

                mask.clearLeftMostSet();
            }

            RewindBytes(firstRangeEnd, _SimdType_::width());
        } while (firstRangeAlignedLimit != firstRangeEnd);
    }

    return (firstRangeEnd == firstPointer)
        ? nullptr
        : FindEndScalar<_Type_>(
            firstPointer, firstRangeByteLength - alignedFirstRangeByteLength,
            secondPointer, _NeedleLength_);
}


template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Type_>
simd_stl_declare_const_function simd_stl_always_inline const void* FindEndVectorizedInternal(
    const void*     firstPointer,
    const sizetype  firstRangeLength,
    const void*     secondPointer,
    const sizetype  secondRangeLength) noexcept
{
   

}

template <class _Type_>
simd_stl_declare_const_function simd_stl_always_inline const void* FindEndVectorized(
    const void*     firstPointer,
    const sizetype  firstRangeLength,
    const void*     secondPointer,
    const sizetype  secondRangeLength) noexcept
{
    /* if (arch::ProcessorFeatures::AVX512F())
         return FindVectorizedInternal<arch::CpuFeature::AVX512F, _Type_>(firstPointer, lastPointer, value);
     else if (arch::ProcessorFeatures::AVX2())
         return FindVectorizedInternal<arch::CpuFeature::AVX2, _Type_>(firstPointer, lastPointer, value);*/
    /*else*/ if (arch::ProcessorFeatures::SSE2())
        return FindEndVectorizedInternal<arch::CpuFeature::SSE2, _Type_>(
            firstPointer, firstRangeLength, secondPointer, secondRangeLength);

    return FindEndScalar(firstPointer, lastPointer, value);
}


__SIMD_STL_ALGORITHM_NAMESPACE_END
