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
#include <simd_stl/algorithm/find/FindLast.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <class _Type_>
simd_stl_declare_const_function simd_stl_always_inline const _Type_* FindEndScalar(
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

    return static_cast<const _Type_*>(firstPointer) + firstRangeLength;
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Type_>
simd_stl_declare_const_function simd_stl_always_inline const _Type_* FindEndVectorizedInternalAnySize(
    const void*     firstPointer,
    const sizetype  firstRangeLength,
    const void*     secondPointer,
    const sizetype  secondRangeLength) noexcept
{
    using _SimdType_ = numeric::basic_simd<_SimdGeneration_, _Type_>;
    return std::find_end(static_cast<const _Type_*>(firstPointer), static_cast<const _Type_*>(firstPointer) + firstRangeLength,
        static_cast<const _Type_*>(secondPointer), static_cast<const _Type_*>(secondPointer) + secondRangeLength);
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
    numeric::zero_upper_at_exit_guard<_SimdGeneration_> _Guard;

    if constexpr ((_NeedleLength_ * sizeof(_Type_)) > sizeof(_SimdType_))
        return FindEndScalar<_Type_>(
            firstPointer, firstRangeLength, secondPointer, _NeedleLength_);

    const auto firstRangeByteLength         = firstRangeLength * sizeof(_Type_);
    const auto alignedFirstRangeByteLength  = (firstRangeByteLength & (~(_SimdType_::width() - 1)));

    const auto needleBeginComparand = _SimdType_(static_cast<const _Type_*>(secondPointer)[0]);
    const auto needleEndComparand   = _SimdType_(static_cast<const _Type_*>(secondPointer)[_NeedleLength_ - 1]);

    const void* firstRangeEnd           = static_cast<const char*>(firstPointer) + (firstRangeByteLength);
    const void* firstRangeAlignedLimit  = firstRangeEnd;

    RewindBytes(firstRangeAlignedLimit, alignedFirstRangeByteLength);

    if (alignedFirstRangeByteLength != 0) {
        do {
            const auto loadedEnd    = _SimdType_::loadUnaligned(static_cast<const char*>(firstRangeEnd) - _SimdType_::width());
            const auto loadedBegin  = _SimdType_::loadUnaligned(static_cast<const char*>(firstRangeEnd) - _SimdType_::width() - (_NeedleLength_ * sizeof(_Type_)));

            const auto needleEndMask = needleEndComparand.maskEqual(loadedEnd);
            const auto needleBeginMask = needleBeginComparand.maskEqual(loadedBegin);

            auto mask = needleEndMask & needleBeginMask;

            while (mask.anyOf()) {
                const auto trailingZeros = mask.countTrailingZeroBits();

                const auto mainRangeBegin = static_cast<const char*>(firstRangeEnd) + trailingZeros - _SimdType_::width() - (_NeedleLength_ * sizeof(_Type_)) + sizeof(_Type_);

                if (memcmpLike(mainRangeBegin, secondPointer))
                    return reinterpret_cast<const _Type_*>(static_cast<const char*>(firstRangeEnd));

                mask.clearLeftMostSetBit();
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
simd_stl_declare_const_function simd_stl_always_inline const _Type_* FindEndVectorizedInternal(
    const void*     firstPointer,
    const sizetype  firstRangeLength,
    const void*     secondPointer,
    const sizetype  secondRangeLength) noexcept
{
    const auto needleSizeInBytes = secondRangeLength * sizeof(_Type_);

    switch (needleSizeInBytes) {
        case 0: 
            return static_cast<const _Type_*>(firstPointer) + firstRangeLength;

        case 1: 
            return simd_stl::algorithm::find_last(
            static_cast<const _Type_*>(firstPointer), static_cast<const _Type_*>(firstPointer) + firstRangeLength, 
            *static_cast<const _Type_*>(secondPointer));

        case 2: 
            return FindEndVectorizedInternalFixedSize<_SimdGeneration_, _Type_, 2>(
                firstPointer, firstRangeLength, secondPointer, _Choose_fixed_memcmp_function<2>::value);

        case 3: 
            return FindEndVectorizedInternalFixedSize<_SimdGeneration_, _Type_, 3>(
                firstPointer, firstRangeLength, secondPointer, _Choose_fixed_memcmp_function<3>::value);

        case 4: 
            return FindEndVectorizedInternalFixedSize<_SimdGeneration_, _Type_, 4>(
                firstPointer, firstRangeLength, secondPointer, _Choose_fixed_memcmp_function<4>::value);

        case 5: 
            return FindEndVectorizedInternalFixedSize<_SimdGeneration_, _Type_, 5>(
                firstPointer, firstRangeLength, secondPointer, _Choose_fixed_memcmp_function<5>::value);

        case 6: 
            return FindEndVectorizedInternalFixedSize<_SimdGeneration_, _Type_, 6>(
                firstPointer, firstRangeLength, secondPointer, _Choose_fixed_memcmp_function<6>::value);

        case 7: 
            return FindEndVectorizedInternalFixedSize<_SimdGeneration_, _Type_, 7>(
                firstPointer, firstRangeLength, secondPointer, _Choose_fixed_memcmp_function<7>::value);

        case 8: 
            return FindEndVectorizedInternalFixedSize<_SimdGeneration_, _Type_, 8>(
                firstPointer, firstRangeLength, secondPointer, _Choose_fixed_memcmp_function<8>::value);

        case 9: 
            return FindEndVectorizedInternalFixedSize<_SimdGeneration_, _Type_, 9>(
                firstPointer, firstRangeLength, secondPointer, _Choose_fixed_memcmp_function<9>::value);

        case 10: 
            return FindEndVectorizedInternalFixedSize<_SimdGeneration_, _Type_, 10>(
                firstPointer, firstRangeLength, secondPointer, _Choose_fixed_memcmp_function<10>::value);

        case 11: 
            return FindEndVectorizedInternalFixedSize<_SimdGeneration_, _Type_, 11>(
                firstPointer, firstRangeLength, secondPointer, _Choose_fixed_memcmp_function<11>::value);

        case 12: 
            return FindEndVectorizedInternalFixedSize<_SimdGeneration_, _Type_, 12>(
                firstPointer, firstRangeLength, secondPointer, _Choose_fixed_memcmp_function<12>::value);

        default: 
            return FindEndVectorizedInternalAnySize<_SimdGeneration_, _Type_>(
                firstPointer, firstRangeLength, secondPointer, secondRangeLength);

    }

    AssertUnreachable();
    return nullptr;
}

template <class _Type_>
simd_stl_declare_const_function simd_stl_always_inline const _Type_* FindEndVectorized(
    const void*     firstPointer,
    const sizetype  firstRangeLength,
    const void*     secondPointer,
    const sizetype  secondRangeLength) noexcept
{
    if (arch::ProcessorFeatures::SSE2())
        return FindEndVectorizedInternal<arch::CpuFeature::SSE2, _Type_>(
            firstPointer, firstRangeLength, secondPointer, secondRangeLength);

    return FindEndScalar<_Type_>(firstPointer, firstRangeLength, secondPointer, secondRangeLength);
}


__SIMD_STL_ALGORITHM_NAMESPACE_END
