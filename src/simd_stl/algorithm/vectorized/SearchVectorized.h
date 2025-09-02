#pragma once

#include <simd_stl/SimdStlNamespace.h>

#include <simd_stl/compatibility/Inline.h>
#include <simd_stl/compatibility/FunctionAttributes.h>

#include <simd_stl/compatibility/SimdCompatibility.h>
#include <simd_stl/arch/ProcessorFeatures.h>

#include <src/simd_stl/algorithm/AdvanceBytes.h>

__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <class _Type_>
simd_stl_declare_const_function simd_stl_always_inline simd_stl_constexpr_cxx20 const void* SearchScalar(
    const void* first1,
    const void* last1,
    const void* first2,
    const void* last2)  noexcept
{
    const auto first1T  = reinterpret_cast<const _Type_*>(first1);
    const auto first2T  = reinterpret_cast<const _Type_*>(first2);

    const auto last1T   = reinterpret_cast<const _Type_*>(last1);
    const auto last2T   = reinterpret_cast<const _Type_*>(last2);

    const auto firstRangeLength     = IteratorsDifference(first1T, last1T);
    const auto secondRangeLength    = IteratorsDifference(first2T, last2T);

    for (; secondRangeLength <= firstRangeLength; ++first1T, --firstRangeLength) {
        auto mid1 = first1T;

        for (auto mid2 = first2T; ; ++mid1, ++mid2)
            if (mid2 == last2T)
                return (first1T);
            else if (!(*mid1 == *mid2))
                break;
    }

    return last1;
}


SIMD_STL_DECLARE_CPU_FEATURE_GUARDED_FUNCTION(
    SIMD_STL_ECHO(
        template <arch::CpuFeature feature, typename _Type_>
        simd_stl_declare_const_function const char* SearchVectorizedInternal(
            const _Type_* mainString,
            const sizetype	mainLength,
            const _Type_* subString,
            const sizetype	subLength) noexcept
    ),
    feature,
    "simd_stl::algorithm",
    arch::CpuFeature::None, arch::CpuFeature::AVX512F, arch::CpuFeature::AVX2, arch::CpuFeature::SSE2
)


template <typename _Type_>
simd_stl_declare_const_function const char* SearchVectorizedInternal<arch::CpuFeature::None>(
    const char* mainString,
    const sizetype	mainLength,
    const char* subString,
    const sizetype	subLength) noexcept
{
    if (mainLength == subLength)
        return (memcmp(mainString, subString, mainLength) == 0) ? mainString : nullptr;

    const char first = subString[0];
    const sizetype maxpos = sizetype(mainLength) - sizetype(subLength) + 1;

    for (sizetype i = 0; i < maxpos; i++) {
        if (mainString[i] != first) {
            i++;

            while (i < maxpos && mainString[i] != first)
                i++;

            if (i == maxpos)
                break;
        }

        sizetype j = 1;

        for (; j < subLength; ++j)
            if (mainString[i + j] != subString[j])
                break;

        if (j == subLength)
            return (mainString + i);
    }

    return nullptr;
}

template <typename _Type_>
simd_stl_declare_const_function const char* SearchVectorizedInternal<arch::CpuFeature::AVX512F>(
    const _Type_*   mainString,
    const sizetype	mainLength,
    const _Type_*   subString,
    const sizetype	subLength) noexcept
{
    using _Implementation_ = SearchTraits<arch::CpuFeature::AVX512F>;

    const char* result = nullptr;

    if (mainLength < subLength)
        return result;

    switch (subLength) {
    case 0:
        return mainString;

    case 1: {
        const _Type_* res = reinterpret_cast<const _Type_*>();
        return res;
    }

    case 2:
        result = _Implementation_::Memcmp<2, _Type_>(mainString, mainLength, subString, memory::memcmp2);
        break;

    case 3:
        result = _Implementation_::Memcmp<3>(mainString, mainLength, subString, memory::memcmp3);
        break;

    case 4:
        result = _Implementation_::Memcmp<4>(mainString, mainLength, subString, memory::memcmp4);
        break;

    case 5:
        result = _Implementation_::Memcmp<5>(mainString, mainLength, subString, memory::memcmp5);
        break;

    case 6:
        result = _Implementation_::Memcmp<6>(mainString, mainLength, subString, memory::memcmp6);
        break;

    case 7:
        result = _Implementation_::Memcmp<7>(mainString, mainLength, subString, memory::memcmp7);
        break;

    case 8:
        result = _Implementation_::Memcmp<8>(mainString, mainLength, subString, memory::memcmp8);
        break;

    case 9:
        result = _Implementation_::Memcmp<9>(mainString, mainLength, subString, memory::memcmp9);
        break;

    case 10:
        result = _Implementation_::Memcmp<10>(mainString, mainLength, subString, memory::memcmp10);
        break;

    case 11:
        result = _Implementation_::Memcmp<11>(mainString, mainLength, subString, memory::memcmp11);
        break;

    case 12:
        result = _Implementation_::Memcmp<12>(mainString, mainLength, subString, memory::memcmp12);
        break;

    default:
        result = _Implementation_::AnySize(mainString, mainLength, subString, subLength);
        break;
    }

    if (result - mainString <= mainLength - subLength)
        return result;

    return nullptr;
}

template <typename _Type_>
simd_stl_declare_const_function const char* SearchVectorizedInternal<arch::CpuFeature::AVX2>(
    const char* mainString,
    const sizetype	mainLength,
    const char* subString,
    const sizetype	subLength) noexcept
{
    using _Implementation_ = BaseStrstrnImplementationsInternal<arch::CpuFeature::AVX2>;

    const char* result = nullptr;

    if (mainLength < subLength)
        return result;

    switch (subLength) {
    case 0:
        return mainString;

    case 1: {
        const char* res = reinterpret_cast<const char*>(strchr(mainString, subString[0]));
        return res;
    }

    case 2:
        result = _Implementation_::Equal<2>(mainString, mainLength, subString);
        break;

    case 3:
        result = _Implementation_::Memcmp<3>(mainString, mainLength, subString, memory::memcmp1);
        break;

    case 4:
        result = _Implementation_::Memcmp<4>(mainString, mainLength, subString, memory::memcmp2);
        break;

    case 5:
        result = _Implementation_::Memcmp<5>(mainString, mainLength, subString, memory::memcmp4);
        break;

    case 6:
        result = _Implementation_::Memcmp<6>(mainString, mainLength, subString, memory::memcmp4);
        break;

    case 7:
        result = _Implementation_::Memcmp<7>(mainString, mainLength, subString, memory::memcmp5);
        break;

    case 8:
        result = _Implementation_::Memcmp<8>(mainString, mainLength, subString, memory::memcmp6);
        break;

    case 9:
        result = _Implementation_::Memcmp<9>(mainString, mainLength, subString, memory::memcmp8);
        break;

    case 10:
        result = _Implementation_::Memcmp<10>(mainString, mainLength, subString, memory::memcmp8);
        break;

    case 11:
        result = _Implementation_::Memcmp<11>(mainString, mainLength, subString, memory::memcmp9);
        break;

    case 12:
        result = _Implementation_::Memcmp<12>(mainString, mainLength, subString, memory::memcmp10);
        break;

    default:
        result = _Implementation_::AnySize(mainString, mainLength, subString, subLength);
        break;
    }

    if (result - mainString <= mainLength - subLength)
        return result;

    return nullptr;
}

template <>
simd_stl_declare_const_function const char* SearchVectorizedInternal<arch::CpuFeature::SSE2>(
    const char* mainString,
    const sizetype	mainLength,
    const char* subString,
    const sizetype	subLength) noexcept
{
    using _Implementation_ = BaseStrstrnImplementationsInternal<arch::CpuFeature::AVX2>;

    const char* result = nullptr;

    if (mainLength < subLength)
        return result;

    switch (subLength) {
    case 0:
        return mainString;

    case 1: {
        const char* res = reinterpret_cast<const char*>(strchr(mainString, subString[0]));
        return res;
    }

    case 2:
        result = _Implementation_::Memcmp<2>(mainString, mainLength, subString, memory::alwaysTrue);
        break;

    case 3:
        result = _Implementation_::Memcmp<3>(mainString, mainLength, subString, memory::memcmp1);
        break;

    case 4:
        result = _Implementation_::Memcmp<4>(mainString, mainLength, subString, memory::memcmp2);
        break;

    case 5:
        result = _Implementation_::Memcmp<5>(mainString, mainLength, subString, memory::memcmp4);
        break;

    case 6:
        result = _Implementation_::Memcmp<6>(mainString, mainLength, subString, memory::memcmp4);
        break;

    case 7:
        result = _Implementation_::Memcmp<7>(mainString, mainLength, subString, memory::memcmp5);
        break;

    case 8:
        result = _Implementation_::Memcmp<8>(mainString, mainLength, subString, memory::memcmp6);
        break;

    case 9:
        result = _Implementation_::Memcmp<9>(mainString, mainLength, subString, memory::memcmp8);
        break;

    case 10:
        result = _Implementation_::Memcmp<10>(mainString, mainLength, subString, memory::memcmp8);
        break;

    case 11:
        result = _Implementation_::Memcmp<11>(mainString, mainLength, subString, memory::memcmp9);
        break;

    case 12:
        result = _Implementation_::Memcmp<12>(mainString, mainLength, subString, memory::memcmp10);
        break;

    default:
        result = _Implementation_::AnySize(mainString, mainLength, subString, subLength);
        break;
    }

    if (result - mainString <= mainLength - subLength)
        return result;

    return nullptr;
}

template <class _Type_>
simd_stl_declare_const_function simd_stl_constexpr_cxx20 const void* SearchVectorized(
    const void* firstPointer,
    const void* lastPointer,
    _Type_      value) noexcept
{
    if (arch::ProcessorFeatures::AVX512F())
        return FindVectorizedInternal<FindTraits<arch::CpuFeature::AVX512F>, _Type_>(firstPointer, lastPointer, value);
    else if (arch::ProcessorFeatures::AVX2())
        return FindVectorizedInternal<FindTraits<arch::CpuFeature::AVX2>, _Type_>(firstPointer, lastPointer, value);
    else if (arch::ProcessorFeatures::SSE2())
        return FindVectorizedInternal<FindTraits<arch::CpuFeature::SSE2>, _Type_>(firstPointer, lastPointer, value);

    return FindScalar(firstPointer, lastPointer, value);
}


__SIMD_STL_ALGORITHM_NAMESPACE_END
