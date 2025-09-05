#pragma once

#include <simd_stl/SimdStlNamespace.h>

#include <simd_stl/arch/ProcessorFeatures.h>

#include <src/simd_stl/algorithm/vectorized/traits/SearchVectorizedTraits.h>
#include <src/simd_stl/algorithm/FixedMemcmp.h>

#include <simd_stl/algorithm/find/Find.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

SIMD_STL_DECLARE_CPU_FEATURE_GUARDED_CLASS(
    template <arch::CpuFeature feature> struct _Search,
    feature,
    "simd_stl::algorithm",
    arch::CpuFeature::None, arch::CpuFeature::AVX512F, arch::CpuFeature::AVX2, arch::CpuFeature::SSE2
);

template <>
struct _Search<arch::CpuFeature::None> {
    template <typename _Type_>
    simd_stl_declare_const_function simd_stl_constexpr_cxx20 const _Type_* operator()(
        const _Type_*   mainRange,
        const sizetype	mainLength,
        const _Type_*   subRange,
        const sizetype	subLength) noexcept
    {
        return SearchTraits<arch::CpuFeature::None>()(mainRange, mainLength, subRange, subLength);
    }
};
//
//template <>
//struct _Search<arch::CpuFeature::AVX512F> {
//    template <typename _Type_>
//    simd_stl_declare_const_function simd_stl_constexpr_cxx20 const _Type_* operator()(
//        const _Type_*   mainRange,
//        const sizetype	mainLength,
//        const _Type_*   subRange,
//        const sizetype	subLength) noexcept
//    {
//        using _Implementation_ = SearchTraits<arch::CpuFeature::AVX512F>;
//
//        const _Type_* result = nullptr;
//
//        if (mainLength < subLength)
//            return result;
//
//        switch (subLength) {
//        case 0:
//            return mainRange;
//
//        case 1:
//            return simd_stl::algorithm::find(mainRange, mainRange + mainLength, *subRange);
//
//        case 2:
//            result = _Implementation_::Memcmp<2>(mainRange, mainLength, subRange, memcmp2);
//            break;
//
//        case 3:
//            result = _Implementation_::Memcmp<3>(mainRange, mainLength, subRange, memcmp3);
//            break;
//
//        case 4:
//            result = _Implementation_::Memcmp<4>(mainRange, mainLength, subRange, memcmp4);
//            break;
//
//        case 5:
//            result = _Implementation_::Memcmp<5>(mainRange, mainLength, subRange, memcmp5);
//            break;
//
//        case 6:
//            result = _Implementation_::Memcmp<6>(mainRange, mainLength, subRange, memcmp6);
//            break;
//
//        case 7:
//            result = _Implementation_::Memcmp<7>(mainRange, mainLength, subRange, memcmp7);
//            break;
//
//        case 8:
//            result = _Implementation_::Memcmp<8>(mainRange, mainLength, subRange, memcmp8);
//            break;
//
//        case 9:
//            result = _Implementation_::Memcmp<9>(mainRange, mainLength, subRange, memcmp9);
//            break;
//
//        case 10:
//            result = _Implementation_::Memcmp<10>(mainRange, mainLength, subRange, memcmp10);
//            break;
//
//        case 11:
//            result = _Implementation_::Memcmp<11>(mainRange, mainLength, subRange, memcmp11);
//            break;
//
//        case 12:
//            result = _Implementation_::Memcmp<12>(mainRange, mainLength, subRange, memcmp12);
//            break;
//
//        default:
//            result = _Implementation_::AnySize(mainRange, mainLength, subRange, subLength);
//            break;
//        }
//
//        if (result - mainRange <= mainLength - subLength)
//            return result;
//
//        return nullptr;
//    }
//};

template <>
struct _Search<arch::CpuFeature::AVX2> {
    template <typename _Type_>
    simd_stl_declare_const_function simd_stl_constexpr_cxx20 const _Type_* operator()(
        const _Type_*   mainRange,
        const sizetype	mainLength,
        const _Type_*   subRange,
        const sizetype	subLength) noexcept
    {

        using _Implementation_ = SearchTraits<arch::CpuFeature::AVX2>;
        const _Type_* result = nullptr;

        if (mainLength < subLength)
            return result;

        switch (subLength) {
            case 0: return mainRange;
            case 1: return simd_stl::algorithm::find(mainRange, mainRange + mainLength, *subRange); 
            case 2: result = _Implementation_::Equal<2>(mainRange, mainLength, subRange); break;
            case 3: result = _Implementation_::Memcmp<3>(mainRange, mainLength, subRange, memcmp1); break;
            case 4: result = _Implementation_::Memcmp<4>(mainRange, mainLength, subRange, memcmp2); break;
            case 5: result = _Implementation_::Memcmp<5>(mainRange, mainLength, subRange, memcmp4); break;
            case 6: result = _Implementation_::Memcmp<6>(mainRange, mainLength, subRange, memcmp4); break;
            case 7: result = _Implementation_::Memcmp<7>(mainRange, mainLength, subRange, memcmp5); break;
            case 8: result = _Implementation_::Memcmp<8>(mainRange, mainLength, subRange, memcmp6); break;
            case 9: result = _Implementation_::Memcmp<9>(mainRange, mainLength, subRange, memcmp8); break;
            case 10: result = _Implementation_::Memcmp<10>(mainRange, mainLength, subRange, memcmp8); break;
            case 11: result = _Implementation_::Memcmp<11>(mainRange, mainLength, subRange, memcmp9); break;
            case 12: result = _Implementation_::Memcmp<12>(mainRange, mainLength, subRange, memcmp10); break;
            default: result = _Implementation_::AnySize(mainRange, mainLength, subRange, subLength); break;
        }

        if (result - mainRange <= mainLength - subLength)
            return result;

        return nullptr;
    }
};

template <>
struct _Search<arch::CpuFeature::SSE2> {
    template <typename _Type_>
    simd_stl_declare_const_function simd_stl_constexpr_cxx20 const _Type_* operator()(
        const _Type_*   mainRange,
        const sizetype	mainLength,
        const _Type_*   subRange,
        const sizetype	subLength) noexcept
    {
        using _Implementation_ = SearchTraits<arch::CpuFeature::SSE2>;
        const _Type_* result = nullptr;

        if (mainLength < subLength)
            return result;

        switch (subLength) {
            case 0: return mainRange;
            case 1: return simd_stl::algorithm::find(mainRange, mainRange + mainLength, *subRange);
            case 2: result = _Implementation_::Memcmp<2>(mainRange, mainLength, subRange, alwaysTrue); break;
            case 3: result = _Implementation_::Memcmp<3>(mainRange, mainLength, subRange, memcmp1); break;
            case 4: result = _Implementation_::Memcmp<4>(mainRange, mainLength, subRange, memcmp2); break;
            case 5: result = _Implementation_::Memcmp<5>(mainRange, mainLength, subRange, memcmp4); break;
            case 6: result = _Implementation_::Memcmp<6>(mainRange, mainLength, subRange, memcmp4); break;
            case 7: result = _Implementation_::Memcmp<7>(mainRange, mainLength, subRange, memcmp5); break;
            case 8: result = _Implementation_::Memcmp<8>(mainRange, mainLength, subRange, memcmp6); break;
            case 9: result = _Implementation_::Memcmp<9>(mainRange, mainLength, subRange, memcmp8); break;
            case 10: result = _Implementation_::Memcmp<10>(mainRange, mainLength, subRange, memcmp8); break;
            case 11: result = _Implementation_::Memcmp<11>(mainRange, mainLength, subRange, memcmp9); break;
            case 12: result = _Implementation_::Memcmp<12>(mainRange, mainLength, subRange, memcmp10); break;
            default: result = _Implementation_::AnySize(mainRange, mainLength, subRange, subLength); break;
        }

        if (result - mainRange <= mainLength - subLength)
            return result;

        return nullptr;
    }
};

template <class _Type_>
simd_stl_declare_const_function simd_stl_constexpr_cxx20 const _Type_* SearchVectorized(
    const _Type_*   first1,
    const sizetype  mainRangeLength,
    const _Type_*   first2,
    const sizetype  subRangeLength) noexcept
{
   /* if (arch::ProcessorFeatures::AVX512F())
        return _Search<arch::CpuFeature::AVX512F>()(first1, mainRangeLength, first2, subRangeLength);
    else */ if (arch::ProcessorFeatures::AVX2())
        return _Search<arch::CpuFeature::AVX2>()(first1, mainRangeLength, first2, subRangeLength);
    else if (arch::ProcessorFeatures::SSE2())
        return _Search<arch::CpuFeature::SSE2>()(first1, mainRangeLength, first2, subRangeLength);

    return _Search<arch::CpuFeature::None>()(first1, mainRangeLength, first2, subRangeLength);
}


__SIMD_STL_ALGORITHM_NAMESPACE_END
