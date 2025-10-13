#pragma once

#include <simd_stl/SimdStlNamespace.h>

#include <simd_stl/arch/ProcessorFeatures.h>

#include <src/simd_stl/algorithm/FixedMemcmp.h>
#include <simd_stl/algorithm/find/Find.h>

#include <simd_stl/numeric/BasicSimd.h>


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
		if (mainLength == subLength)
			return (memcmp(mainRange, subRange, mainLength) == 0) ? mainRange : nullptr;

		const _Type_& first = subRange[0];
		const sizetype maxpos = sizetype(mainLength) - sizetype(subLength) + 1;

		for (sizetype i = 0; i < maxpos; i++) {
			if (mainRange[i] != first) {
				i++;

				while (i < maxpos && mainRange[i] != first)
					i++;

				if (i == maxpos)
					break;
			}

			sizetype j = 1;

			for (; j < subLength; ++j)
				if (mainRange[i + j] != subRange[j])
					break;

			if (j == subLength)
				return (mainRange + i);
		}

		return nullptr;
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

//template <>
//struct _Search<arch::CpuFeature::AVX2> {
//    template <typename _Type_>
//    simd_stl_declare_const_function simd_stl_constexpr_cxx20 const _Type_* operator()(
//        const _Type_*   mainRange,
//        const sizetype	mainLength,
//        const _Type_*   subRange,
//        const sizetype	subLength) noexcept
//    {
//
//        using _Implementation_ = SearchTraits<arch::CpuFeature::AVX2>;
//        const _Type_* result = nullptr;
//
//        if (mainLength < subLength)
//            return result;
//
//        switch (subLength) {
//            case 0: return mainRange;
//            case 1: return simd_stl::algorithm::find(mainRange, mainRange + mainLength, *subRange); 
//            case 2: result = _Implementation_::Equal<2>(mainRange, mainLength, subRange); break;
//            case 3: result = _Implementation_::Memcmp<3>(mainRange, mainLength, subRange, memcmp1); break;
//            case 4: result = _Implementation_::Memcmp<4>(mainRange, mainLength, subRange, memcmp2); break;
//            case 5: result = _Implementation_::Memcmp<5>(mainRange, mainLength, subRange, memcmp4); break;
//            case 6: result = _Implementation_::Memcmp<6>(mainRange, mainLength, subRange, memcmp4); break;
//            case 7: result = _Implementation_::Memcmp<7>(mainRange, mainLength, subRange, memcmp5); break;
//            case 8: result = _Implementation_::Memcmp<8>(mainRange, mainLength, subRange, memcmp6); break;
//            case 9: result = _Implementation_::Memcmp<9>(mainRange, mainLength, subRange, memcmp8); break;
//            case 10: result = _Implementation_::Memcmp<10>(mainRange, mainLength, subRange, memcmp8); break;
//            case 11: result = _Implementation_::Memcmp<11>(mainRange, mainLength, subRange, memcmp9); break;
//            case 12: result = _Implementation_::Memcmp<12>(mainRange, mainLength, subRange, memcmp10); break;
//            default: result = _Implementation_::AnySize(mainRange, mainLength, subRange, subLength); break;
//        }
//
//        if (result - mainRange <= mainLength - subLength)
//            return result;
//
//        return nullptr;
//    }
//};

template <>
struct _Search<arch::CpuFeature::SSE2> {
    template <typename _Type_>
    simd_stl_declare_const_function simd_stl_constexpr_cxx20 const _Type_* operator()(
        const _Type_*   mainRange,
        const sizetype	mainLength,
        const _Type_*   subRange,
        const sizetype	subLength) noexcept
    {
        const _Type_* result = nullptr;

        if (mainLength < subLength)
            return result;

        switch (subLength) {
            case 0: return mainRange;
            case 1: return simd_stl::algorithm::find(mainRange, mainRange + mainLength, *subRange);
            case 2: result = Memcmp<2>(mainRange, mainLength, subRange, alwaysTrue); break;
            case 3: result = Memcmp<3>(mainRange, mainLength, subRange, memcmp1); break;
            case 4: result = Memcmp<4>(mainRange, mainLength, subRange, memcmp2); break;
            case 5: result = Memcmp<5>(mainRange, mainLength, subRange, memcmp4); break;
            case 6: result = Memcmp<6>(mainRange, mainLength, subRange, memcmp4); break;
            case 7: result = Memcmp<7>(mainRange, mainLength, subRange, memcmp5); break;
            case 8: result = Memcmp<8>(mainRange, mainLength, subRange, memcmp6); break;
            case 9: result = Memcmp<9>(mainRange, mainLength, subRange, memcmp8); break;
            case 10: result = Memcmp<10>(mainRange, mainLength, subRange, memcmp8); break;
            case 11: result = Memcmp<11>(mainRange, mainLength, subRange, memcmp9); break;
            case 12: result = Memcmp<12>(mainRange, mainLength, subRange, memcmp10); break;
            default: result = AnySize(mainRange, mainLength, subRange, subLength); break;
        }

        if (result - mainRange <= mainLength - subLength)
            return result;

        return nullptr;
    }
private:
    template <
		sizetype needleLength,
		typename _Type_,
		typename _MemCmpLike_>
	static simd_stl_declare_const_function const _Type_* Memcmp(
		const _Type_*	mainRange,
		const sizetype	mainLength,
		const _Type_*	subRange,
		_MemCmpLike_	memcmpLike) noexcept
	{
		if constexpr (needleLength <= 0)
			return nullptr;

		if (mainLength <= 0)
			return nullptr;

		if constexpr (needleLength < 16)
			return _Search<arch::CpuFeature::None>()(mainRange, mainLength, subRange, needleLength);

		using _SimdType_ = numeric::basic_simd<arch::CpuFeature::SSE2, _Type_>;

		const auto mainRangeSizeInBytes = sizeof(_Type_) * mainLength;
		constexpr auto subSizeInBytes	= sizeof(_Type_) * needleLength;

		const auto first	= _SimdType_(subRange[0]);
		const auto last		= _SimdType_(subRange[needleLength - 1]);

		const char* mainRangeChar	= reinterpret_cast<const char*>(mainRange);
		const char* subRangeChar	= reinterpret_cast<const char*>(subRange);

		for (sizetype i = 0; i < mainRangeSizeInBytes; i += 16) {
			const auto blockFirst	= _SimdType_::loadUnaligned(mainRangeChar + i);
			const auto blockLast	= _SimdType_::loadUnaligned(mainRangeChar + i + subSizeInBytes - sizeof(_Type_));

			const auto equalFirst	= first.maskEqual<_Type_>(blockFirst).unwrap();
			const auto equalLast	= last.maskEqual<_Type_>(blockLast).unwrap();

			uint32 mask				= equalFirst & equalLast;

			while (mask != 0) {
				const auto bitpos = math::CountTrailingZeroBits(mask);

				if (memcmpLike(mainRangeChar + i + bitpos + sizeof(_Type_), reinterpret_cast<const char*>(subRange) + (sizeof(_Type_))))
					return reinterpret_cast<const _Type_*>(mainRangeChar + i + bitpos);

				mask = math::ClearLeftMostSet(mask);
			}
		}

		return nullptr;
	}
	
	template <typename _Type_>
	static simd_stl_declare_const_function const _Type_* AnySize(
		const _Type_*	mainRange,
		const sizetype	mainLength,
		const _Type_*	subRange,
		const sizetype	subLength) noexcept
	{
		if (mainLength <= 0 || subLength <= 0)
			return nullptr;

		if (((subLength & (~sizetype{ 0xF }))) != 0)
			return _Search<arch::CpuFeature::None>()(mainRange, mainLength, subRange, subLength);

		using _SimdType_ = numeric::basic_simd<arch::CpuFeature::SSE2, _Type_>;

		const auto mainRangeSizeInBytes = sizeof(_Type_) * mainLength;
		const auto subInBytes			= sizeof(_Type_) * subLength;

		const auto first	= _SimdType_(subRange[0]);
		const auto last		= _SimdType_(subRange[subLength - 1]);

		const char* charMainRange = reinterpret_cast<const char*>(mainRange);

		for (sizetype i = 0; i < mainRangeSizeInBytes; i += 16) {

			const auto blockFirst	= _SimdType_::loadUnaligned(charMainRange + i);
			const auto blockLast	= _SimdType_::loadUnaligned(charMainRange + i + subInBytes - sizeof(_Type_));

			const auto equalFirst	= first.maskEqual<_Type_>(blockFirst).unwrap();
			const auto equalLast	= last.maskEqual<_Type_>(blockLast).unwrap();

			uint32 mask				= equalFirst & equalLast;

			while (mask != 0) {
				const auto bitpos = math::CountTrailingZeroBits(mask);

				if (memcmp(charMainRange + i + bitpos + sizeof(_Type_), reinterpret_cast<const char*>(subRange) + 1, subInBytes - (2 * sizeof(_Type_))) == 0)
					return reinterpret_cast<const _Type_*>(charMainRange + i + bitpos);

				mask = math::ClearLeftMostSet(mask);
			}
		}

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
    else */ /*if (arch::ProcessorFeatures::AVX2())
        return _Search<arch::CpuFeature::AVX2>()(first1, mainRangeLength, first2, subRangeLength);*/
    /*else */if (arch::ProcessorFeatures::SSE2())
        return _Search<arch::CpuFeature::SSE2>()(first1, mainRangeLength, first2, subRangeLength);

    return _Search<arch::CpuFeature::None>()(first1, mainRangeLength, first2, subRangeLength);
}


__SIMD_STL_ALGORITHM_NAMESPACE_END
