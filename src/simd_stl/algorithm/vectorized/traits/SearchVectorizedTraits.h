#pragma once 

#include <simd_stl/compatibility/Compatibility.h>
#include <simd_stl/math/BitMath.h>

#include <src/simd_stl/algorithm/vectorized/traits/FindVectorizedTraits.h>
#include <src/simd_stl/algorithm/vectorized/traits/SearchLoop.h>

#include <algorithm>

__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

SIMD_STL_DECLARE_CPU_FEATURE_GUARDED_CLASS(
	template <simd_stl::arch::CpuFeature feature>
	class SearchTraits,
	feature,
	"simd_stl::string",
	simd_stl::arch::CpuFeature::None, simd_stl::arch::CpuFeature::AVX512F, simd_stl::arch::CpuFeature::AVX2, simd_stl::arch::CpuFeature::SSE2
);


simd_stl_always_inline __mmask16 ZeroByteMask(const __m512i vector) noexcept {
	const auto vector01 = _mm512_set1_epi8(0x01);
	const auto vector80 = _mm512_set1_epi8(int8(0x80));

	const auto vector1 = _mm512_sub_epi32(vector, vector01);
	// tempVector1 = (vector - 0x01010101) & ~vector & 0x80808080
	const auto tempVector1 = _mm512_ternarylogic_epi32(vector1, vector, vector80, 0x20);

	return _mm512_test_epi32_mask(tempVector1, tempVector1);
}

template <>
struct SearchTraits<arch::CpuFeature::None> {
	template <typename _Type_>
	simd_stl_declare_const_function simd_stl_constexpr_cxx20 const _Type_* operator()(
		const _Type_*	mainRange,
		const sizetype	mainLength,
		const _Type_*	subRange,
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

template <>
class SearchTraits<arch::CpuFeature::SSE2>: public FindTraits<arch::CpuFeature::SSE2> {
public:
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
			return SearchTraits<arch::CpuFeature::None>()(mainRange, mainLength, subRange, needleLength);

		const auto mainRangeSizeInBytes = sizeof(_Type_) * mainLength;
		constexpr auto subSizeInBytes	= sizeof(_Type_) * needleLength;

		const auto first	= Set(subRange[0]);
		const auto last		= Set(subRange[needleLength - 1]);

		const char* mainRangeChar	= reinterpret_cast<const char*>(mainRange);
		const char* subRangeChar	= reinterpret_cast<const char*>(subRange);

		for (sizetype i = 0; i < mainRangeSizeInBytes; i += 16) {
			const auto blockFirst	= _mm_loadu_si128(reinterpret_cast<const __m128i*>(mainRangeChar + i));
			const auto blockLast	= _mm_loadu_si128(reinterpret_cast<const __m128i*>(mainRangeChar + i + subSizeInBytes - sizeof(_Type_)));

			const auto equalFirst	= Compare<sizeof(_Type_)>(first, blockFirst);
			const auto equalLast	= Compare<sizeof(_Type_)>(last, blockLast);

			uint32_t mask = ToMask(_mm_and_si128(equalFirst, equalLast));

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
			return SearchTraits<arch::CpuFeature::None>()(mainRange, mainLength, subRange, subLength);

		const auto mainRangeSizeInBytes = sizeof(_Type_) * mainLength;
		const auto subInBytes			= sizeof(_Type_) * subLength;

		const auto first	= Set(subRange[0]);
		const auto last		= Set(subRange[subLength - 1]);

		const char* charMainRange = reinterpret_cast<const char*>(mainRange);

		for (sizetype i = 0; i < mainRangeSizeInBytes; i += 16) {

			const auto blockFirst	= _mm_loadu_si128(reinterpret_cast<const __m128i*>(charMainRange + i));
			const auto blockLast	= _mm_loadu_si128(reinterpret_cast<const __m128i*>(charMainRange + i + subInBytes - sizeof(_Type_)));

			const auto equalFirst	= Compare<sizeof(_Type_)>(first, blockFirst);
			const auto equalLast	= Compare<sizeof(_Type_)>(last, blockLast);

			uint16_t mask = ToMask(_mm_and_si128(equalFirst, equalLast));

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

//
//template <>
//class SearchTraits<arch::CpuFeature::AVX512F>: public FindTraits<arch::CpuFeature::AVX512F> 
//{
//public:
//	template <
//		sizetype needleLength,
//		typename _Type_,
//		typename _MemCmpLike_>
//	static simd_stl_declare_const_function const _Type_* Memcmp(
//		const _Type_*		mainRange,
//		const sizetype	mainLength,
//		const _Type_*		subRange,
//		_MemCmpLike_	memcmpLike) noexcept
//	{
//		if constexpr (needleLength <= 0)
//			return nullptr;
//
//		if (mainLength <= 0)
//			return nullptr;
//
//		const auto first	= Set(subRange[0]);
//		const auto last		= Set(subRange[needleLength - 1]);
//
//		char* haystack = const_cast<char*>(mainRange);
//		_Type_* end = haystack + mainLength;
//
//		for (; haystack < end; haystack += 64) {
//
//			const auto block_first = _mm512_loadu_si512(haystack + 0);
//			const auto block_last = _mm512_loadu_si512(haystack + needleLength - 1);
//
//			const auto first_zeros = _mm512_xor_si512(block_first, first);
//			const auto zeros = _mm512_ternarylogic_epi32(first_zeros, block_last, last, 0xf6);
//
//			uint32_t mask = ZeroByteMask(zeros);
//
//			const char* charSubRange	= reinterpret_cast<const char*>(subRange);
//			const char* charMainRange	= reinterpret_cast<const char*>(haystack);
//
//			while (mask) {
//				const uint64_t p = math::CountTrailingZeroBits(mask);
//
//				if (memcmpLike(haystack + 4 * p + 0, charSubRange))
//					return mainRange + ((haystack - charMainRange) + 4 * p + 0);
//
//				if (memcmpLike(haystack + 4 * p + 1, charSubRange))
//					return mainRange + ((haystack - charMainRange) + 4 * p + 1);
//
//				if (memcmpLike(haystack + 4 * p + 2, charSubRange))
//					return mainRange + ((haystack - charMainRange) + 4 * p + 2);
//
//				if (memcmpLike(haystack + 4 * p + 3, charSubRange))
//					return mainRange + ((haystack - charMainRange) + 4 * p + 3);
//
//				mask = math::ClearLeftMostSet(mask);
//			}
//		}
//
//		return nullptr;
//	}
//	
//	template <typename _Type_>
//	static simd_stl_declare_const_function const _Type_* AnySize(
//		const _Type_*	mainRange,
//		const sizetype	mainLength,
//		const _Type_*	subRange,
//		const sizetype	subLength) noexcept
//	{
//		if (mainLength <= 0 || subLength <= 0)
//			return nullptr;
//
//		const auto first	= Set(subRange[0]);
//		const auto last		= Set(subRange[subLength - 1]);
//
//		char* haystack = const_cast<char*>(mainRange);
//		_Type_* end = haystack + mainLength;
//
//		for (; haystack < end; haystack += 64) {
//
//			const auto blockFirst = _mm512_loadu_si512(haystack + 0);
//			const auto blockLast = _mm512_loadu_si512(haystack + subLength - 1);
//
//			const auto firstZeros = _mm512_xor_si512(blockFirst, first);
//
//			const auto zeros = _mm512_ternarylogic_epi32(firstZeros, blockLast, last, 0xf6);
//			uint32_t mask = ZeroByteMask(zeros);
//
//			while (mask) {
//
//				const uint64_t p = math::CountTrailingZeroBits(mask);
//
//				if (memcmp(haystack + 4 * p + 0, subRange, subLength) == 0)
//					return mainRange + ((haystack - mainRange) + 4 * p + 0);
//
//				if (memcmp(haystack + 4 * p + 1, subRange, subLength) == 0)
//					return mainRange + ((haystack - mainRange) + 4 * p + 1);
//
//				if (memcmp(haystack + 4 * p + 2, subRange, subLength) == 0)
//					return mainRange + ((haystack - mainRange) + 4 * p + 2);
//
//				if (memcmp(haystack + 4 * p + 3, subRange, subLength) == 0)
//					return mainRange + ((haystack - mainRange) + 4 * p + 3);
//
//				mask = math::ClearLeftMostSet(mask);
//			}
//		}
//
//		return nullptr;
//	}
//};


template <>
class SearchTraits<arch::CpuFeature::AVX2> : public FindTraits<arch::CpuFeature::AVX2> {
public:
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
			return SearchTraits<arch::CpuFeature::None>()(mainRange, mainLength, subRange, needleLength);
		
		const auto mainRangeSizeInBytes = sizeof(_Type_) * mainLength;
		constexpr auto subInBytes		= sizeof(_Type_) * needleLength;

		const auto first	= Set(subRange[0]);
		const auto last		= Set(subRange[needleLength - 1]);

		const char* mainRangeChar = reinterpret_cast<const char*>(mainRange);

		for (sizetype i = 0; i < mainRangeSizeInBytes; i += 32) {
			const auto blockFirst	= _mm256_lddqu_si256(reinterpret_cast<const __m256i*>(mainRangeChar + i));
			const auto blockLast	= _mm256_lddqu_si256(reinterpret_cast<const __m256i*>(mainRangeChar + i + subInBytes - sizeof(_Type_)));

			const auto equalFirst	= Compare<sizeof(_Type_)>(first, blockFirst);
			const auto equalLast	= Compare<sizeof(_Type_)>(last, blockLast);

			uint32_t mask = _mm256_movemask_epi8(_mm256_and_si256(equalFirst, equalLast));

			while (mask != 0) {
				const auto bitpos = math::CountTrailingZeroBits(mask);

				if (memcmpLike(mainRangeChar + i + bitpos + sizeof(_Type_), reinterpret_cast<const char*>(subRange) + sizeof(_Type_)))
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
		if (subLength <= 0 || mainLength <= 0)
			return nullptr;

		if (((subLength & (~sizetype{ 0x1F }))) != 0)
			return SearchTraits<arch::CpuFeature::None>()(mainRange, mainLength, subRange, subLength);

		const auto first	= Set(subRange[0]);
		const auto last		= Set(subRange[subLength - 1]);

		const auto mainRangeSizeInBytes = sizeof(_Type_) * mainLength;
		const auto subSizeInBytes = sizeof(_Type_) * subLength;

		const char* mainRangeChar = reinterpret_cast<const char*>(mainRange);

		for (sizetype i = 0; i < mainRangeSizeInBytes; i += 32) {
			const auto blockFirst	= _mm256_lddqu_si256(reinterpret_cast<const __m256i*>(mainRangeChar + i));
			const auto blockLast	= _mm256_lddqu_si256(reinterpret_cast<const __m256i*>(mainRangeChar + i + subSizeInBytes - sizeof(_Type_)));

			const auto equalFirst	= Compare<sizeof(_Type_)>(first, blockFirst);
			const auto equalLast	= Compare<sizeof(_Type_)>(last, blockLast);

			uint32 mask = _mm256_movemask_epi8(_mm256_and_si256(equalFirst, equalLast));

			while (mask != 0) {
				const auto bitpos = math::CountTrailingZeroBits(mask);

				if (memcmp(mainRangeChar + i + bitpos + 1, reinterpret_cast<const char*>(subRange) + sizeof(_Type_), subSizeInBytes - (2 * sizeof(_Type_))) == 0)
					return reinterpret_cast<const _Type_*>(mainRangeChar + i + bitpos);

				mask = math::ClearLeftMostSet(mask);
			}
		}

		return nullptr;
	}

	template <
		sizetype subLength,
		typename _Type_>
	static simd_stl_declare_const_function const _Type_* Equal(
		const _Type_*	mainRange,
		sizetype		mainLength,
		const _Type_*	subRange) noexcept
	{
		static_assert(subLength > 0 && subLength < 16, "subLength must be in range [1..15]");

		if (mainLength <= 0)
			return nullptr;

		if constexpr (subLength < 32)
			return SearchTraits<arch::CpuFeature::None>()(mainRange, mainLength, subRange, subLength);

		constexpr sizetype subSizeInBytes = sizeof(_Type_) * subLength;

		if constexpr (subSizeInBytes < 32)
			return ::std::search(mainRange, mainRange + mainLength, subRange, subRange + subLength);//SearchTraits<arch::CpuFeature::SSE2>::


		__m256i broadcasted[subLength];

		for (unsigned i = 0; i < subLength; i++)
			broadcasted[i] = Set(subRange[i]);

		const char* mainRangeChar	= reinterpret_cast<const char*>(mainRange);
		const char* subRangeChar	= reinterpret_cast<const char*>(subRange);

		auto current				= _mm256_lddqu_si256(reinterpret_cast<const __m256i*>(mainRangeChar));

		for (sizetype i = 0; i < subSizeInBytes; i += 32) {
			const auto next = _mm256_lddqu_si256(reinterpret_cast<const __m256i*>(mainRangeChar + i + 32));
			auto equal		= Compare<sizeof(_Type_)>(current, broadcasted[0]);

			__m256i next1 = _mm256_undefined_si256();

			next1 = _mm256_inserti128_si256(next1, _mm256_extracti128_si256(current, 1), 0); // b
			next1 = _mm256_inserti128_si256(next1, _mm256_extracti128_si256(next, 0), 1); // c

#if !defined(simd_stl_cpp_clang) && !defined(simd_stl_cpp_msvc)
			for (unsigned j = 1; j < subLength; j++) {
				 auto subRangeVector = _mm256_alignr_epi8(next1, current, j);
				equal = _mm256_and_si256(equal, Compare<sizeof(_Type_)>(subRangeVector, broadcasted[j]));
			}
#else
			StringFindLoop<_Type_, FindTraits<arch::CpuFeature::AVX2>, subLength>()(equal, next1, current, broadcasted);
#endif

			current = next;
			const uint32 mask = ToMask(equal);

			if (mask != 0)
				return reinterpret_cast<const _Type_*>(subRangeChar + i + math::CountTrailingZeroBits(mask));
		}

		return nullptr;
	}
};

__SIMD_STL_ALGORITHM_NAMESPACE_END
