#pragma once 

#include <simd_stl/compatibility/Compatibility.h>
#include <src/simd_stl/math/BitMath.h>

#include <src/simd_stl/algorithm/vectorized/traits/FindVectorizedTraits.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

SIMD_STL_DECLARE_CPU_FEATURE_GUARDED_CLASS(
	template <simd_stl::arch::CpuFeature feature>
	class SearchTraits,
	feature,
	"simd_stl::string",
	simd_stl::arch::CpuFeature::AVX512F, simd_stl::arch::CpuFeature::AVX2, simd_stl::arch::CpuFeature::SSE2
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
class SearchTraits<arch::CpuFeature::AVX512F>: public FindTraits<arch::CpuFeature::AVX512F> 
{
public:
	template <
		sizetype needleLength,
		typename _Type_,
		typename _MemCmpLike_>
	static simd_stl_declare_const_function const _Type_* Memcmp(
		const _Type_*		mainRange,
		const sizetype	mainLength,
		const _Type_*		subRange,
		_MemCmpLike_	memcmpLike) noexcept
	{
		if constexpr (needleLength <= 0)
			return nullptr;

		if (mainLength <= 0)
			return nullptr;

		const auto first = _mm512_set1_epi8(subRange[0]);
		const auto last = _mm512_set1_epi8(subRange[needleLength - 1]);

		_Type_* haystack = const_cast<_Type_*>(mainRange);
		_Type_* end = haystack + mainLength;

		for (; haystack < end; haystack += 64) {

			const auto block_first = _mm512_loadu_si512(haystack + 0);
			const auto block_last = _mm512_loadu_si512(haystack + needleLength - 1);

			const auto first_zeros = _mm512_xor_si512(block_first, first);
			const auto zeros = _mm512_ternarylogic_epi32(first_zeros, block_last, last, 0xf6);

			uint32_t mask = ZeroByteMask(zeros);

			while (mask) {

				const uint64_t p = math::CountTrailingZeroBits(mask);

				if (memcmpLike(haystack + 4 * p + 0, subRange))
					return mainRange + ((haystack - mainRange) + 4 * p + 0);

				if (memcmpLike(haystack + 4 * p + 1, subRange))
					return mainRange + ((haystack - mainRange) + 4 * p + 1);

				if (memcmpLike(haystack + 4 * p + 2, subRange))
					return mainRange + ((haystack - mainRange) + 4 * p + 2);

				if (memcmpLike(haystack + 4 * p + 3, subRange))
					return mainRange + ((haystack - mainRange) + 4 * p + 3);

				mask = ClearLeftMostSet(mask);
			}
		}

		return nullptr;
	}
	
	template <typename _Type_>
	static simd_stl_declare_const_function const _Type_* AnySize(
		const _Type_* mainRange,
		const sizetype	mainLength,
		const _Type_* subRange,
		const sizetype	subLength) noexcept
	{
		if (mainLength <= 0 || subLength <= 0)
			return nullptr;

		const auto first = _mm512_set1_epu8(subRange[0]);
		const auto last = _mm512_set1_epu8(subRange[subLength - 1]);

		_Type_* haystack = const_cast<_Type_*>(mainRange);
		_Type_* end = haystack + mainLength;

		for (; haystack < end; haystack += 64) {

			const auto blockFirst = _mm512_loadu_si512(haystack + 0);
			const auto blockLast = _mm512_loadu_si512(haystack + subLength - 1);

			const auto firstZeros = _mm512_xor_si512(blockFirst, first);

			const auto zeros = _mm512_ternarylogic_epi32(firstZeros, blockLast, last, 0xf6);
			uint32_t mask = ZeroByteMask(zeros);

			while (mask) {

				const uint64_t p = math::CountTrailingZeroBits(mask);

				if (memcmp(haystack + 4 * p + 0, subRange, subLength) == 0)
					return mainRange + ((haystack - mainRange) + 4 * p + 0);

				if (memcmp(haystack + 4 * p + 1, subRange, subLength) == 0)
					return mainRange + ((haystack - mainRange) + 4 * p + 1);

				if (memcmp(haystack + 4 * p + 2, subRange, subLength) == 0)
					return mainRange + ((haystack - mainRange) + 4 * p + 2);

				if (memcmp(haystack + 4 * p + 3, subRange, subLength) == 0)
					return mainRange + ((haystack - mainRange) + 4 * p + 3);

				mask = math::ClearLeftMostSet(mask);
			}
		}

		return nullptr;
	}
};

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

		const auto first	= Set(subRange[0]);
		const auto last		= Set(subRange[needleLength - 1]);

		for (sizetype i = 0; i < mainLength; i += 32) {
			const auto blockFirst	= _mm256_lddqu_si256(reinterpret_cast<const __m256i*>(mainRange + i));
			const auto blockLast	= _mm256_lddqu_si256(reinterpret_cast<const __m256i*>(mainRange + i + needleLength - 1));

			const auto equalFirst	= Compare(first, blockFirst);
			const auto equalLast	= Compare(last, blockLast);

			uint32_t mask = _mm256_movemask_epi8(_mm256_and_si256(equalFirst, equalLast));

			while (mask != 0) {
				const auto bitpos = math::CountTrailingZeroBits(mask);

				if (memcmpLike(mainRange + i + bitpos + 1, subRange + 1))
					return mainRange + i + bitpos;

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

		const auto first	= Set(subRange[0]);
		const auto last		= Set(subRange[subLength - 1]);

		for (sizetype i = 0; i < mainLength; i += 32) {
			const auto blockFirst	= _mm256_lddqu_si256(reinterpret_cast<const __m256i*>(mainRange + i));
			const auto blockLast	= _mm256_lddqu_si256(reinterpret_cast<const __m256i*>(mainRange + i + subLength - 1));

			const auto equalFirst	= Compare(first, blockFirst);
			const auto equalLast	= Compare(last, blockLast);

			uint32_t mask = _mm256_movemask_epi8(_mm256_and_si256(equalFirst, equalLast));

			while (mask != 0) {
				const auto bitpos = math::CountTrailingZeroBits(mask);

				if (memcmp(mainRange + i + bitpos + 1, subRange + 1, subLength - 2) == 0)
					return mainRange + i + bitpos;

				mask = math::ClearLeftMostSet(mask);
			}
		}

		return nullptr;
	}

	template <
		sizetype needleLength,
		typename _Type_>
	static simd_stl_declare_const_function const _Type_* Equal(
		const _Type_*	string,
		sizetype		stringLength,
		const _Type_*	needle) noexcept
	{
		static_assert(needleLength > 0 && needleLength < 16, "needleLength must be in range [1..15]");

		if (stringLength <= 0)
			return nullptr;

		__m256i broadcasted[needleLength];

		for (unsigned i = 0; i < needleLength; i++)
			broadcasted[i] = Set(needle[i]);

		auto curr = _mm256_lddqu_si256(reinterpret_cast<const __m256i*>(string));

		for (sizetype i = 0; i < needleLength; i += 32) {
			const auto next = _mm256_lddqu_si256(reinterpret_cast<const __m256i*>(string + i + 32));
			auto equal		= Compare(curr, broadcasted[0]);

			__m256i next1;

			next1 = _mm256_inserti128_si256(next1, _mm256_extracti128_si256(curr, 1), 0); // b
			next1 = _mm256_inserti128_si256(next1, _mm256_extracti128_si256(next, 0), 1); // c

#if !defined(simd_stl_cpp_clang) && !defined(simd_stl_cpp_msvc)
			for (unsigned i = 1; i < needleLength; i++) {
				const auto subrange = _mm256_alignr_epi8(next1, curr, i);
				equal = _mm256_and_si256(equal, _mm256_cmpeq_epi8(subrange, broadcasted[i]));
			}
#else
			StringFindLoop<needleLength>()(equal, next1, curr, broadcasted);
#endif

			curr = next;
			const uint32_t mask = ToMask(equal);

			if (mask != 0)
				return string + i + math::CountTrailingZeroBits(mask);
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

		const auto first	= Set(subRange[0]);
		const auto last		= Set(subRange[needleLength - 1]);

		for (sizetype i = 0; i < mainLength; i += 16) {

			const auto blockFirst	= _mm_loadu_si128(reinterpret_cast<const __m128i*>(mainRange + i));
			const auto blockLast	= _mm_loadu_si128(reinterpret_cast<const __m128i*>(mainRange + i + needleLength - 1));

			const auto equalFirst	= Compare(first, blockFirst);
			const auto equalLast	= Compare(last, blockLast);

			uint32_t mask = _mm_movemask_epi8(_mm_and_si128(equalFirst, equalLast));

			while (mask != 0) {

				const auto bitpos = math::CountTrailingZeroBits(mask);

				if (memcmpLike(mainRange + i + bitpos + 1, subRange + 1))
					return mainRange + i + bitpos;

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

		const auto first	= Set(subRange[0]);
		const auto last		= Set(subRange[subLength - 1]);

		for (sizetype i = 0; i < mainLength; i += 16) {

			const auto blockFirst	= _mm_loadu_si128(reinterpret_cast<const __m128i*>(mainRange + i));
			const auto blockLast	= _mm_loadu_si128(reinterpret_cast<const __m128i*>(mainRange + i + subLength - 1));

			const auto equalFirst	= Compare(first, blockFirst);
			const auto equalLast	= Compare(last, blockLast);

			uint16_t mask = _mm_movemask_epi8(_mm_and_si128(equalFirst, equalLast));

			while (mask != 0) {
				const auto bitpos = math::CountTrailingZeroBits(mask);

				if (memcmp(mainRange + i + bitpos + 1, subRange + 1, subLength - 2) == 0)
					return mainRange + i + bitpos;

				mask = math::ClearLeftMostSet(mask);
			}
		}

		return nullptr;
	}
};

__SIMD_STL_ALGORITHM_NAMESPACE_END
