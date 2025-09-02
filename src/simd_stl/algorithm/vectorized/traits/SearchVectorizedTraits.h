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


/*
	The following templates implement the loop, where K is a template parameter.

		for (unsigned i=1; i < K; i++) {
			const __m256i substring = _mm256_alignr_epi8(next1, curr, i);
			eq = _mm256_and_si256(eq, _mm256_cmpeq_epi8(substring, broadcasted[i]));
		}

	Clang and MSVC complains that the loop parameter `i` is a variable and it cannot be
	applied as a parameter _mm256_alignr_epi8.  GCC somehow deals with it.
*/

simd_stl_always_inline __mmask16 ZeroByteMask(const __m512i vector) noexcept {
	const auto vector01 = _mm512_set1_epi8(0x01);
	const auto vector80 = _mm512_set1_epi8(int8(0x80));

	const auto vector1 = _mm512_sub_epi32(vector, vector01);
	// tempVector1 = (vector - 0x01010101) & ~vector & 0x80808080
	const auto tempVector1 = _mm512_ternarylogic_epi32(vector1, vector, vector80, 0x20);

	return _mm512_test_epi32_mask(tempVector1, tempVector1);
}

simd_stl_always_inline bool IsXmmZero(__m128i xmmRegister) noexcept {
	return _mm_movemask_epi8(_mm_cmpeq_epi64(xmmRegister, _mm_setzero_si128())) == 0xFFFFFFFFFFFFFFFF;
}

simd_stl_always_inline bool IsYmmZero(__m256i ymmRegister) noexcept {
	return _mm256_cmpeq_epi64_mask(ymmRegister, _mm256_setzero_si256()) == 0xFFFFFFFFFFFFFFFF;
}

simd_stl_always_inline bool IsZmmZero(__m512i zmmRegister) noexcept {
	return _mm512_cmpeq_epi64_mask(zmmRegister, _mm512_setzero_si512()) == 0xFFFFFFFFFFFFFFFF;
}

#if defined(simd_stl_cpp_clang) || defined(simd_stl_cpp_msvc)

template <
	sizetype    K,
	int         i,
	bool        terminate>
struct StringFindLoopImplementation;

template <
	sizetype K,
	int i>
struct StringFindLoopImplementation<K, i, false> {
	void operator()(
		__m256i& eq,
		const __m256i& next1,
		const __m256i& curr,
		const __m256i   (&broadcasted)[K])
	{
		const __m256i substring = _mm256_alignr_epi8(next1, curr, i);
		eq = _mm256_and_si256(eq, _mm256_cmpeq_epi8(substring, broadcasted[i]));

		StringFindLoopImplementation<K, i + 1, i + 1 == K>()(eq, next1, curr, broadcasted);
	}
};

template <
	sizetype    K,
	int         i>
struct StringFindLoopImplementation<K, i, true> {
	void operator()(
		__m256i&,
		const __m256i&,
		const __m256i&,
		const __m256i (&)[K])
	{
		// nop
	}
};

template <sizetype K>
struct StringFindLoop {
	void operator()(
		__m256i& eq,
		const __m256i& next1,
		const __m256i& curr,
		const __m256i   (&broadcasted)[K])
	{
		static_assert(K > 0, "wrong value");
		StringFindLoopImplementation<K, 0, false>()(eq, next1, curr, broadcasted);
	}
};

#endif

template <>
class SearchTraits<arch::CpuFeature::AVX512F>: public FindTraits<arch::CpuFeature::AVX512F> 
{
public:
	template <
		sizetype needleLength,
		typename _Type_,
		typename _MemCmpLike_>
	static simd_stl_declare_const_function const _Type_* Memcmp(
		const _Type_*		mainString,
		const sizetype	mainLength,
		const _Type_*		subString,
		_MemCmpLike_	memcmpLike) noexcept
	{
		if constexpr (needleLength <= 0)
			return nullptr;

		if (mainLength <= 0)
			return nullptr;

		const auto first = _mm512_set1_epi8(subString[0]);
		const auto last = _mm512_set1_epi8(subString[needleLength - 1]);

		_Type_* haystack = const_cast<_Type_*>(mainString);
		_Type_* end = haystack + mainLength;

		for (; haystack < end; haystack += 64) {

			const auto block_first = _mm512_loadu_si512(haystack + 0);
			const auto block_last = _mm512_loadu_si512(haystack + needleLength - 1);

			const auto first_zeros = _mm512_xor_si512(block_first, first);
			const auto zeros = _mm512_ternarylogic_epi32(first_zeros, block_last, last, 0xf6);

			uint32_t mask = ZeroByteMask(zeros);

			while (mask) {

				const uint64_t p = math::CountTrailingZeroBits(mask);

				if (memcmpLike(haystack + 4 * p + 0, subString))
					return mainString + ((haystack - mainString) + 4 * p + 0);

				if (memcmpLike(haystack + 4 * p + 1, subString))
					return mainString + ((haystack - mainString) + 4 * p + 1);

				if (memcmpLike(haystack + 4 * p + 2, subString))
					return mainString + ((haystack - mainString) + 4 * p + 2);

				if (memcmpLike(haystack + 4 * p + 3, subString))
					return mainString + ((haystack - mainString) + 4 * p + 3);

				mask = ClearLeftMostSet(mask);
			}
		}

		return nullptr;
	}
	
	template <typename _Type_>
	static simd_stl_declare_const_function const char* AnySize(
		const _Type_* mainString,
		const sizetype	mainLength,
		const _Type_* subString,
		const sizetype	subLength) noexcept
	{
		if (mainLength <= 0 || subLength <= 0)
			return nullptr;

		const auto first = _mm512_set1_epu8(subString[0]);
		const auto last = _mm512_set1_epu8(subString[subLength - 1]);

		_Type_* haystack = const_cast<_Type_*>(mainString);
		_Type_* end = haystack + mainLength;

		for (; haystack < end; haystack += 64) {

			const auto blockFirst = _mm512_loadu_si512(haystack + 0);
			const auto blockLast = _mm512_loadu_si512(haystack + subLength - 1);

			const auto firstZeros = _mm512_xor_si512(blockFirst, first);

			/*
				first_zeros | block_last | last |  first_zeros | (block_last ^ last)
				------------+------------+------+------------------------------------
					 0      |      0     |   0  |      0
					 0      |      0     |   1  |      1
					 0      |      1     |   0  |      1
					 0      |      1     |   1  |      0
					 1      |      0     |   0  |      1
					 1      |      0     |   1  |      1
					 1      |      1     |   0  |      1
					 1      |      1     |   1  |      1
			*/

			const auto zeros = _mm512_ternarylogic_epi32(firstZeros, blockLast, last, 0xf6);
			uint32_t mask = ZeroByteMask(zeros);

			while (mask) {

				const uint64_t p = math::CountTrailingZeroBits(mask);

				if (memcmp(haystack + 4 * p + 0, subString, subLength) == 0)
					return mainString + ((haystack - mainString) + 4 * p + 0);

				if (memcmp(haystack + 4 * p + 1, subString, subLength) == 0)
					return mainString + ((haystack - mainString) + 4 * p + 1);

				if (memcmp(haystack + 4 * p + 2, subString, subLength) == 0)
					return mainString + ((haystack - mainString) + 4 * p + 2);

				if (memcmp(haystack + 4 * p + 3, subString, subLength) == 0)
					return mainString + ((haystack - mainString) + 4 * p + 3);

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
		const char*		mainString,
		const sizetype	mainLength,
		const char*		subString,
		_MemCmpLike_	memcmpLike) noexcept
	{
		if constexpr (needleLength <= 0)
			return nullptr;

		if (mainLength <= 0)
			return nullptr;

		const auto first	= Set(subString[0]);
		const auto last		= Set(subString[needleLength - 1]);

		for (sizetype i = 0; i < mainLength; i += 32) {
			const auto blockFirst	= _mm256_lddqu_si256(reinterpret_cast<const __m256i*>(mainString + i));
			const auto blockLast	= _mm256_lddqu_si256(reinterpret_cast<const __m256i*>(mainString + i + needleLength - 1));

			const auto equalFirst	= Compare(first, blockFirst);
			const auto equalLast	= Compare(last, blockLast);

			uint32_t mask = _mm256_movemask_epi8(_mm256_and_si256(equalFirst, equalLast));

			while (mask != 0) {
				const auto bitpos = math::CountTrailingZeroBits(mask);

				if (memcmpLike(mainString + i + bitpos + 1, subString + 1))
					return mainString + i + bitpos;

				mask = math::ClearLeftMostSet(mask);
			}
		}

		return nullptr;
	}

	template <typename _Type_>
	static simd_stl_declare_const_function const _Type_* AnySize(
		const _Type_*	mainString,
		const sizetype	mainLength,
		const _Type_*	subString,
		const sizetype	subLength) noexcept
	{
		if (subLength <= 0 || mainLength <= 0)
			return nullptr;

		const auto first	= Set(subString[0]);
		const auto last		= Set(subString[subLength - 1]);

		for (sizetype i = 0; i < mainLength; i += 32) {
			const auto blockFirst	= _mm256_lddqu_si256(reinterpret_cast<const __m256i*>(mainString + i));
			const auto blockLast	= _mm256_lddqu_si256(reinterpret_cast<const __m256i*>(mainString + i + subLength - 1));

			const auto equalFirst	= Compare(first, blockFirst);
			const auto equalLast	= Compare(last, blockLast);

			uint32_t mask = _mm256_movemask_epi8(_mm256_and_si256(equalFirst, equalLast));

			while (mask != 0) {
				const auto bitpos = math::CountTrailingZeroBits(mask);

				if (memcmp(mainString + i + bitpos + 1, subString + 1, subLength - 2) == 0)
					return mainString + i + bitpos;

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
				const auto substring = _mm256_alignr_epi8(next1, curr, i);
				equal = _mm256_and_si256(equal, _mm256_cmpeq_epi8(substring, broadcasted[i]));
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
		const _Type_*	mainString,
		const sizetype	mainLength,
		const _Type_*	subString,
		_MemCmpLike_	memcmpLike) noexcept
	{
		if constexpr (needleLength <= 0)
			return nullptr;

		if (mainLength <= 0)
			return nullptr;

		const auto first	= Set(subString[0]);
		const auto last		= Set(subString[needleLength - 1]);

		for (sizetype i = 0; i < mainLength; i += 16) {

			const auto blockFirst	= _mm_loadu_si128(reinterpret_cast<const __m128i*>(mainString + i));
			const auto blockLast	= _mm_loadu_si128(reinterpret_cast<const __m128i*>(mainString + i + needleLength - 1));

			const auto equalFirst	= Compare(first, blockFirst);
			const auto equalLast	= Compare(last, blockLast);

			uint32_t mask = _mm_movemask_epi8(_mm_and_si128(equalFirst, equalLast));

			while (mask != 0) {

				const auto bitpos = math::CountTrailingZeroBits(mask);

				if (memcmpLike(mainString + i + bitpos + 1, subString + 1))
					return mainString + i + bitpos;

				mask = math::ClearLeftMostSet(mask);
			}
		}

		return nullptr;
	}
	
	template <typename _Type_>
	static simd_stl_declare_const_function const char* AnySize(
		const _Type_*	mainString,
		const sizetype	mainLength,
		const _Type_*	subString,
		const sizetype	subLength) noexcept
	{
		if (mainLength <= 0 || subLength <= 0)
			return nullptr;

		const auto first	= Set(subString[0]);
		const auto last		= Set(subString[subLength - 1]);

		for (sizetype i = 0; i < mainLength; i += 16) {

			const auto blockFirst	= _mm_loadu_si128(reinterpret_cast<const __m128i*>(mainString + i));
			const auto blockLast	= _mm_loadu_si128(reinterpret_cast<const __m128i*>(mainString + i + subLength - 1));

			const auto equalFirst	= Compare(first, blockFirst);
			const auto equalLast	= Compare(last, blockLast);

			uint16_t mask = _mm_movemask_epi8(_mm_and_si128(equalFirst, equalLast));

			while (mask != 0) {
				const auto bitpos = math::CountTrailingZeroBits(mask);

				if (memcmp(mainString + i + bitpos + 1, subString + 1, subLength - 2) == 0)
					return mainString + i + bitpos;

				mask = math::ClearLeftMostSet(mask);
			}
		}

		return nullptr;
	}
};

__SIMD_STL_ALGORITHM_NAMESPACE_END
