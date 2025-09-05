#pragma once 

#include <simd_stl/compatibility/SimdCompatibility.h>
#include <simd_stl/compatibility/CompilerDetection.h>

__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

/*
	for (unsigned j = 1; j < K; j++) {
		const __m256i subRange = _mm256_alignr_epi8(next1, curr, j);
		eq = _mm256_and_si256(eq, _mm256_cmpeq_epi8(subRange, broadcasted[j]));
	}
*/


#if defined(simd_stl_cpp_clang) || defined(simd_stl_cpp_msvc)

template <
	typename	_Type_,
	class		_Traits_,
	sizetype    K,
	int         i,
	bool        terminate>
struct StringFindLoopImplementation;

template <
	typename	_Type_,
	class		_Traits_,
	sizetype	K,
	int			i>
struct StringFindLoopImplementation<_Type_, _Traits_, K, i, false> {
	void operator()(
		__m256i& eq,
		const __m256i& next1,
		const __m256i& curr,
		const __m256i   (&broadcasted)[K])
	{
		const __m256i substring = _mm256_alignr_epi8(next1, curr, i);
		eq = _mm256_and_si256(eq, _Traits_::Compare<sizeof(_Type_)>(substring, broadcasted[i]));

		StringFindLoopImplementation<_Type_, _Traits_, K, i + 1, i + 1 == K>()(eq, next1, curr, broadcasted);
	}
};

template <
	typename	_Type_,
	class		_Traits_,
	sizetype    K,
	int         i>
struct StringFindLoopImplementation<_Type_, _Traits_, K, i, true> {
	void operator()(
		__m256i&,
		const __m256i&,
		const __m256i&,
		const __m256i (&)[K])
	{
		// nop
	}
};

template <
	typename	_Type_,
	class		_Traits_,
	sizetype	K>
struct StringFindLoop {
	void operator()(
		__m256i& eq,
		const __m256i& next1,
		const __m256i& curr,
		const __m256i   (&broadcasted)[K])
	{
		static_assert(K > 0);
		StringFindLoopImplementation<_Type_, _Traits_, K, 0, false>()(eq, next1, curr, broadcasted);
	}
};

#endif


__SIMD_STL_ALGORITHM_NAMESPACE_END