#pragma once 

__SIMD_STL_NUMERIC_NAMESPACE_BEGIN

#pragma region Sse2-Sse4.2 Simd broadcast

template <
	class _DesiredType_,
	class _VectorType_>
__simd_nodiscard_inline _VectorType_ _SimdBroadcastImplementation<
	arch::CpuFeature::SSE2, numeric::xmm128>::_Broadcast(_DesiredType_ _Value) noexcept
{
	if constexpr (_Is_epi64_v<_DesiredType_> || _Is_epu64_v<_DesiredType_>)
		return __intrin_bitcast<_VectorType_>(_mm_set1_epi64x(memory::pointerToIntegral(_Value)));

	else if constexpr (_Is_epi32_v<_DesiredType_> || _Is_epu32_v<_DesiredType_>)
		return __intrin_bitcast<_VectorType_>(_mm_set1_epi32(memory::pointerToIntegral(_Value)));

	else if constexpr (_Is_epi16_v<_DesiredType_> || _Is_epu16_v<_DesiredType_>)
		return __intrin_bitcast<_VectorType_>(_mm_set1_epi16(_Value));

	else if constexpr (_Is_epi8_v<_DesiredType_> || _Is_epu8_v<_DesiredType_>)
		return __intrin_bitcast<_VectorType_>(_mm_set1_epi8(_Value));

	else if constexpr (_Is_ps_v<_DesiredType_>)
		return __intrin_bitcast<_VectorType_>(_mm_set1_ps(_Value));

	else if constexpr (_Is_pd_v<_DesiredType_>)
		return __intrin_bitcast<_VectorType_>(_mm_set1_pd(_Value));
}

template <class _VectorType_>
__simd_nodiscard_inline _VectorType_ _SimdBroadcastImplementation<
	arch::CpuFeature::SSE2, numeric::xmm128>::_BroadcastZeros() noexcept 
{
	if constexpr (std::is_same_v<_VectorType_, __m128i>)
		return _mm_setzero_si128();

	else if constexpr (std::is_same_v<_VectorType_, __m128d>)
		return _mm_setzero_pd();

	else if constexpr (std::is_same_v<_VectorType_, __m128>)
		return _mm_setzero_ps();
}

#pragma endregion

#pragma region Avx-Avx2 Simd broadcast

template <
	class _DesiredType_,
	class _VectorType_>
__simd_nodiscard_inline _VectorType_ _SimdBroadcastImplementation<
	arch::CpuFeature::AVX, numeric::ymm256>::_Broadcast(_DesiredType_ _Value) noexcept 
{
	if constexpr (_Is_epi64_v<_DesiredType_> || _Is_epu64_v<_DesiredType_>)
		return __intrin_bitcast<_VectorType_>(_mm256_set1_epi64x(memory::pointerToIntegral(_Value)));

	else if constexpr (_Is_epi32_v<_DesiredType_> || _Is_epu32_v<_DesiredType_>)
		return __intrin_bitcast<_VectorType_>(_mm256_set1_epi32(memory::pointerToIntegral(_Value)));

	else if constexpr (_Is_epi16_v<_DesiredType_> || _Is_epu16_v<_DesiredType_>)
		return __intrin_bitcast<_VectorType_>(_mm256_set1_epi16(_Value));

	else if constexpr (_Is_epi8_v<_DesiredType_> || _Is_epu8_v<_DesiredType_>)
		return __intrin_bitcast<_VectorType_>(_mm256_set1_epi8(_Value));

	else if constexpr (_Is_ps_v<_DesiredType_>)
		return __intrin_bitcast<_VectorType_>(_mm256_set1_ps(_Value));

	else if constexpr (_Is_pd_v<_DesiredType_>)
		return __intrin_bitcast<_VectorType_>(_mm256_set1_pd(_Value));
}

template <class _VectorType_>
__simd_nodiscard_inline _VectorType_ _SimdBroadcastImplementation<
	arch::CpuFeature::AVX, numeric::ymm256>::_BroadcastZeros() noexcept 
{
	if constexpr (std::is_same_v<_VectorType_, __m256i>)
		return _mm256_setzero_si256();

	else if constexpr (std::is_same_v<_VectorType_, __m256d>)
		return _mm256_setzero_pd();

	else if constexpr (std::is_same_v<_VectorType_, __m256>)
		return _mm256_setzero_ps();
}


template <
	class _DesiredType_,
	class _VectorType_>
__simd_nodiscard_inline _VectorType_ _SimdBroadcastImplementation<
	arch::CpuFeature::AVX2, ymm256>::_Broadcast(_DesiredType_ _Value) noexcept
{
	if constexpr (_Is_epi64_v<_DesiredType_> || _Is_epu64_v<_DesiredType_>)
#if !defined(simd_stl_os_win64)
		return _IntrinBitcast<_VectorType_>(_mm256_set1_epi64x(memory::pointerToIntegral(_Value)));
#else
		return __intrin_bitcast<_VectorType_>(_mm256_broadcastq_epi64(_mm_cvtsi64_si128(memory::pointerToIntegral(_Value))));
#endif // !defined(simd_stl_os_win64)

	else if constexpr (_Is_epi32_v<_DesiredType_> || _Is_epu32_v<_DesiredType_>)
		return __intrin_bitcast<_VectorType_>(_mm256_broadcastd_epi32(_mm_cvtsi32_si128(memory::pointerToIntegral(_Value))));

	else if constexpr (_Is_epi16_v<_DesiredType_> || _Is_epu16_v<_DesiredType_>)
		return __intrin_bitcast<_VectorType_>(_mm256_set1_epi16(_Value));

	else if constexpr (_Is_epi8_v<_DesiredType_> || _Is_epu8_v<_DesiredType_>)
		return __intrin_bitcast<_VectorType_>(_mm256_set1_epi8(_Value));

	else if constexpr (_Is_ps_v<_DesiredType_>)
		return __intrin_bitcast<_VectorType_>(_mm256_set1_ps(_Value));

	else if constexpr (_Is_pd_v<_DesiredType_>)
		return __intrin_bitcast<_VectorType_>(_mm256_set1_pd(_Value));
}

#pragma endregion

#pragma region Avx512 Simd broadcast

template <
	class _DesiredType_,
	class _VectorType_>
__simd_nodiscard_inline _VectorType_ _SimdBroadcastImplementation<
	arch::CpuFeature::AVX512F, numeric::zmm512>::_Broadcast(_DesiredType_ _Value) noexcept 
{
	if constexpr (_Is_epi64_v<_DesiredType_> || _Is_epu64_v<_DesiredType_>)
#if !defined(simd_stl_os_win64)
		return _IntrinBitcast<_VectorType_>(_mm512_set1_epi64(memory::pointerToIntegral(_Value)));
#else
		return __intrin_bitcast<_VectorType_>(_mm512_broadcastq_epi64(_mm_cvtsi64_si128(memory::pointerToIntegral(_Value))));
#endif

	else if constexpr (_Is_epi32_v<_DesiredType_> || _Is_epu32_v<_DesiredType_>)
		return __intrin_bitcast<_VectorType_>(_mm512_broadcastd_epi32(_mm_cvtsi32_si128(memory::pointerToIntegral(_Value))));

	else if constexpr (_Is_epi16_v<_DesiredType_> || _Is_epu16_v<_DesiredType_>)
		return __intrin_bitcast<_VectorType_>(_mm512_set1_epi16(_Value));

	else if constexpr (_Is_epi8_v<_DesiredType_> || _Is_epu8_v<_DesiredType_>)
		return __intrin_bitcast<_VectorType_>(_mm512_set1_epi8(_Value));

	else if constexpr (_Is_ps_v<_DesiredType_>)
		return __intrin_bitcast<_VectorType_>(_mm512_set1_ps(_Value));

	else if constexpr (_Is_pd_v<_DesiredType_>)
		return __intrin_bitcast<_VectorType_>(_mm512_set1_pd(_Value));
}

template <class _VectorType_>
__simd_nodiscard_inline _VectorType_ _SimdBroadcastImplementation<
	arch::CpuFeature::AVX512F, numeric::zmm512>::_BroadcastZeros() noexcept
{
	if constexpr (std::is_same_v<_VectorType_, __m512i>)
		return _mm512_setzero_si512();

	else if constexpr (std::is_same_v<_VectorType_, __m512d>)
		return _mm512_setzero_pd();

	else if constexpr (std::is_same_v<_VectorType_, __m512>)
		return _mm512_setzero_ps();
}

#pragma endregion

__SIMD_STL_NUMERIC_NAMESPACE_END
