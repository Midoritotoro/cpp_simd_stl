#pragma once 

__SIMD_STL_DATAPAR_NAMESPACE_BEGIN

#pragma region Sse2-Sse4.2 Simd broadcast

template <
	class _DesiredType_,
	class _VectorType_>
__simd_nodiscard_inline _VectorType_ __simd_broadcast_implementation<
	arch::CpuFeature::SSE2, datapar::xmm128>::__broadcast(_DesiredType_ __value) noexcept
{
	if constexpr (__is_epi64_v<_DesiredType_> || __is_epu64_v<_DesiredType_>)
		return __intrin_bitcast<_VectorType_>(_mm_set1_epi64x(memory::pointer_to_integral(__value)));

	else if constexpr (__is_epi32_v<_DesiredType_> || __is_epu32_v<_DesiredType_>)
		return __intrin_bitcast<_VectorType_>(_mm_set1_epi32(memory::pointer_to_integral(__value)));

	else if constexpr (__is_epi16_v<_DesiredType_> || __is_epu16_v<_DesiredType_>)
		return __intrin_bitcast<_VectorType_>(_mm_set1_epi16(__value));

	else if constexpr (__is_epi8_v<_DesiredType_> || __is_epu8_v<_DesiredType_>)
		return __intrin_bitcast<_VectorType_>(_mm_set1_epi8(__value));

	else if constexpr (__is_ps_v<_DesiredType_>)
		return __intrin_bitcast<_VectorType_>(_mm_set1_ps(__value));

	else if constexpr (__is_pd_v<_DesiredType_>)
		return __intrin_bitcast<_VectorType_>(_mm_set1_pd(__value));
}

template <class _VectorType_>
__simd_nodiscard_inline _VectorType_ __simd_broadcast_implementation<
	arch::CpuFeature::SSE2, datapar::xmm128>::__broadcast_zeros() noexcept 
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
__simd_nodiscard_inline _VectorType_ __simd_broadcast_implementation<
	arch::CpuFeature::AVX, datapar::ymm256>::__broadcast(_DesiredType_ __value) noexcept
{
	if constexpr (__is_epi64_v<_DesiredType_> || __is_epu64_v<_DesiredType_>)
		return __intrin_bitcast<_VectorType_>(_mm256_set1_epi64x(memory::pointer_to_integral(__value)));

	else if constexpr (__is_epi32_v<_DesiredType_> || __is_epu32_v<_DesiredType_>)
		return __intrin_bitcast<_VectorType_>(_mm256_set1_epi32(memory::pointer_to_integral(__value)));

	else if constexpr (__is_epi16_v<_DesiredType_> || __is_epu16_v<_DesiredType_>)
		return __intrin_bitcast<_VectorType_>(_mm256_set1_epi16(__value));

	else if constexpr (__is_epi8_v<_DesiredType_> || __is_epu8_v<_DesiredType_>)
		return __intrin_bitcast<_VectorType_>(_mm256_set1_epi8(__value));

	else if constexpr (__is_ps_v<_DesiredType_>)
		return __intrin_bitcast<_VectorType_>(_mm256_set1_ps(__value));

	else if constexpr (__is_pd_v<_DesiredType_>)
		return __intrin_bitcast<_VectorType_>(_mm256_set1_pd(__value));
}

template <class _VectorType_>
__simd_nodiscard_inline _VectorType_ __simd_broadcast_implementation<
	arch::CpuFeature::AVX, datapar::ymm256>::__broadcast_zeros() noexcept
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
__simd_nodiscard_inline _VectorType_ __simd_broadcast_implementation<
	arch::CpuFeature::AVX2, ymm256>::__broadcast(_DesiredType_ __value) noexcept
{
	if constexpr (__is_epi64_v<_DesiredType_> || __is_epu64_v<_DesiredType_>)
#if !defined(simd_stl_os_win64)
		return __intrin_bitcast<_VectorType_>(_mm256_set1_epi64x(memory::pointer_to_integral(__value)));
#else
		return __intrin_bitcast<_VectorType_>(_mm256_broadcastq_epi64(_mm_cvtsi64_si128(memory::pointer_to_integral(__value))));
#endif // !defined(simd_stl_os_win64)

	else if constexpr (__is_epi32_v<_DesiredType_> || __is_epu32_v<_DesiredType_>)
		return __intrin_bitcast<_VectorType_>(_mm256_broadcastd_epi32(_mm_cvtsi32_si128(memory::pointer_to_integral(__value))));

	else if constexpr (__is_epi16_v<_DesiredType_> || __is_epu16_v<_DesiredType_>)
		return __intrin_bitcast<_VectorType_>(_mm256_set1_epi16(__value));

	else if constexpr (__is_epi8_v<_DesiredType_> || __is_epu8_v<_DesiredType_>)
		return __intrin_bitcast<_VectorType_>(_mm256_set1_epi8(__value));

	else if constexpr (__is_ps_v<_DesiredType_>)
		return __intrin_bitcast<_VectorType_>(_mm256_set1_ps(__value));

	else if constexpr (__is_pd_v<_DesiredType_>)
		return __intrin_bitcast<_VectorType_>(_mm256_set1_pd(__value));
}

#pragma endregion

#pragma region Avx512 Simd broadcast

template <
	class _DesiredType_,
	class _VectorType_>
__simd_nodiscard_inline _VectorType_ __simd_broadcast_implementation<
	arch::CpuFeature::AVX512F, datapar::zmm512>::__broadcast(_DesiredType_ __value) noexcept 
{
	if constexpr (__is_epi64_v<_DesiredType_> || __is_epu64_v<_DesiredType_>)
#if !defined(simd_stl_os_win64)
		return __intrin_bitcast<_VectorType_>(_mm512_set1_epi64(memory::pointer_to_integral(__value)));
#else
		return __intrin_bitcast<_VectorType_>(_mm512_broadcastq_epi64(_mm_cvtsi64_si128(memory::pointer_to_integral(__value))));
#endif

	else if constexpr (__is_epi32_v<_DesiredType_> || __is_epu32_v<_DesiredType_>)
		return __intrin_bitcast<_VectorType_>(_mm512_broadcastd_epi32(_mm_cvtsi32_si128(memory::pointer_to_integral(__value))));

	else if constexpr (__is_epi16_v<_DesiredType_> || __is_epu16_v<_DesiredType_>)
		return __intrin_bitcast<_VectorType_>(_mm512_set1_epi16(__value));

	else if constexpr (__is_epi8_v<_DesiredType_> || __is_epu8_v<_DesiredType_>)
		return __intrin_bitcast<_VectorType_>(_mm512_set1_epi8(__value));

	else if constexpr (__is_ps_v<_DesiredType_>)
		return __intrin_bitcast<_VectorType_>(_mm512_set1_ps(__value));

	else if constexpr (__is_pd_v<_DesiredType_>)
		return __intrin_bitcast<_VectorType_>(_mm512_set1_pd(__value));
}

template <class _VectorType_>
__simd_nodiscard_inline _VectorType_ __simd_broadcast_implementation<
	arch::CpuFeature::AVX512F, datapar::zmm512>::__broadcast_zeros() noexcept
{
	if constexpr (std::is_same_v<_VectorType_, __m512i>)
		return _mm512_setzero_si512();

	else if constexpr (std::is_same_v<_VectorType_, __m512d>)
		return _mm512_setzero_pd();

	else if constexpr (std::is_same_v<_VectorType_, __m512>)
		return _mm512_setzero_ps();
}

#pragma endregion

__SIMD_STL_DATAPAR_NAMESPACE_END
