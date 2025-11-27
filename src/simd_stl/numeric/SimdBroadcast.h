#pragma once 

#include <src/simd_stl/numeric/IntrinBitcast.h>
#include <simd_stl/arch/CpuFeature.h>

#include <simd_stl/memory/PointerToIntegral.h>


__SIMD_STL_NUMERIC_NAMESPACE_BEGIN

template <
	arch::CpuFeature	_SimdGeneration,
	class				_RegisterPolicy_>
class _SimdBroadcastImplementation;

#pragma region Sse2-Sse4.2 Simd broadcast

template <>
class _SimdBroadcastImplementation<arch::CpuFeature::SSE2, numeric::xmm128> {
public:
	template <
		class _DesiredType_,
		class _VectorType_>
	static simd_stl_nodiscard simd_stl_always_inline _VectorType_ _Broadcast(_DesiredType_ _Value) noexcept {
		if constexpr (_Is_epi64_v<_DesiredType_> || _Is_epu64_v<_DesiredType_>)
			return _IntrinBitcast<_VectorType_>(_mm_set1_epi64x(memory::pointerToIntegral(_Value)));

		else if constexpr (_Is_epi32_v<_DesiredType_> || _Is_epu32_v<_DesiredType_>)
			return _IntrinBitcast<_VectorType_>(_mm_set1_epi32(memory::pointerToIntegral(_Value)));

		else if constexpr (_Is_epi16_v<_DesiredType_> || _Is_epu16_v<_DesiredType_>)
			return _IntrinBitcast<_VectorType_>(_mm_set1_epi16(_Value));

		else if constexpr (_Is_epi8_v<_DesiredType_> || _Is_epu8_v<_DesiredType_>)
			return _IntrinBitcast<_VectorType_>(_mm_set1_epi8(_Value));

		else if constexpr (_Is_ps_v<_DesiredType_>)
			return _IntrinBitcast<_VectorType_>(_mm_set1_ps(_Value));

		else if constexpr (_Is_pd_v<_DesiredType_>)
			return _IntrinBitcast<_VectorType_>(_mm_set1_pd(_Value));
	}

	template <class _VectorType_>
	static simd_stl_nodiscard simd_stl_always_inline _VectorType_ _BroadcastZeros() noexcept {
		if constexpr (std::is_same_v<_VectorType_, __m128i>)
			return _mm_setzero_si128();

		else if constexpr (std::is_same_v<_VectorType_, __m128d>)
			return _mm_setzero_pd();

		else if constexpr (std::is_same_v<_VectorType_, __m128>)
			return _mm_setzero_ps();
	}
};

template <>
class _SimdBroadcastImplementation<arch::CpuFeature::SSE3, xmm128> :
	public _SimdBroadcastImplementation<arch::CpuFeature::SSE2, xmm128>
{};

template <>
class _SimdBroadcastImplementation<arch::CpuFeature::SSSE3, xmm128> :
	public _SimdBroadcastImplementation<arch::CpuFeature::SSE3, xmm128>
{};

template <>
class _SimdBroadcastImplementation<arch::CpuFeature::SSE41, xmm128> :
	public _SimdBroadcastImplementation<arch::CpuFeature::SSSE3, xmm128>
{};

template <>
class _SimdBroadcastImplementation<arch::CpuFeature::SSE42, xmm128> :
	public _SimdBroadcastImplementation<arch::CpuFeature::SSE41, xmm128>
{};

#pragma endregion

#pragma region Avx-Avx2 Simd broadcast

template <>
class _SimdBroadcastImplementation<arch::CpuFeature::AVX, numeric::ymm256> {
public:
	template <
		class _DesiredType_,
		class _VectorType_>
	static simd_stl_nodiscard simd_stl_always_inline _VectorType_ _Broadcast(_DesiredType_ _Value) noexcept {
		if constexpr (_Is_epi64_v<_DesiredType_> || _Is_epu64_v<_DesiredType_>)
			return _IntrinBitcast<_VectorType_>(_mm256_set1_epi64x(memory::pointerToIntegral(_Value)));

		else if constexpr (_Is_epi32_v<_DesiredType_> || _Is_epu32_v<_DesiredType_>)
			return _IntrinBitcast<_VectorType_>(_mm256_set1_epi32(memory::pointerToIntegral(_Value)));

		else if constexpr (_Is_epi16_v<_DesiredType_> || _Is_epu16_v<_DesiredType_>)
			return _IntrinBitcast<_VectorType_>(_mm256_set1_epi16(_Value));

		else if constexpr (_Is_epi8_v<_DesiredType_> || _Is_epu8_v<_DesiredType_>)
			return _IntrinBitcast<_VectorType_>(_mm256_set1_epi8(_Value));

		else if constexpr (_Is_ps_v<_DesiredType_>)
			return _IntrinBitcast<_VectorType_>(_mm256_set1_ps(_Value));

		else if constexpr (_Is_pd_v<_DesiredType_>)
			return _IntrinBitcast<_VectorType_>(_mm256_set1_pd(_Value));
	}

	template <class _VectorType_>
	static simd_stl_nodiscard simd_stl_always_inline _VectorType_ _BroadcastZeros() noexcept {
		if constexpr (std::is_same_v<_VectorType_, __m256>)
			return _mm256_setzero_si256();

		else if constexpr (std::is_same_v<_VectorType_, __m256d>)
			return _mm256_setzero_pd();

		else if constexpr (std::is_same_v<_VectorType_, __m256>)
			return _mm256_setzero_ps();
	}
};

template <>
class _SimdBroadcastImplementation<arch::CpuFeature::AVX2, ymm256> :
	public _SimdBroadcastImplementation<arch::CpuFeature::AVX, ymm256>
{};

#pragma endregion

template <
	arch::CpuFeature	_SimdGeneration_,
	class				_RegisterPolicy_,
	class				_VectorType_,
	class				_DesiredType_>
simd_stl_nodiscard simd_stl_always_inline _VectorType_ _SimdBroadcast(_DesiredType_ _Value) noexcept {
	_VerifyRegisterPolicy(_SimdGeneration_, _RegisterPolicy_);
	return _SimdBroadcastImplementation<_SimdGeneration_, _RegisterPolicy_>
		::template _Broadcast<_DesiredType_, _VectorType_>(_Value);
}

template <
	arch::CpuFeature	_SimdGeneration_,
	class				_RegisterPolicy_,
	class				_VectorType_>
simd_stl_nodiscard simd_stl_always_inline _VectorType_ _SimdBroadcastZeros() noexcept {
	_VerifyRegisterPolicy(_SimdGeneration_, _RegisterPolicy_);
	return _SimdBroadcastImplementation<_SimdGeneration_, _RegisterPolicy_>::template _BroadcastZeros<_VectorType_>();
}

__SIMD_STL_NUMERIC_NAMESPACE_END
