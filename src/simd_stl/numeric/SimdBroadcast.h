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
	static simd_stl_nodiscard simd_stl_always_inline _VectorType_ _Broadcast(_DesiredType_ _Value) noexcept;

	template <class _VectorType_>
	static simd_stl_nodiscard simd_stl_always_inline _VectorType_ _BroadcastZeros() noexcept;
};

template <>
class _SimdBroadcastImplementation<arch::CpuFeature::SSE3, xmm128>:
	public _SimdBroadcastImplementation<arch::CpuFeature::SSE2, xmm128>
{};

template <>
class _SimdBroadcastImplementation<arch::CpuFeature::SSSE3, xmm128>:
	public _SimdBroadcastImplementation<arch::CpuFeature::SSE3, xmm128>
{};

template <>
class _SimdBroadcastImplementation<arch::CpuFeature::SSE41, xmm128>:
	public _SimdBroadcastImplementation<arch::CpuFeature::SSSE3, xmm128>
{};

template <>
class _SimdBroadcastImplementation<arch::CpuFeature::SSE42, xmm128>:
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
	static simd_stl_nodiscard simd_stl_always_inline _VectorType_ _Broadcast(_DesiredType_ _Value) noexcept;

	template <class _VectorType_>
	static simd_stl_nodiscard simd_stl_always_inline _VectorType_ _BroadcastZeros() noexcept;
};

template <>
class _SimdBroadcastImplementation<arch::CpuFeature::AVX2, ymm256>:
	public _SimdBroadcastImplementation<arch::CpuFeature::AVX, ymm256>
{
public:
	template <
		class _DesiredType_,
		class _VectorType_>
	static simd_stl_nodiscard simd_stl_always_inline _VectorType_ _Broadcast(_DesiredType_ _Value) noexcept;
};

#pragma endregion

#pragma region Avx512 Simd broadcast

template <>
class _SimdBroadcastImplementation<arch::CpuFeature::AVX512F, numeric::zmm512> {
public:
	template <
		class _DesiredType_,
		class _VectorType_>
	static simd_stl_nodiscard simd_stl_always_inline _VectorType_ _Broadcast(_DesiredType_ _Value) noexcept;

	template <class _VectorType_>
	static simd_stl_nodiscard simd_stl_always_inline _VectorType_ _BroadcastZeros() noexcept;
};

template <>
class _SimdBroadcastImplementation<arch::CpuFeature::AVX512BW, zmm512> :
	public _SimdBroadcastImplementation<arch::CpuFeature::AVX512F, zmm512>
{};

template <>
class _SimdBroadcastImplementation<arch::CpuFeature::AVX512DQ, zmm512> :
	public _SimdBroadcastImplementation<arch::CpuFeature::AVX512BW, zmm512>
{};

template <>
class _SimdBroadcastImplementation<arch::CpuFeature::AVX512VLF, ymm256> :
	public _SimdBroadcastImplementation<arch::CpuFeature::AVX2, ymm256>
{};

template <>
class _SimdBroadcastImplementation<arch::CpuFeature::AVX512VLBW, ymm256> :
	public _SimdBroadcastImplementation<arch::CpuFeature::AVX512VLF, ymm256>
{};

template <>
class _SimdBroadcastImplementation<arch::CpuFeature::AVX512VLDQ, ymm256> :
	public _SimdBroadcastImplementation<arch::CpuFeature::AVX512VLBW, ymm256>
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

#include <src/simd_stl/numeric/SimdBroadcast.inl>
