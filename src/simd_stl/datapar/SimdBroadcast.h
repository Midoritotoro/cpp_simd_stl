#pragma once 

#include <src/simd_stl/datapar/IntrinBitcast.h>
#include <simd_stl/arch/CpuFeature.h>

#include <simd_stl/memory/PointerToIntegral.h>


__SIMD_STL_DATAPAR_NAMESPACE_BEGIN

template <
	arch::CpuFeature	_SimdGeneration,
	class				_RegisterPolicy_>
class __simd_broadcast_implementation;

#pragma region Sse2-Sse4.2 Simd broadcast

template <>
class __simd_broadcast_implementation<arch::CpuFeature::SSE2, datapar::xmm128> {
public:
	template <
		class _DesiredType_,
		class _VectorType_>
	simd_stl_nodiscard static simd_stl_always_inline _VectorType_ __broadcast(_DesiredType_ __value) noexcept;

	template <class _VectorType_>
	simd_stl_nodiscard static simd_stl_always_inline _VectorType_ __broadcast_zeros() noexcept;
};

template <>
class __simd_broadcast_implementation<arch::CpuFeature::SSE3, xmm128>:
	public __simd_broadcast_implementation<arch::CpuFeature::SSE2, xmm128>
{};

template <>
class __simd_broadcast_implementation<arch::CpuFeature::SSSE3, xmm128>:
	public __simd_broadcast_implementation<arch::CpuFeature::SSE3, xmm128>
{};

template <>
class __simd_broadcast_implementation<arch::CpuFeature::SSE41, xmm128>:
	public __simd_broadcast_implementation<arch::CpuFeature::SSSE3, xmm128>
{};

template <>
class __simd_broadcast_implementation<arch::CpuFeature::SSE42, xmm128>:
	public __simd_broadcast_implementation<arch::CpuFeature::SSE41, xmm128>
{};

#pragma endregion

#pragma region Avx-Avx2 Simd broadcast

template <>
class __simd_broadcast_implementation<arch::CpuFeature::AVX2, xmm128> :
	public __simd_broadcast_implementation<arch::CpuFeature::SSE42, xmm128>
{};

template <>
class __simd_broadcast_implementation<arch::CpuFeature::AVX, datapar::ymm256> {
public:
	template <
		class _DesiredType_,
		class _VectorType_>
	simd_stl_nodiscard static simd_stl_always_inline _VectorType_ __broadcast(_DesiredType_ __value) noexcept;

	template <class _VectorType_>
	simd_stl_nodiscard static simd_stl_always_inline _VectorType_ __broadcast_zeros() noexcept;
};

template <>
class __simd_broadcast_implementation<arch::CpuFeature::AVX2, ymm256>:
	public __simd_broadcast_implementation<arch::CpuFeature::AVX, ymm256>
{
public:
	template <
		class _DesiredType_,
		class _VectorType_>
	simd_stl_nodiscard static simd_stl_always_inline _VectorType_ __broadcast(_DesiredType_ __value) noexcept;
};

#pragma endregion

#pragma region Avx512 Simd broadcast

template <>
class __simd_broadcast_implementation<arch::CpuFeature::AVX512F, datapar::zmm512> {
public:
	template <
		class _DesiredType_,
		class _VectorType_>
	simd_stl_nodiscard static simd_stl_always_inline _VectorType_ __broadcast(_DesiredType_ __value) noexcept;

	template <class _VectorType_>
	simd_stl_nodiscard static simd_stl_always_inline _VectorType_ __broadcast_zeros() noexcept;
};

template <>
class __simd_broadcast_implementation<arch::CpuFeature::AVX512BW, zmm512> :
	public __simd_broadcast_implementation<arch::CpuFeature::AVX512F, zmm512>
{};

template <>
class __simd_broadcast_implementation<arch::CpuFeature::AVX512DQ, zmm512> :
	public __simd_broadcast_implementation<arch::CpuFeature::AVX512F, zmm512>
{};

template <>
class __simd_broadcast_implementation<arch::CpuFeature::AVX512BWDQ, zmm512> :
	public __simd_broadcast_implementation<arch::CpuFeature::AVX512BW, zmm512>
{};

template <>
class __simd_broadcast_implementation<arch::CpuFeature::AVX512VLF, ymm256> :
	public __simd_broadcast_implementation<arch::CpuFeature::AVX2, ymm256>
{};

template <>
class __simd_broadcast_implementation<arch::CpuFeature::AVX512VLBW, ymm256> :
	public __simd_broadcast_implementation<arch::CpuFeature::AVX512VLF, ymm256>
{};

template <>
class __simd_broadcast_implementation<arch::CpuFeature::AVX512VLDQ, ymm256> :
	public __simd_broadcast_implementation<arch::CpuFeature::AVX512VLF, ymm256>
{};

template <>
class __simd_broadcast_implementation<arch::CpuFeature::AVX512VLBWDQ, ymm256> :
	public __simd_broadcast_implementation<arch::CpuFeature::AVX512VLBW, ymm256>
{};


template <>
class __simd_broadcast_implementation<arch::CpuFeature::AVX512VLF, xmm128> :
	public __simd_broadcast_implementation<arch::CpuFeature::SSE42, xmm128>
{};

template <>
class __simd_broadcast_implementation<arch::CpuFeature::AVX512VLBW, xmm128> :
	public __simd_broadcast_implementation<arch::CpuFeature::AVX512VLF, xmm128>
{};

template <>
class __simd_broadcast_implementation<arch::CpuFeature::AVX512VLDQ, xmm128> :
	public __simd_broadcast_implementation<arch::CpuFeature::AVX512VLF, xmm128>
{};

template <>
class __simd_broadcast_implementation<arch::CpuFeature::AVX512VLBWDQ, xmm128> :
	public __simd_broadcast_implementation<arch::CpuFeature::AVX512VLBW, xmm128>
{};

#pragma endregion

template <
	arch::CpuFeature	_SimdGeneration_,
	class				_RegisterPolicy_,
	class				_VectorType_,
	class				_DesiredType_>
simd_stl_nodiscard simd_stl_always_inline _VectorType_ __simd_broadcast(_DesiredType_ __value) noexcept {
	__verify_register_policy(_SimdGeneration_, _RegisterPolicy_);
	return __simd_broadcast_implementation<_SimdGeneration_, _RegisterPolicy_>::template __broadcast<_DesiredType_, _VectorType_>(__value);
}

template <
	arch::CpuFeature	_SimdGeneration_,
	class				_RegisterPolicy_,
	class				_VectorType_>
simd_stl_nodiscard simd_stl_always_inline _VectorType_ __simd_broadcast_zeros() noexcept {
	__verify_register_policy(_SimdGeneration_, _RegisterPolicy_);
	return __simd_broadcast_implementation<_SimdGeneration_, _RegisterPolicy_>::template __broadcast_zeros<_VectorType_>();
}

__SIMD_STL_DATAPAR_NAMESPACE_END

#include <src/simd_stl/datapar/SimdBroadcast.inl>
