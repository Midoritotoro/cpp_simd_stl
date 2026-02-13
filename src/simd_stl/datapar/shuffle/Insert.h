#pragma once 

#include <src/simd_stl/datapar/arithmetic/Sub.h>
#include <src/simd_stl/datapar/shuffle/BroadcastZeros.h>

#include <simd_stl/memory/PointerToIntegral.h>


__SIMD_STL_DATAPAR_NAMESPACE_BEGIN

template <
	arch::ISA	_ISA_,
	uint32		_Width_>
struct _Simd_insert;

template <>
struct _Simd_insert<arch::ISA::SSE2, 128> {
	template <
		class _IntrinType_,
		class _DesiredType_>
	simd_stl_nodiscard simd_stl_static_operator simd_stl_always_inline _IntrinType_ operator()() simd_stl_const_operator noexcept {

	}
};

template <>
struct _Simd_insert<arch::ISA::AVX2, 256> {
	template <
		class _IntrinType_,
		class _DesiredType_>
	simd_stl_nodiscard simd_stl_static_operator simd_stl_always_inline _IntrinType_ operator()() simd_stl_const_operator noexcept {
		
	}
};

template <>
struct _Simd_insert<arch::ISA::AVX512F, 512> {
	template <
		class _IntrinType_,
		class _DesiredType_>
	simd_stl_nodiscard simd_stl_static_operator simd_stl_always_inline _IntrinType_ operator()() simd_stl_const_operator noexcept {
		
	}
};

template <> struct _Simd_insert<arch::ISA::SSE3, 128>: _Simd_insert<arch::ISA::SSE2, 128> {};
template <> struct _Simd_insert<arch::ISA::SSSE3, 128>: _Simd_insert<arch::ISA::SSE3, 128> {};
template <> struct _Simd_insert<arch::ISA::SSE41, 128>: _Simd_insert<arch::ISA::SSSE3, 128> {};
template <> struct _Simd_insert<arch::ISA::SSE42, 128>: _Simd_insert<arch::ISA::SSE41, 128> {};
template <> struct _Simd_insert<arch::ISA::AVX2, 128>: _Simd_insert<arch::ISA::SSE42, 128> {};

template <> struct _Simd_insert<arch::ISA::AVX512BW, 512>: _Simd_insert<arch::ISA::AVX512F, 512> {};
template <> struct _Simd_insert<arch::ISA::AVX512DQ, 512>: _Simd_insert<arch::ISA::AVX512F, 512> {};
template <> struct _Simd_insert<arch::ISA::AVX512BWDQ, 512>: _Simd_insert<arch::ISA::AVX512BW, 512> {};
template <> struct _Simd_insert<arch::ISA::AVX512VBMI, 512>: _Simd_insert<arch::ISA::AVX512BW, 512> {};
template <> struct _Simd_insert<arch::ISA::AVX512VBMI2, 512>: _Simd_insert<arch::ISA::AVX512VBMI, 512> {};
template <> struct _Simd_insert<arch::ISA::AVX512VBMIDQ, 512>: _Simd_insert<arch::ISA::AVX512BWDQ, 512> {};
template <> struct _Simd_insert<arch::ISA::AVX512VBMI2DQ, 512>: _Simd_insert<arch::ISA::AVX512VBMIDQ, 512> {};

template <> struct _Simd_insert<arch::ISA::AVX512VLF, 256>: _Simd_insert<arch::ISA::AVX2, 256> {};
template <> struct _Simd_insert<arch::ISA::AVX512VLBW, 256>: _Simd_insert<arch::ISA::AVX512VLF, 256> {};
template <> struct _Simd_insert<arch::ISA::AVX512VLDQ, 256>: _Simd_insert<arch::ISA::AVX512VLF, 256> {};
template <> struct _Simd_insert<arch::ISA::AVX512VLBWDQ, 256>: _Simd_insert<arch::ISA::AVX512VLBW, 256> {};
template <> struct _Simd_insert<arch::ISA::AVX512VBMIVL, 256>: _Simd_insert<arch::ISA::AVX512VLBW, 256> {};
template <> struct _Simd_insert<arch::ISA::AVX512VBMI2VL, 256>: _Simd_insert<arch::ISA::AVX512VBMIVL, 256> {};
template <> struct _Simd_insert<arch::ISA::AVX512VBMIVLDQ, 256>: _Simd_insert<arch::ISA::AVX512VLBWDQ, 256> {};
template <> struct _Simd_insert<arch::ISA::AVX512VBMI2VLDQ, 256>: _Simd_insert<arch::ISA::AVX512VBMIVLDQ, 256> {};

template <> struct _Simd_insert<arch::ISA::AVX512VLF, 128>: _Simd_insert<arch::ISA::AVX2, 128> {};
template <> struct _Simd_insert<arch::ISA::AVX512VLBW, 128>: _Simd_insert<arch::ISA::AVX512VLF, 128> {};
template <> struct _Simd_insert<arch::ISA::AVX512VLDQ, 128>: _Simd_insert<arch::ISA::AVX512VLF, 128> {};
template <> struct _Simd_insert<arch::ISA::AVX512VLBWDQ, 128>: _Simd_insert<arch::ISA::AVX512VLBW, 128> {};
template <> struct _Simd_insert<arch::ISA::AVX512VBMIVL, 128>: _Simd_insert<arch::ISA::AVX512VLBW, 128> {};
template <> struct _Simd_insert<arch::ISA::AVX512VBMI2VL, 128>: _Simd_insert<arch::ISA::AVX512VBMIVL, 128> {};
template <> struct _Simd_insert<arch::ISA::AVX512VBMIVLDQ, 128>: _Simd_insert<arch::ISA::AVX512VLBWDQ, 128> {};
template <> struct _Simd_insert<arch::ISA::AVX512VBMI2VLDQ, 128>: _Simd_insert<arch::ISA::AVX512VBMIVLDQ, 128> {};

__SIMD_STL_DATAPAR_NAMESPACE_BEGIN
