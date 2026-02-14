#pragma once 

#include <src/simd_stl/datapar/arithmetic/Sub.h>
#include <src/simd_stl/datapar/shuffle/BroadcastZeros.h>

#include <simd_stl/memory/PointerToIntegral.h>


__SIMD_STL_DATAPAR_NAMESPACE_BEGIN

template <
	arch::ISA	_ISA_,
	uint32		_Width_,
	class		_DesiredType_>
struct _Simd_to_mask;

template <class _DesiredType_>
struct _Simd_to_mask<arch::ISA::SSE2, 128, _DesiredType_> {
	template <class _IntrinType_>
	simd_stl_nodiscard simd_stl_static_operator simd_stl_always_inline auto operator()(_IntrinType_ __vector) simd_stl_const_operator noexcept {

	}
};

template <class _DesiredType_>
struct _Simd_to_mask<arch::ISA::AVX2, 256, _DesiredType_> {
	template <class _IntrinType_>
	simd_stl_nodiscard simd_stl_static_operator simd_stl_always_inline auto operator()(_IntrinType_ __vector) simd_stl_const_operator noexcept {

	}
};

template <class _DesiredType_>
struct _Simd_to_mask<arch::ISA::AVX512F, 512, _DesiredType_> {
	template <class _IntrinType_>
	simd_stl_nodiscard simd_stl_static_operator simd_stl_always_inline auto operator()(_IntrinType_ __vector) simd_stl_const_operator noexcept {

	}
};

template <class _DesiredType_> struct _Simd_to_mask<arch::ISA::SSE3, 128, _DesiredType_> : _Simd_to_mask<arch::ISA::SSE2, 128, _DesiredType_> {};
template <class _DesiredType_> struct _Simd_to_mask<arch::ISA::SSSE3, 128, _DesiredType_> : _Simd_to_mask<arch::ISA::SSE3, 128, _DesiredType_> {};
template <class _DesiredType_> struct _Simd_to_mask<arch::ISA::SSE41, 128, _DesiredType_> : _Simd_to_mask<arch::ISA::SSSE3, 128, _DesiredType_> {};
template <class _DesiredType_> struct _Simd_to_mask<arch::ISA::SSE42, 128, _DesiredType_> : _Simd_to_mask<arch::ISA::SSE41, 128, _DesiredType_> {};
template <class _DesiredType_> struct _Simd_to_mask<arch::ISA::AVX2, 128, _DesiredType_> : _Simd_to_mask<arch::ISA::SSE42, 128, _DesiredType_> {};

template <class _DesiredType_> struct _Simd_to_mask<arch::ISA::AVX512BW, 512, _DesiredType_> : _Simd_to_mask<arch::ISA::AVX512F, 512, _DesiredType_> {};
template <class _DesiredType_> struct _Simd_to_mask<arch::ISA::AVX512DQ, 512, _DesiredType_> : _Simd_to_mask<arch::ISA::AVX512F, 512, _DesiredType_> {};
template <class _DesiredType_> struct _Simd_to_mask<arch::ISA::AVX512BWDQ, 512, _DesiredType_> : _Simd_to_mask<arch::ISA::AVX512BW, 512, _DesiredType_> {};
template <class _DesiredType_> struct _Simd_to_mask<arch::ISA::AVX512VBMI, 512, _DesiredType_> : _Simd_to_mask<arch::ISA::AVX512BW, 512, _DesiredType_> {};
template <class _DesiredType_> struct _Simd_to_mask<arch::ISA::AVX512VBMI2, 512, _DesiredType_> : _Simd_to_mask<arch::ISA::AVX512VBMI, 512, _DesiredType_> {};
template <class _DesiredType_> struct _Simd_to_mask<arch::ISA::AVX512VBMIDQ, 512, _DesiredType_> : _Simd_to_mask<arch::ISA::AVX512BWDQ, 512, _DesiredType_> {};
template <class _DesiredType_> struct _Simd_to_mask<arch::ISA::AVX512VBMI2DQ, 512, _DesiredType_> : _Simd_to_mask<arch::ISA::AVX512VBMIDQ, 512, _DesiredType_> {};

template <class _DesiredType_> struct _Simd_to_mask<arch::ISA::AVX512VLF, 256, _DesiredType_> : _Simd_to_mask<arch::ISA::AVX2, 256, _DesiredType_> {};
template <class _DesiredType_> struct _Simd_to_mask<arch::ISA::AVX512VLBW, 256, _DesiredType_> : _Simd_to_mask<arch::ISA::AVX512VLF, 256, _DesiredType_> {};
template <class _DesiredType_> struct _Simd_to_mask<arch::ISA::AVX512VLDQ, 256, _DesiredType_> : _Simd_to_mask<arch::ISA::AVX512VLF, 256, _DesiredType_> {};
template <class _DesiredType_> struct _Simd_to_mask<arch::ISA::AVX512VLBWDQ, 256, _DesiredType_> : _Simd_to_mask<arch::ISA::AVX512VLBW, 256, _DesiredType_> {};
template <class _DesiredType_> struct _Simd_to_mask<arch::ISA::AVX512VBMIVL, 256, _DesiredType_> : _Simd_to_mask<arch::ISA::AVX512VLBW, 256, _DesiredType_> {};
template <class _DesiredType_> struct _Simd_to_mask<arch::ISA::AVX512VBMI2VL, 256, _DesiredType_> : _Simd_to_mask<arch::ISA::AVX512VBMIVL, 256, _DesiredType_> {};
template <class _DesiredType_> struct _Simd_to_mask<arch::ISA::AVX512VBMIVLDQ, 256, _DesiredType_> : _Simd_to_mask<arch::ISA::AVX512VLBWDQ, 256, _DesiredType_> {};
template <class _DesiredType_> struct _Simd_to_mask<arch::ISA::AVX512VBMI2VLDQ, 256, _DesiredType_> : _Simd_to_mask<arch::ISA::AVX512VBMIVLDQ, 256, _DesiredType_> {};

template <class _DesiredType_> struct _Simd_to_mask<arch::ISA::AVX512VLF, 128, _DesiredType_> : _Simd_to_mask<arch::ISA::AVX2, 128, _DesiredType_> {};
template <class _DesiredType_> struct _Simd_to_mask<arch::ISA::AVX512VLBW, 128, _DesiredType_> : _Simd_to_mask<arch::ISA::AVX512VLF, 128, _DesiredType_> {};
template <class _DesiredType_> struct _Simd_to_mask<arch::ISA::AVX512VLDQ, 128, _DesiredType_> : _Simd_to_mask<arch::ISA::AVX512VLF, 128, _DesiredType_> {};
template <class _DesiredType_> struct _Simd_to_mask<arch::ISA::AVX512VLBWDQ, 128, _DesiredType_> : _Simd_to_mask<arch::ISA::AVX512VLBW, 128, _DesiredType_> {};
template <class _DesiredType_> struct _Simd_to_mask<arch::ISA::AVX512VBMIVL, 128, _DesiredType_> : _Simd_to_mask<arch::ISA::AVX512VLBW, 128, _DesiredType_> {};
template <class _DesiredType_> struct _Simd_to_mask<arch::ISA::AVX512VBMI2VL, 128, _DesiredType_> : _Simd_to_mask<arch::ISA::AVX512VBMIVL, 128, _DesiredType_> {};
template <class _DesiredType_> struct _Simd_to_mask<arch::ISA::AVX512VBMIVLDQ, 128, _DesiredType_> : _Simd_to_mask<arch::ISA::AVX512VLBWDQ, 128, _DesiredType_> {};
template <class _DesiredType_> struct _Simd_to_mask<arch::ISA::AVX512VBMI2VLDQ, 128, _DesiredType_> : _Simd_to_mask<arch::ISA::AVX512VBMIVLDQ, 128, _DesiredType_> {};

__SIMD_STL_DATAPAR_NAMESPACE_BEGIN
