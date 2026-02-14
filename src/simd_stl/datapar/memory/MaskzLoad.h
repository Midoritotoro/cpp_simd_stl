#pragma once 

#include <src/simd_stl/datapar/IntrinBitcast.h>


__SIMD_STL_DATAPAR_NAMESPACE_BEGIN

template <
	arch::ISA	_ISA_,
	uint32		_Width_,
	class		_DesiredType_,
	class		_IntrinType_>
struct _Simd_maskz_load;

template <	
	class _DesiredType_,
	class _IntrinType_>
struct _Simd_maskz_load<arch::ISA::SSE2, 128, _DesiredType_, _IntrinType_> {
	template <
		class _MaskType_,
		class _AlignmentPolicy_>
	simd_stl_nodiscard simd_stl_static_operator simd_stl_always_inline _IntrinType_ operator()(
		const void*			__address,
		_MaskType_			__mask,
		_AlignmentPolicy_&&	__alignment_policy) simd_stl_const_operator noexcept 
	{
		
	}
};

template <class _IntrinType_> 
struct _Simd_maskz_load<arch::ISA::SSE3, 128, _IntrinType_>:
	_Simd_maskz_load<arch::ISA::SSE2, 128, _IntrinType_>
{
	
};


template <class _IntrinType_>
struct _Simd_maskz_load<arch::ISA::AVX2, 256, _IntrinType_> {

};

template <class _IntrinType_>
struct _Simd_maskz_load<arch::ISA::AVX512F, 512, _IntrinType_> {

};

template <class _IntrinType_> struct _Simd_maskz_load<arch::ISA::SSSE3, 128, _IntrinType_> : _Simd_maskz_load<arch::ISA::SSE3, 128, _IntrinType_> {};
template <class _IntrinType_> struct _Simd_maskz_load<arch::ISA::SSE41, 128, _IntrinType_> : _Simd_maskz_load<arch::ISA::SSSE3, 128, _IntrinType_> {};
template <class _IntrinType_> struct _Simd_maskz_load<arch::ISA::SSE42, 128, _IntrinType_> : _Simd_maskz_load<arch::ISA::SSE41, 128, _IntrinType_> {};
template <class _IntrinType_> struct _Simd_maskz_load<arch::ISA::AVX2, 128, _IntrinType_> : _Simd_maskz_load<arch::ISA::SSE42, 128, _IntrinType_> {};

template <class _IntrinType_> struct _Simd_maskz_load<arch::ISA::AVX512BW, 512, _IntrinType_> : _Simd_maskz_load<arch::ISA::AVX512F, 512, _IntrinType_> {};
template <class _IntrinType_> struct _Simd_maskz_load<arch::ISA::AVX512DQ, 512, _IntrinType_> : _Simd_maskz_load<arch::ISA::AVX512F, 512, _IntrinType_> {};
template <class _IntrinType_> struct _Simd_maskz_load<arch::ISA::AVX512BWDQ, 512, _IntrinType_> : _Simd_maskz_load<arch::ISA::AVX512BW, 512, _IntrinType_> {};
template <class _IntrinType_> struct _Simd_maskz_load<arch::ISA::AVX512VBMI, 512, _IntrinType_> : _Simd_maskz_load<arch::ISA::AVX512BW, 512, _IntrinType_> {};
template <class _IntrinType_> struct _Simd_maskz_load<arch::ISA::AVX512VBMI2, 512, _IntrinType_> : _Simd_maskz_load<arch::ISA::AVX512VBMI, 512, _IntrinType_> {};
template <class _IntrinType_> struct _Simd_maskz_load<arch::ISA::AVX512VBMIDQ, 512, _IntrinType_> : _Simd_maskz_load<arch::ISA::AVX512BWDQ, 512, _IntrinType_> {};
template <class _IntrinType_> struct _Simd_maskz_load<arch::ISA::AVX512VBMI2DQ, 512, _IntrinType_> : _Simd_maskz_load<arch::ISA::AVX512VBMIDQ, 512, _IntrinType_> {};

template <class _IntrinType_> struct _Simd_maskz_load<arch::ISA::AVX512VLF, 256, _IntrinType_> : _Simd_maskz_load<arch::ISA::AVX2, 256, _IntrinType_> {};
template <class _IntrinType_> struct _Simd_maskz_load<arch::ISA::AVX512VLBW, 256, _IntrinType_> : _Simd_maskz_load<arch::ISA::AVX512VLF, 256, _IntrinType_> {};
template <class _IntrinType_> struct _Simd_maskz_load<arch::ISA::AVX512VLDQ, 256, _IntrinType_> : _Simd_maskz_load<arch::ISA::AVX512VLF, 256, _IntrinType_> {};
template <class _IntrinType_> struct _Simd_maskz_load<arch::ISA::AVX512VLBWDQ, 256, _IntrinType_> : _Simd_maskz_load<arch::ISA::AVX512VLBW, 256, _IntrinType_> {};
template <class _IntrinType_> struct _Simd_maskz_load<arch::ISA::AVX512VBMIVL, 256, _IntrinType_> : _Simd_maskz_load<arch::ISA::AVX512VLBW, 256, _IntrinType_> {};
template <class _IntrinType_> struct _Simd_maskz_load<arch::ISA::AVX512VBMI2VL, 256, _IntrinType_> : _Simd_maskz_load<arch::ISA::AVX512VBMIVL, 256, _IntrinType_> {};
template <class _IntrinType_> struct _Simd_maskz_load<arch::ISA::AVX512VBMIVLDQ, 256, _IntrinType_> : _Simd_maskz_load<arch::ISA::AVX512VLBWDQ, 256, _IntrinType_> {};
template <class _IntrinType_> struct _Simd_maskz_load<arch::ISA::AVX512VBMI2VLDQ, 256, _IntrinType_> : _Simd_maskz_load<arch::ISA::AVX512VBMIVLDQ, 256, _IntrinType_> {};

template <class _IntrinType_> struct _Simd_maskz_load<arch::ISA::AVX512VLF, 128, _IntrinType_> : _Simd_maskz_load<arch::ISA::AVX2, 128, _IntrinType_> {};
template <class _IntrinType_> struct _Simd_maskz_load<arch::ISA::AVX512VLBW, 128, _IntrinType_> : _Simd_maskz_load<arch::ISA::AVX512VLF, 128, _IntrinType_> {};
template <class _IntrinType_> struct _Simd_maskz_load<arch::ISA::AVX512VLDQ, 128, _IntrinType_> : _Simd_maskz_load<arch::ISA::AVX512VLF, 128, _IntrinType_> {};
template <class _IntrinType_> struct _Simd_maskz_load<arch::ISA::AVX512VLBWDQ, 128, _IntrinType_> : _Simd_maskz_load<arch::ISA::AVX512VLBW, 128, _IntrinType_> {};
template <class _IntrinType_> struct _Simd_maskz_load<arch::ISA::AVX512VBMIVL, 128, _IntrinType_> : _Simd_maskz_load<arch::ISA::AVX512VLBW, 128, _IntrinType_> {};
template <class _IntrinType_> struct _Simd_maskz_load<arch::ISA::AVX512VBMI2VL, 128, _IntrinType_> : _Simd_maskz_load<arch::ISA::AVX512VBMIVL, 128, _IntrinType_> {};
template <class _IntrinType_> struct _Simd_maskz_load<arch::ISA::AVX512VBMIVLDQ, 128, _IntrinType_> : _Simd_maskz_load<arch::ISA::AVX512VLBWDQ, 128, _IntrinType_> {};
template <class _IntrinType_> struct _Simd_maskz_load<arch::ISA::AVX512VBMI2VLDQ, 128, _IntrinType_> : _Simd_maskz_load<arch::ISA::AVX512VBMIVLDQ, 128, _IntrinType_> {};

__SIMD_STL_DATAPAR_NAMESPACE_BEGIN
