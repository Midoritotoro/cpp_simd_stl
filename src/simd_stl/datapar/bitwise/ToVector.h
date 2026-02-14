#pragma once 

#include <src/simd_stl/datapar/arithmetic/Sub.h>
#include <src/simd_stl/datapar/shuffle/BroadcastZeros.h>

#include <simd_stl/memory/PointerToIntegral.h>


__SIMD_STL_DATAPAR_NAMESPACE_BEGIN

template <
	arch::ISA	_ISA_,
	uint32		_Width_,
    class       _IntrinType_,
	class		_DesiredType_>
struct _Simd_to_vector;

template <
    class _IntrinType_,
    class _DesiredType_>
struct _Simd_to_vector<arch::ISA::SSE2, 128, _IntrinType_, _DesiredType_> {
	template <class _MaskType_>
	simd_stl_nodiscard simd_stl_static_operator simd_stl_always_inline _IntrinType_ operator()(_MaskType_ __mask) simd_stl_const_operator noexcept {
	    if constexpr (sizeof(_DesiredType_) == 8) {
            const auto __broadcasted_mask = _mm_set1_epi8(static_cast<int8>(__mask));
            const auto __selected = _mm_and_si128(__broadcasted_mask, _mm_setr_epi32(1, 1, 2, 2));

            return __intrin_bitcast<_IntrinType_>(_mm_cmpgt_epi32(__selected, _mm_setzero_si128()));
        }
        else if constexpr (sizeof(_DesiredType_) == 4) {
            const auto __broadcasted_mask = _mm_set1_epi8(static_cast<int8>(__mask));
            const auto __selected = _mm_and_si128(__broadcasted_mask, _mm_setr_epi32(1, 2, 4, 8));

            return __intrin_bitcast<_IntrinType_>(_mm_cmpgt_epi32(__selected, _mm_setzero_si128()));
        }
        else if constexpr (sizeof(_DesiredType_) == 2) {
            const auto __broadcasted_mask = _mm_set1_epi8(static_cast<int8>(__mask));
            const auto __selected = _mm_and_si128(__broadcasted_mask, _mm_setr_epi32(0x00020001, 0x00080004, 0x00200010, 0x00800040));
        
            return __intrin_bitcast<_IntrinType_>(_mm_cmpgt_epi16(__selected, _mm_setzero_si128()));
        }
        else if constexpr (sizeof(_DesiredType_) == 1) {
            const auto __not_mask = uint16(~__mask);

            const auto __broadcasted_low_mask = _mm_set1_epi8(static_cast<int8>(__not_mask));
            const auto __broadcasted_high_mask = _mm_set1_epi8(static_cast<int8>(__not_mask >> 8));

            const auto __vector_mask_low = _mm_setr_epi32(0x08040201, 0x80402010, 0, 0);
            const auto __vector_mask_high = _mm_setr_epi32(0, 0, 0x08040201, 0x80402010);

            const auto __selected_low = _mm_and_si128(__broadcasted_low_mask, __vector_mask_low);
            const auto __selected_high = _mm_and_si128(__broadcasted_high_mask, __vector_mask_high);

            const auto __combined = _mm_or_si128(__selected_low, __selected_high);
            return __intrin_bitcast<_IntrinType_>(_mm_cmpeq_epi8(__combined, _mm_setzero_si128()));
        }
	}
};

template <
    class _IntrinType_,
    class _DesiredType_>
struct _Simd_to_vector<arch::ISA::AVX2, 256, _IntrinType_, _DesiredType_> {
	template <class _MaskType_>
	simd_stl_nodiscard simd_stl_static_operator simd_stl_always_inline auto operator()(_MaskType_ __mask) simd_stl_const_operator noexcept {

	}
};

template <
    class _IntrinType_,
    class _DesiredType_>
struct _Simd_to_vector<arch::ISA::AVX512F, 512, _IntrinType_, _DesiredType_> {
	template <class _MaskType_>
	simd_stl_nodiscard simd_stl_static_operator simd_stl_always_inline auto operator()(_MaskType_ __mask) simd_stl_const_operator noexcept {

	}
};

template <class _IntrinType_, class _DesiredType_> struct _Simd_to_vector<arch::ISA::SSE3, 128, _IntrinType_, _DesiredType_> : _Simd_to_vector<arch::ISA::SSE2, 128, _IntrinType_, _DesiredType_> {};
template <class _IntrinType_, class _DesiredType_> struct _Simd_to_vector<arch::ISA::SSSE3, 128, _IntrinType_, _DesiredType_> : _Simd_to_vector<arch::ISA::SSE3, 128, _IntrinType_, _DesiredType_> {};
template <class _IntrinType_, class _DesiredType_> struct _Simd_to_vector<arch::ISA::SSE41, 128, _IntrinType_, _DesiredType_> : _Simd_to_vector<arch::ISA::SSSE3, 128, _IntrinType_, _DesiredType_> {};
template <class _IntrinType_, class _DesiredType_> struct _Simd_to_vector<arch::ISA::SSE42, 128, _IntrinType_, _DesiredType_> : _Simd_to_vector<arch::ISA::SSE41, 128, _IntrinType_, _DesiredType_> {};
template <class _IntrinType_, class _DesiredType_> struct _Simd_to_vector<arch::ISA::AVX2, 128, _IntrinType_, _DesiredType_> : _Simd_to_vector<arch::ISA::SSE42, 128, _IntrinType_, _DesiredType_> {};

template <class _IntrinType_, class _DesiredType_> struct _Simd_to_vector<arch::ISA::AVX512BW, 512, _IntrinType_, _DesiredType_> : _Simd_to_vector<arch::ISA::AVX512F, 512, _IntrinType_, _DesiredType_> {};
template <class _IntrinType_, class _DesiredType_> struct _Simd_to_vector<arch::ISA::AVX512DQ, 512, _IntrinType_, _DesiredType_> : _Simd_to_vector<arch::ISA::AVX512F, 512, _IntrinType_, _DesiredType_> {};
template <class _IntrinType_, class _DesiredType_> struct _Simd_to_vector<arch::ISA::AVX512BWDQ, 512, _IntrinType_, _DesiredType_> : _Simd_to_vector<arch::ISA::AVX512BW, 512, _IntrinType_, _DesiredType_> {};
template <class _IntrinType_, class _DesiredType_> struct _Simd_to_vector<arch::ISA::AVX512VBMI, 512, _IntrinType_, _DesiredType_> : _Simd_to_vector<arch::ISA::AVX512BW, 512, _IntrinType_, _DesiredType_> {};
template <class _IntrinType_, class _DesiredType_> struct _Simd_to_vector<arch::ISA::AVX512VBMI2, 512, _IntrinType_, _DesiredType_> : _Simd_to_vector<arch::ISA::AVX512VBMI, 512, _IntrinType_, _DesiredType_> {};
template <class _IntrinType_, class _DesiredType_> struct _Simd_to_vector<arch::ISA::AVX512VBMIDQ, 512, _IntrinType_, _DesiredType_> : _Simd_to_vector<arch::ISA::AVX512BWDQ, 512, _IntrinType_, _DesiredType_> {};
template <class _IntrinType_, class _DesiredType_> struct _Simd_to_vector<arch::ISA::AVX512VBMI2DQ, 512, _IntrinType_, _DesiredType_> : _Simd_to_vector<arch::ISA::AVX512VBMIDQ, 512, _IntrinType_, _DesiredType_> {};

template <class _IntrinType_, class _DesiredType_> struct _Simd_to_vector<arch::ISA::AVX512VLF, 256, _IntrinType_, _DesiredType_> : _Simd_to_vector<arch::ISA::AVX2, 256, _IntrinType_, _DesiredType_> {};
template <class _IntrinType_, class _DesiredType_> struct _Simd_to_vector<arch::ISA::AVX512VLBW, 256, _IntrinType_, _DesiredType_> : _Simd_to_vector<arch::ISA::AVX512VLF, 256, _IntrinType_, _DesiredType_> {};
template <class _IntrinType_, class _DesiredType_> struct _Simd_to_vector<arch::ISA::AVX512VLDQ, 256, _IntrinType_, _DesiredType_> : _Simd_to_vector<arch::ISA::AVX512VLF, 256, _IntrinType_, _DesiredType_> {};
template <class _IntrinType_, class _DesiredType_> struct _Simd_to_vector<arch::ISA::AVX512VLBWDQ, 256, _IntrinType_, _DesiredType_> : _Simd_to_vector<arch::ISA::AVX512VLBW, 256, _IntrinType_, _DesiredType_> {};
template <class _IntrinType_, class _DesiredType_> struct _Simd_to_vector<arch::ISA::AVX512VBMIVL, 256, _IntrinType_, _DesiredType_> : _Simd_to_vector<arch::ISA::AVX512VLBW, 256, _IntrinType_, _DesiredType_> {};
template <class _IntrinType_, class _DesiredType_> struct _Simd_to_vector<arch::ISA::AVX512VBMI2VL, 256, _IntrinType_, _DesiredType_> : _Simd_to_vector<arch::ISA::AVX512VBMIVL, 256, _IntrinType_, _DesiredType_> {};
template <class _IntrinType_, class _DesiredType_> struct _Simd_to_vector<arch::ISA::AVX512VBMIVLDQ, 256, _IntrinType_, _DesiredType_> : _Simd_to_vector<arch::ISA::AVX512VLBWDQ, 256, _IntrinType_, _DesiredType_> {};
template <class _IntrinType_, class _DesiredType_> struct _Simd_to_vector<arch::ISA::AVX512VBMI2VLDQ, 256, _IntrinType_, _DesiredType_> : _Simd_to_vector<arch::ISA::AVX512VBMIVLDQ, 256, _IntrinType_, _DesiredType_> {};

template <class _IntrinType_, class _DesiredType_> struct _Simd_to_vector<arch::ISA::AVX512VLF, 128, _IntrinType_, _DesiredType_> : _Simd_to_vector<arch::ISA::AVX2, 128, _IntrinType_, _DesiredType_> {};
template <class _IntrinType_, class _DesiredType_> struct _Simd_to_vector<arch::ISA::AVX512VLBW, 128, _IntrinType_, _DesiredType_> : _Simd_to_vector<arch::ISA::AVX512VLF, 128, _IntrinType_, _DesiredType_> {};
template <class _IntrinType_, class _DesiredType_> struct _Simd_to_vector<arch::ISA::AVX512VLDQ, 128, _IntrinType_, _DesiredType_> : _Simd_to_vector<arch::ISA::AVX512VLF, 128, _IntrinType_, _DesiredType_> {};
template <class _IntrinType_, class _DesiredType_> struct _Simd_to_vector<arch::ISA::AVX512VLBWDQ, 128, _IntrinType_, _DesiredType_> : _Simd_to_vector<arch::ISA::AVX512VLBW, 128, _IntrinType_, _DesiredType_> {};
template <class _IntrinType_, class _DesiredType_> struct _Simd_to_vector<arch::ISA::AVX512VBMIVL, 128, _IntrinType_, _DesiredType_> : _Simd_to_vector<arch::ISA::AVX512VLBW, 128, _IntrinType_, _DesiredType_> {};
template <class _IntrinType_, class _DesiredType_> struct _Simd_to_vector<arch::ISA::AVX512VBMI2VL, 128, _IntrinType_, _DesiredType_> : _Simd_to_vector<arch::ISA::AVX512VBMIVL, 128, _IntrinType_, _DesiredType_> {};
template <class _IntrinType_, class _DesiredType_> struct _Simd_to_vector<arch::ISA::AVX512VBMIVLDQ, 128, _IntrinType_, _DesiredType_> : _Simd_to_vector<arch::ISA::AVX512VLBWDQ, 128, _IntrinType_, _DesiredType_> {};
template <class _IntrinType_, class _DesiredType_> struct _Simd_to_vector<arch::ISA::AVX512VBMI2VLDQ, 128, _IntrinType_, _DesiredType_> : _Simd_to_vector<arch::ISA::AVX512VBMIVLDQ, 128, _IntrinType_, _DesiredType_> {};

__SIMD_STL_DATAPAR_NAMESPACE_BEGIN
