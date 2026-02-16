#pragma once 

#include <simd_stl/arch/CpuFeature.h>


__SIMD_STL_DATAPAR_NAMESPACE_BEGIN

template <
    arch::ISA   _ISA_,
    uint32      _Width_,
    sizetype    _TypeSize_>
inline constexpr bool __is_native_mask_load_supported_v = false;

template <sizetype _TypeSize_>
inline constexpr bool __is_native_mask_load_supported_v<arch::ISA::AVX512VLF, 128, _TypeSize_> = (_TypeSize_ == 4) || (_TypeSize_ == 8);

template <sizetype _TypeSize_>
inline constexpr bool __is_native_mask_load_supported_v<arch::ISA::AVX512VLBW, 128, _TypeSize_> = true;

template <sizetype _TypeSize_>
inline constexpr bool __is_native_mask_load_supported_v<arch::ISA::AVX2, 256, _TypeSize_> = (_TypeSize_ == 4) || (_TypeSize_ == 8);

template <sizetype _TypeSize_>
inline constexpr bool __is_native_mask_load_supported_v<arch::ISA::AVX512VLF, 256, _TypeSize_> = (_TypeSize_ == 4) || (_TypeSize_ == 8);

template <sizetype _TypeSize_>
inline constexpr bool __is_native_mask_load_supported_v<arch::ISA::AVX512VLBW, 256, _TypeSize_> = true;

template <sizetype _TypeSize_>
inline constexpr bool __is_native_mask_load_supported_v<arch::ISA::AVX512F, 512, _TypeSize_> = (_TypeSize_ == 4) || (_TypeSize_ == 8);

template <sizetype _TypeSize_>
inline constexpr bool __is_native_mask_load_supported_v<arch::ISA::AVX512BW, 512, _TypeSize_> = true;

__SIMD_STL_DATAPAR_NAMESPACE_END
