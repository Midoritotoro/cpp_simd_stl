#pragma once 

#include <src/simd_stl/datapar/SimdMemoryAccess.h>
#include <simd_stl/memory/PointerToIntegral.h>


__SIMD_STL_DATAPAR_NAMESPACE_BEGIN

template <
    arch::CpuFeature    _SimdGeneration_, 
    class               _RegisterPolicy_>
class __simd_element_access;

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_,
    typename            _DesiredType_,
    typename            _VectorType_>
simd_stl_always_inline void __simd_insert(
    _VectorType_&       __vector,
    const uint8         __position,
    const _DesiredType_ __value) noexcept;

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_,
    typename            _DesiredType_,
    typename            _VectorType_>
simd_stl_always_inline _DesiredType_ __simd_extract(
    _VectorType_    __vector,
    const uint8     __where) noexcept;

#pragma region Sse2-Sse4.2 Simd element access

template <>
class __simd_element_access<arch::CpuFeature::SSE2, xmm128> {
    static constexpr auto __generation   = arch::CpuFeature::SSE2;
    using __register_policy = xmm128;
public:
    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline void __insert(
        _VectorType_&       __vector,
        const uint8         __position,
        const _DesiredType_ __value) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _DesiredType_ __extract(
        _VectorType_    __vector,
        const uint8     __where) noexcept;
};

template <>
class __simd_element_access<arch::CpuFeature::SSE3, xmm128>:
    public __simd_element_access<arch::CpuFeature::SSE2, xmm128>
{};

template <>
class __simd_element_access<arch::CpuFeature::SSSE3, xmm128> :
    public __simd_element_access<arch::CpuFeature::SSE3, xmm128>
{};

template <>
class __simd_element_access<arch::CpuFeature::SSE41, xmm128> :
    public __simd_element_access<arch::CpuFeature::SSSE3, xmm128>
{
    static constexpr auto __generation   = arch::CpuFeature::SSE41;
    using __register_policy = xmm128;
public:
    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline void __insert(
        _VectorType_&       __vector,
        const uint8         __position,
        const _DesiredType_ __value) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _DesiredType_ __extract(
        _VectorType_    __vector,
        const uint8     __where) noexcept;
};

template <>
class __simd_element_access<arch::CpuFeature::SSE42, xmm128>:
    public __simd_element_access<arch::CpuFeature::SSE41, xmm128>
{};

#pragma endregion

#pragma region Avx-Avx2 Simd element access 

template <>
class __simd_element_access<arch::CpuFeature::AVX2, xmm128>:
    public __simd_element_access<arch::CpuFeature::SSE42, xmm128>
{};

template <>
class __simd_element_access<arch::CpuFeature::AVX2, ymm256>
{
    static constexpr auto __generation      = arch::CpuFeature::AVX2;
    using __register_policy                 = datapar::ymm256;
public:
    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline void __insert(
        _VectorType_&       __vector,
        const uint8         __position,
        const _DesiredType_ __value) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _DesiredType_ __extract(
        _VectorType_    __vector,
        const uint8     __where) noexcept;
};

#pragma endregion

#pragma region Avx512 Simd element access

template <>
class __simd_element_access<arch::CpuFeature::AVX512F, zmm512>
{
    static constexpr auto __generation   = arch::CpuFeature::AVX512F;
    using __register_policy               = zmm512;
public:
    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline void __insert(
        _VectorType_&       __vector,
        const uint8         __position,
        const _DesiredType_ __value) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _DesiredType_ __extract(
        _VectorType_    __vector,
        const uint8     __where) noexcept;
};

template <>
class __simd_element_access<arch::CpuFeature::AVX512BW, zmm512>:
    public __simd_element_access<arch::CpuFeature::AVX512F, zmm512>
{};

template <>
class __simd_element_access<arch::CpuFeature::AVX512DQ, zmm512> :
    public __simd_element_access<arch::CpuFeature::AVX512F, zmm512>
{};

template <>
class __simd_element_access<arch::CpuFeature::AVX512BWDQ, zmm512> :
    public __simd_element_access<arch::CpuFeature::AVX512BW, zmm512>
{};

template <> 
class __simd_element_access<arch::CpuFeature::AVX512VBMI, zmm512>:
    public __simd_element_access<arch::CpuFeature::AVX512BW, zmm512>
{};

template <> 
class __simd_element_access<arch::CpuFeature::AVX512VBMI2, zmm512>:
    public __simd_element_access<arch::CpuFeature::AVX512BW, zmm512>
{};

template <> 
class __simd_element_access<arch::CpuFeature::AVX512VBMIDQ, zmm512>:
    public __simd_element_access<arch::CpuFeature::AVX512BWDQ, zmm512>
{};

template <> 
class __simd_element_access<arch::CpuFeature::AVX512VBMI2DQ, zmm512>:
    public __simd_element_access<arch::CpuFeature::AVX512BWDQ, zmm512>
{};

template <>
class __simd_element_access<arch::CpuFeature::AVX512VLF, ymm256> :
    public __simd_element_access<arch::CpuFeature::AVX2, ymm256>
{};

template <>
class __simd_element_access<arch::CpuFeature::AVX512VLBW, ymm256> :
    public __simd_element_access<arch::CpuFeature::AVX512VLF, ymm256>
{};

template <>
class __simd_element_access<arch::CpuFeature::AVX512VLDQ, ymm256> :
    public __simd_element_access<arch::CpuFeature::AVX512VLF, ymm256>
{};

template <>
class __simd_element_access<arch::CpuFeature::AVX512VLBWDQ, ymm256> :
    public __simd_element_access<arch::CpuFeature::AVX512VLBW, ymm256>
{};

template <>
class __simd_element_access<arch::CpuFeature::AVX512VLF, xmm128> :
    public __simd_element_access<arch::CpuFeature::SSE42, xmm128>
{};

template <>
class __simd_element_access<arch::CpuFeature::AVX512VLBW, xmm128> :
    public __simd_element_access<arch::CpuFeature::AVX512VLF, xmm128>
{};

template <>
class __simd_element_access<arch::CpuFeature::AVX512VLDQ, xmm128> :
    public __simd_element_access<arch::CpuFeature::AVX512VLF, xmm128>
{};

template <>
class __simd_element_access<arch::CpuFeature::AVX512VLBWDQ, xmm128> :
    public __simd_element_access<arch::CpuFeature::AVX512VLBW, xmm128>
{};

template <>
class __simd_element_access<arch::CpuFeature::AVX512VBMIVL, xmm128> :
	public __simd_element_access<arch::CpuFeature::AVX512VLBW, xmm128>
{};

template <>
class __simd_element_access<arch::CpuFeature::AVX512VBMI2VL, xmm128> :
	public __simd_element_access<arch::CpuFeature::AVX512VLBW, xmm128>
{};

template <>
class __simd_element_access<arch::CpuFeature::AVX512VBMIVLDQ, xmm128> :
	public __simd_element_access<arch::CpuFeature::AVX512VLBWDQ, xmm128>
{};

template <>
class __simd_element_access<arch::CpuFeature::AVX512VBMI2VLDQ, xmm128> :
	public __simd_element_access<arch::CpuFeature::AVX512VLBWDQ, xmm128>
{};

template <>
class __simd_element_access<arch::CpuFeature::AVX512VBMIVL, ymm256> :
	public __simd_element_access<arch::CpuFeature::AVX512VLBW, ymm256>
{};

template <>
class __simd_element_access<arch::CpuFeature::AVX512VBMI2VL, ymm256> :
	public __simd_element_access<arch::CpuFeature::AVX512VLBW, ymm256>
{};

template <>
class __simd_element_access<arch::CpuFeature::AVX512VBMIVLDQ, ymm256> :
	public __simd_element_access<arch::CpuFeature::AVX512VLBWDQ, ymm256>
{};

template <>
class __simd_element_access<arch::CpuFeature::AVX512VBMI2VLDQ, ymm256> :
	public __simd_element_access<arch::CpuFeature::AVX512VLBWDQ, ymm256>
{};

#pragma endregion 


template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_,
    typename            _DesiredType_,
    typename            _VectorType_>
simd_stl_always_inline void __simd_insert(
    _VectorType_&       __vector,
    const uint8         __position,
    const _DesiredType_ __value) noexcept
{
    __verify_register_policy(_SimdGeneration_, _RegisterPolicy_);
    __simd_element_access<_SimdGeneration_, _RegisterPolicy_>::template __insert<_DesiredType_>(__vector, __position, __value);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_,
    typename            _DesiredType_,
    typename            _VectorType_>
simd_stl_always_inline _DesiredType_ __simd_extract(
    _VectorType_    __vector,
    const uint8     __where) noexcept
{
    __verify_register_policy(_SimdGeneration_, _RegisterPolicy_);
    return __simd_element_access<_SimdGeneration_, _RegisterPolicy_>::template __extract<_DesiredType_>(__vector, __where);
}

__SIMD_STL_DATAPAR_NAMESPACE_END

#include <src/simd_stl/datapar/SimdElementAccess.inl>
