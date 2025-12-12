#pragma once 

#include <src/simd_stl/numeric/SimdMemoryAccess.h>
#include <simd_stl/memory/pointerToIntegral.h>


__SIMD_STL_NUMERIC_NAMESPACE_BEGIN

template <
    arch::CpuFeature    _SimdGeneration_, 
    class               _RegisterPolicy_>
class _SimdElementAccess;

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_,
    typename            _DesiredType_,
    typename            _VectorType_>
simd_stl_always_inline void _SimdInsert(
    _VectorType_&       _Vector,
    const uint8         _Position,
    const _DesiredType_ _Value) noexcept;

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_,
    typename            _DesiredType_,
    typename            _VectorType_>
simd_stl_always_inline _DesiredType_ _SimdExtract(
    _VectorType_    _Vector,
    const uint8     _Where) noexcept;

#pragma region Sse2-Sse4.2 Simd element access

template <>
class _SimdElementAccess<arch::CpuFeature::SSE2, xmm128> {
    static constexpr auto _Generation   = arch::CpuFeature::SSE2;
    using _RegisterPolicy               = xmm128;
public:
    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline void _Insert(
        _VectorType_&       _Vector,
        const uint8         _Position,
        const _DesiredType_ _Value) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _DesiredType_ _Extract(
        _VectorType_    _Vector,
        const uint8     _Where) noexcept;
};

template <>
class _SimdElementAccess<arch::CpuFeature::SSE3, xmm128>:
    public _SimdElementAccess<arch::CpuFeature::SSE2, xmm128>
{};

template <>
class _SimdElementAccess<arch::CpuFeature::SSSE3, xmm128> :
    public _SimdElementAccess<arch::CpuFeature::SSE3, xmm128>
{};

template <>
class _SimdElementAccess<arch::CpuFeature::SSE41, xmm128> :
    public _SimdElementAccess<arch::CpuFeature::SSSE3, xmm128>
{
    static constexpr auto _Generation   = arch::CpuFeature::SSE2;
    using _RegisterPolicy               = xmm128;
public:
    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline void _Insert(
        _VectorType_&       _Vector,
        const uint8         _Position,
        const _DesiredType_ _Value) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _DesiredType_ _Extract(
        _VectorType_    _Vector,
        const uint8     _Where) noexcept;
};

template <>
class _SimdElementAccess<arch::CpuFeature::SSE42, xmm128>:
    public _SimdElementAccess<arch::CpuFeature::SSE41, xmm128>
{};

#pragma endregion

#pragma region Avx-Avx2 Simd element access 

template <>
class _SimdElementAccess<arch::CpuFeature::AVX, ymm256>
{
    static constexpr auto _Generation   = arch::CpuFeature::AVX;
    using _RegisterPolicy               = numeric::ymm256;
public:
    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline void _Insert(
        _VectorType_&       _Vector,
        const uint8         _Position,
        const _DesiredType_ _Value) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _DesiredType_ _Extract(
        _VectorType_    _Vector,
        const uint8     _Where) noexcept;
};

template <>
class _SimdElementAccess<arch::CpuFeature::AVX2, ymm256>:
    public _SimdElementAccess<arch::CpuFeature::AVX, ymm256>
{};

#pragma endregion

#pragma region Avx512 Simd element access

template <>
class _SimdElementAccess<arch::CpuFeature::AVX512F, zmm512>
{
    static constexpr auto _Generation   = arch::CpuFeature::AVX512F;
    using _RegisterPolicy               = zmm512;
public:
    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline void _Insert(
        _VectorType_&       _Vector,
        const uint8         _Position,
        const _DesiredType_ _Value) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _DesiredType_ _Extract(
        _VectorType_    _Vector,
        const uint8     _Where) noexcept;
};

template <>
class _SimdElementAccess<arch::CpuFeature::AVX512BW, zmm512>:
    public _SimdElementAccess<arch::CpuFeature::AVX512F, zmm512>
{};

template <>
class _SimdElementAccess<arch::CpuFeature::AVX512DQ, zmm512> :
    public _SimdElementAccess<arch::CpuFeature::AVX512BW, zmm512>
{};

template <>
class _SimdElementAccess<arch::CpuFeature::AVX512VL, zmm512> :
    public _SimdElementAccess<arch::CpuFeature::AVX512DQ, zmm512>
{};

#pragma endregion 


template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_,
    typename            _DesiredType_,
    typename            _VectorType_>
simd_stl_always_inline void _SimdInsert(
    _VectorType_&       _Vector,
    const uint8         _Position,
    const _DesiredType_ _Value) noexcept
{
    _VerifyRegisterPolicy(_SimdGeneration_, _RegisterPolicy_);
    _SimdElementAccess<_SimdGeneration_, _RegisterPolicy_>::template _Insert(_Vector, _Position, _Value);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_,
    typename            _DesiredType_,
    typename            _VectorType_>
simd_stl_always_inline _DesiredType_ _SimdExtract(
    _VectorType_    _Vector,
    const uint8     _Where) noexcept
{
    _VerifyRegisterPolicy(_SimdGeneration_, _RegisterPolicy_);
    return _SimdElementAccess<_SimdGeneration_, _RegisterPolicy_>::template _Extract<_DesiredType_>(_Vector, _Where);
}

__SIMD_STL_NUMERIC_NAMESPACE_END

#include <src/simd_stl/numeric/SimdElementAccess.inl>
