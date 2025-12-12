#pragma once 

#include <src/simd_stl/numeric/SimdIntegralTypesCheck.h>
#include <src/simd_stl/numeric/SimdConvert.h>

#include <simd_stl/numeric/BasicSimdShuffleMask.h>
#include <src/simd_stl/numeric/ShuffleTables.h>

__SIMD_STL_NUMERIC_NAMESPACE_BEGIN

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_>
class _SimdElementWise;

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_,
    typename            _DesiredType_,
    typename            _VectorType_>
simd_stl_always_inline _VectorType_ _SimdReverse(_VectorType_ _Vector) noexcept;

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_,
    typename            _DesiredType_,
    typename            _VectorType_>
simd_stl_always_inline _VectorType_ _SimdBlend(
    _VectorType_                            _First,
    _VectorType_                            _Second,
    type_traits::__deduce_simd_mask_type<_SimdGeneration_,
        _DesiredType_, _RegisterPolicy_>    _Mask) noexcept;

#pragma region Sse2-Sse4.2 Simd element wise 

template <>
class _SimdElementWise<arch::CpuFeature::SSE2, xmm128> {
    static constexpr auto _Generation   = arch::CpuFeature::SSE2;
    using _RegisterPolicy               = xmm128;

    template <typename _DesiredType_>
    using _Simd_mask_type = type_traits::__deduce_simd_mask_type<_Generation, _DesiredType_, _RegisterPolicy>;
public:
    template <
        typename    _DesiredType_,
        typename    _VectorType_>
    static simd_stl_always_inline _VectorType_ _Blend(
        _VectorType_ _First,
        _VectorType_ _Second,
        _VectorType_ _Mask) noexcept;

    template <
        typename    _DesiredType_,
        typename    _VectorType_>
    static simd_stl_always_inline _VectorType_ _Blend(
        _VectorType_                        _First,
        _VectorType_                        _Second,
        _Simd_mask_type<_DesiredType_>      _Mask) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _Reverse(_VectorType_ _Vector) noexcept;
};

template <>
class _SimdElementWise<arch::CpuFeature::SSE3, xmm128> :
    public _SimdElementWise<arch::CpuFeature::SSE2, xmm128>
{
};

template <>
class _SimdElementWise<arch::CpuFeature::SSSE3, xmm128> :
    public _SimdElementWise<arch::CpuFeature::SSE3, xmm128>
{
public:
    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _Reverse(_VectorType_ _Vector) noexcept;
};

template <>
class _SimdElementWise<arch::CpuFeature::SSE41, xmm128> :
    public _SimdElementWise<arch::CpuFeature::SSSE3, xmm128>
{
    static constexpr auto _Generation   = arch::CpuFeature::SSE41;
    using _RegisterPolicy               = xmm128;

    template <typename _DesiredType_>
    using _Simd_mask_type = type_traits::__deduce_simd_mask_type<_Generation, _DesiredType_, _RegisterPolicy>;
public:
    template <
        typename    _DesiredType_,
        typename    _VectorType_>
    static simd_stl_always_inline _VectorType_ _Blend(
        _VectorType_ _First,
        _VectorType_ _Second,
        _VectorType_ _Mask) noexcept;

    template <
        typename    _DesiredType_,
        typename    _VectorType_>
    static simd_stl_always_inline _VectorType_ _Blend(
        _VectorType_                        _First,
        _VectorType_                        _Second,
        _Simd_mask_type<_DesiredType_>      _Mask) noexcept;
};

template <>
class _SimdElementWise<arch::CpuFeature::SSE42, xmm128> :
    public _SimdElementWise<arch::CpuFeature::SSE41, xmm128>
{};

#pragma endregion

#pragma region Avx-Avx2 Simd element wise

template <>
class _SimdElementWise<arch::CpuFeature::AVX, ymm256>
{
    static constexpr auto _Generation   = arch::CpuFeature::AVX;
    using _RegisterPolicy               = ymm256;

    template <typename _DesiredType_>
    using _Simd_mask_type = type_traits::__deduce_simd_mask_type<_Generation, _DesiredType_, _RegisterPolicy>;
public:
    template <
        typename    _DesiredType_,
        typename    _VectorType_>
    static simd_stl_always_inline _VectorType_ _Blend(
        _VectorType_ _First,
        _VectorType_ _Second,
        _VectorType_ _Mask) noexcept;

    template <
        typename    _DesiredType_,
        typename    _VectorType_>
    static simd_stl_always_inline _VectorType_ _Blend(
        _VectorType_                        _First,
        _VectorType_                        _Second,
        _Simd_mask_type<_DesiredType_>      _Mask) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _Reverse(_VectorType_ _Vector) noexcept;
};

template <>
class _SimdElementWise<arch::CpuFeature::AVX2, ymm256>:
    public _SimdElementWise<arch::CpuFeature::AVX, ymm256>
{
    static constexpr auto _Generation   = arch::CpuFeature::AVX2;
    using _RegisterPolicy               = ymm256;

    template <typename _DesiredType_>
    using _Simd_mask_type = type_traits::__deduce_simd_mask_type<_Generation, _DesiredType_, _RegisterPolicy>;
public:
    template <
        typename    _DesiredType_,
        typename    _VectorType_>
    static simd_stl_always_inline _VectorType_ _Blend(
        _VectorType_ _First,
        _VectorType_ _Second,
        _VectorType_ _Mask) noexcept;
    template <
        typename    _DesiredType_,
        typename    _VectorType_>
    static simd_stl_always_inline _VectorType_ _Blend(
        _VectorType_                        _First,
        _VectorType_                        _Second,
        _Simd_mask_type<_DesiredType_>      _Mask) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _Reverse(_VectorType_ _Vector) noexcept;
};

#pragma endregion

#pragma region Avx512 Simd element wise

template <>
class _SimdElementWise<arch::CpuFeature::AVX512F, zmm512>
{
    static constexpr auto _Generation   = arch::CpuFeature::AVX512F;
    using _RegisterPolicy               = zmm512;

    template <typename _DesiredType_>
    using _Simd_mask_type = type_traits::__deduce_simd_mask_type<_Generation, _DesiredType_, _RegisterPolicy>;
public:
    template <
        typename    _DesiredType_,
        typename    _VectorType_>
    static simd_stl_always_inline _VectorType_ _Blend(
        _VectorType_ _First,
        _VectorType_ _Second,
        _VectorType_ _Mask) noexcept;

    template <
        typename    _DesiredType_,
        typename    _VectorType_>
    static simd_stl_always_inline _VectorType_ _Blend(
        _VectorType_                        _First,
        _VectorType_                        _Second,
        _Simd_mask_type<_DesiredType_>      _Mask) noexcept;
    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _Reverse(_VectorType_ _Vector) noexcept;
};

template <>
class _SimdElementWise<arch::CpuFeature::AVX512BW, zmm512>:
    public _SimdElementWise<arch::CpuFeature::AVX512F, zmm512>
{
    static constexpr auto _Generation   = arch::CpuFeature::AVX512BW;
    using _RegisterPolicy               = zmm512;

    template <typename _DesiredType_>
    using _Simd_mask_type = type_traits::__deduce_simd_mask_type<_Generation, _DesiredType_, _RegisterPolicy>;

    template <typename _Type_>
    static simd_stl_always_inline auto _ExpandMaskBits(_Type_ _Mask) noexcept;
public:
    template <
        typename    _DesiredType_,
        typename    _VectorType_>
    static simd_stl_always_inline _VectorType_ _Blend(
        _VectorType_ _First,
        _VectorType_ _Second,
        _VectorType_ _Mask) noexcept;

    template <
        typename    _DesiredType_,
        typename    _VectorType_>
    static simd_stl_always_inline _VectorType_ _Blend(
        _VectorType_                        _First,
        _VectorType_                        _Second,
        _Simd_mask_type<_DesiredType_>      _Mask) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _Reverse(_VectorType_ _Vector) noexcept;
};

template <>
class _SimdElementWise<arch::CpuFeature::AVX512DQ, zmm512> :
    public _SimdElementWise<arch::CpuFeature::AVX512BW, zmm512>
{};

template <>
class _SimdElementWise<arch::CpuFeature::AVX512VL, zmm512> :
    public _SimdElementWise<arch::CpuFeature::AVX512DQ, zmm512>
{};

#pragma endregion

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_,
    typename            _DesiredType_,
    typename            _VectorType_>
simd_stl_always_inline _VectorType_ _SimdReverse(_VectorType_ _Vector) noexcept {
    _VerifyRegisterPolicy(_SimdGeneration_, _RegisterPolicy_);
    return _SimdElementWise<_SimdGeneration_, _RegisterPolicy_>::template _Reverse<_DesiredType_>(_Vector);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_,
    typename            _DesiredType_,
    typename            _VectorType_>
simd_stl_always_inline _VectorType_ _SimdBlend(
    _VectorType_                            _First,
    _VectorType_                            _Second,
    type_traits::__deduce_simd_mask_type<_SimdGeneration_,
        _DesiredType_, _RegisterPolicy_>    _Mask) noexcept
{
    _VerifyRegisterPolicy(_SimdGeneration_, _RegisterPolicy_);
    return _SimdElementWise<_SimdGeneration_, _RegisterPolicy_>::template _Blend<_DesiredType_>(_First, _Second, _Mask);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_,
    typename            _DesiredType_,
    typename            _VectorType_>
simd_stl_always_inline _VectorType_ _SimdBlend(
    _VectorType_    _First,
    _VectorType_    _Second,
    _VectorType_    _Mask) noexcept
{
    _VerifyRegisterPolicy(_SimdGeneration_, _RegisterPolicy_);
    return _SimdElementWise<_SimdGeneration_, _RegisterPolicy_>::template _Blend<_DesiredType_>(_First, _Second, _Mask);
}

__SIMD_STL_NUMERIC_NAMESPACE_END

#include <src/simd_stl/numeric/SimdElementWise.inl>