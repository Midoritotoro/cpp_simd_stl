#pragma once 

#include <src/simd_stl/numeric/IntrinBitcast.h>
#include <simd_stl/math/IntegralTypesConversions.h>


__SIMD_STL_NUMERIC_NAMESPACE_BEGIN

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_>
class _SimdConvertImplementation;

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_,
    typename            _DesiredType_,
    typename            _VectorType_>
simd_stl_always_inline auto _SimdToMask(_VectorType_ _Vector) noexcept;

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_,
    typename            _VectorType_,
    typename            _DesiredType_,
    typename            _MaskType_>
simd_stl_always_inline _VectorType_ _SimdToVector(_MaskType_ _Mask) noexcept;

#pragma region Sse2-Sse4.2 Simd convert

template <>
class _SimdConvertImplementation<arch::CpuFeature::SSE2, xmm128> {
    static constexpr auto _Generation   = arch::CpuFeature::SSE2;
    using _RegisterPolicy               = xmm128;

    template <typename _DesiredType_>
    using _Simd_mask_type = type_traits::__deduce_simd_mask_type<_Generation, _DesiredType_, _RegisterPolicy>;
public:
    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _Simd_mask_type<_DesiredType_> _ToMask(_VectorType_ _Vector) noexcept;

    template <
        typename _VectorType_,
        typename _DesiredType_>
    static simd_stl_always_inline _VectorType_ _ToVector(_Simd_mask_type<_DesiredType_> _Mask) noexcept;
};

template <>
class _SimdConvertImplementation<arch::CpuFeature::SSE3, xmm128> :
    public _SimdConvertImplementation<arch::CpuFeature::SSE2, xmm128>
{};

template <>
class _SimdConvertImplementation<arch::CpuFeature::SSSE3, xmm128> :
    public _SimdConvertImplementation<arch::CpuFeature::SSE3, xmm128>
{
    static constexpr auto _Generation   = arch::CpuFeature::SSE2;
    using _RegisterPolicy               = xmm128;

    template <typename _DesiredType_>
    using _Simd_mask_type = type_traits::__deduce_simd_mask_type<_Generation, _DesiredType_, _RegisterPolicy>;
public:
    template <
        typename _VectorType_,
        typename _DesiredType_>
    static simd_stl_always_inline _VectorType_ _ToVector(_Simd_mask_type<_DesiredType_> _Mask) noexcept;
};

template <>
class _SimdConvertImplementation<arch::CpuFeature::SSE41, xmm128> :
    public _SimdConvertImplementation<arch::CpuFeature::SSSE3, xmm128>
{};

template <>
class _SimdConvertImplementation<arch::CpuFeature::SSE42, xmm128> :
    public _SimdConvertImplementation<arch::CpuFeature::SSE41, xmm128>
{};

#pragma endregion

#pragma region Avx-Avx2 Simd convert

template <>
class _SimdConvertImplementation<arch::CpuFeature::AVX, ymm256>
{
    static constexpr auto _Generation   = arch::CpuFeature::AVX;
    using _RegisterPolicy               = ymm256;

    template <typename _DesiredType_>
    using _Simd_mask_type = type_traits::__deduce_simd_mask_type<_Generation, _DesiredType_, _RegisterPolicy>;
public:
    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _Simd_mask_type<_DesiredType_> _ToMask(_VectorType_ _Vector) noexcept;

    template <
        typename _VectorType_,
        typename _DesiredType_>
    static simd_stl_always_inline _VectorType_ _ToVector(_Simd_mask_type<_DesiredType_> _Mask) noexcept;
};

template <>
class _SimdConvertImplementation<arch::CpuFeature::AVX2, ymm256> :
    public _SimdConvertImplementation<arch::CpuFeature::AVX, ymm256>
{
    static constexpr auto _Generation   = arch::CpuFeature::AVX2;
    using _RegisterPolicy               = ymm256;

    template <typename _DesiredType_>
    using _Simd_mask_type = type_traits::__deduce_simd_mask_type<_Generation, _DesiredType_, _RegisterPolicy>;
public:
    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _Simd_mask_type<_DesiredType_> _ToMask(_VectorType_ _Vector) noexcept;
    
    template <
        typename _VectorType_,
        typename _DesiredType_>
    static simd_stl_always_inline _VectorType_ _ToVector(_Simd_mask_type<_DesiredType_> _Mask) noexcept;
};

#pragma endregion

#pragma region Avx512 Simd convert

template <>
class _SimdConvertImplementation<arch::CpuFeature::AVX512F, zmm512>
{
    static constexpr auto _Generation = arch::CpuFeature::AVX512F;
    using _RegisterPolicy = zmm512;

    template <typename _DesiredType_>
    using _Simd_mask_type = type_traits::__deduce_simd_mask_type<_Generation, _DesiredType_, _RegisterPolicy>;

    template <
        int32 _First_,
        int32 _Second_>
    static constexpr int32 _Max() noexcept {
        return (_First_ > _Second_) ? _First_ : _Second_;
    }
public:
    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _Simd_mask_type<_DesiredType_> _ToMask(_VectorType_ _Vector) noexcept;
    
    template <
        typename _VectorType_,
        typename _DesiredType_>
    static simd_stl_always_inline _VectorType_ _ToVector(_Simd_mask_type<_DesiredType_> _Mask) noexcept;
};

template <>
class _SimdConvertImplementation<arch::CpuFeature::AVX512BW, zmm512>:
    public _SimdConvertImplementation<arch::CpuFeature::AVX512F, zmm512>
{
    static constexpr auto _Generation = arch::CpuFeature::AVX512BW;
    using _RegisterPolicy = zmm512;

    template <typename _DesiredType_>
    using _Simd_mask_type = type_traits::__deduce_simd_mask_type<_Generation, _DesiredType_, _RegisterPolicy>;
public:
    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _Simd_mask_type<_DesiredType_> _ToMask(_VectorType_ _Vector) noexcept;

    template <
        typename _VectorType_,
        typename _DesiredType_>
    static simd_stl_always_inline _VectorType_ _ToVector(_Simd_mask_type<_DesiredType_> _Mask) noexcept;
};

template <>
class _SimdConvertImplementation<arch::CpuFeature::AVX512DQ, zmm512>:
    public _SimdConvertImplementation<arch::CpuFeature::AVX512F, zmm512>
{
    static constexpr auto _Generation = arch::CpuFeature::AVX512DQ;
    using _RegisterPolicy = zmm512;

    template <typename _DesiredType_>
    using _Simd_mask_type = type_traits::__deduce_simd_mask_type<_Generation, _DesiredType_, _RegisterPolicy>;
public:
    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _Simd_mask_type<_DesiredType_> _ToMask(_VectorType_ _Vector) noexcept;

    template <
        typename _VectorType_,
        typename _DesiredType_>
    static simd_stl_always_inline _VectorType_ _ToVector(_Simd_mask_type<_DesiredType_> _Mask) noexcept;
};

template <>
class _SimdConvertImplementation<arch::CpuFeature::AVX512VLF, ymm256> :
	public _SimdConvertImplementation<arch::CpuFeature::AVX2, ymm256>
{};

template <>
class _SimdConvertImplementation<arch::CpuFeature::AVX512VLBW, ymm256> :
	public _SimdConvertImplementation<arch::CpuFeature::AVX512VLF, ymm256>
{
    static constexpr auto _Generation = arch::CpuFeature::AVX512VLBW;
    using _RegisterPolicy = ymm256;

    template <typename _DesiredType_>
    using _Simd_mask_type = type_traits::__deduce_simd_mask_type<_Generation, _DesiredType_, _RegisterPolicy>;
public:
    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _Simd_mask_type<_DesiredType_> _ToMask(_VectorType_ _Vector) noexcept;

    template <
        typename _VectorType_,
        typename _DesiredType_>
    static simd_stl_always_inline _VectorType_ _ToVector(_Simd_mask_type<_DesiredType_> _Mask) noexcept;
};

template <>
class _SimdConvertImplementation<arch::CpuFeature::AVX512VLDQ, ymm256> :
	public _SimdConvertImplementation<arch::CpuFeature::AVX512VLBW, ymm256>
{
    static constexpr auto _Generation = arch::CpuFeature::AVX512VLDQ;
    using _RegisterPolicy = ymm256;

    template <typename _DesiredType_>
    using _Simd_mask_type = type_traits::__deduce_simd_mask_type<_Generation, _DesiredType_, _RegisterPolicy>;
public:
    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _Simd_mask_type<_DesiredType_> _ToMask(_VectorType_ _Vector) noexcept;

    template <
        typename _VectorType_,
        typename _DesiredType_>
    static simd_stl_always_inline _VectorType_ _ToVector(_Simd_mask_type<_DesiredType_> _Mask) noexcept;
};

template <>
class _SimdConvertImplementation<arch::CpuFeature::AVX512VLF, xmm128> :
	public _SimdConvertImplementation<arch::CpuFeature::SSE42, xmm128>
{};

template <>
class _SimdConvertImplementation<arch::CpuFeature::AVX512VLBW, xmm128> :
	public _SimdConvertImplementation<arch::CpuFeature::AVX512VLF, xmm128>
{
    static constexpr auto _Generation = arch::CpuFeature::AVX512VLBW;
    using _RegisterPolicy = xmm128;

    template <typename _DesiredType_>
    using _Simd_mask_type = type_traits::__deduce_simd_mask_type<_Generation, _DesiredType_, _RegisterPolicy>;
public:
    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _Simd_mask_type<_DesiredType_> _ToMask(_VectorType_ _Vector) noexcept;

    template <
        typename _VectorType_,
        typename _DesiredType_>
    static simd_stl_always_inline _VectorType_ _ToVector(_Simd_mask_type<_DesiredType_> _Mask) noexcept;
};

template <>
class _SimdConvertImplementation<arch::CpuFeature::AVX512VLDQ, xmm128> :
	public _SimdConvertImplementation<arch::CpuFeature::AVX512VLBW, xmm128>
{
    static constexpr auto _Generation = arch::CpuFeature::AVX512VLDQ;
    using _RegisterPolicy = xmm128;

    template <typename _DesiredType_>
    using _Simd_mask_type = type_traits::__deduce_simd_mask_type<_Generation, _DesiredType_, _RegisterPolicy>;
public:
    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _Simd_mask_type<_DesiredType_> _ToMask(_VectorType_ _Vector) noexcept;

    template <
        typename _VectorType_,
        typename _DesiredType_>
    static simd_stl_always_inline _VectorType_ _ToVector(_Simd_mask_type<_DesiredType_> _Mask) noexcept;
};


template <>
class _SimdConvertImplementation<arch::CpuFeature::AVX512VLBWDQ, ymm256>
{
    static constexpr auto _Generation = arch::CpuFeature::AVX512VLBWDQ;
    using _RegisterPolicy = ymm256;

    template <typename _DesiredType_>
    using _Simd_mask_type = type_traits::__deduce_simd_mask_type<_Generation, _DesiredType_, _RegisterPolicy>;
public:
    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _Simd_mask_type<_DesiredType_> _ToMask(_VectorType_ _Vector) noexcept;

    template <
        typename _VectorType_,
        typename _DesiredType_>
    static simd_stl_always_inline _VectorType_ _ToVector(_Simd_mask_type<_DesiredType_> _Mask) noexcept;
};


template <>
class _SimdConvertImplementation<arch::CpuFeature::AVX512VLBWDQ, xmm128>
{
    static constexpr auto _Generation = arch::CpuFeature::AVX512VLBWDQ;
    using _RegisterPolicy = xmm128;

    template <typename _DesiredType_>
    using _Simd_mask_type = type_traits::__deduce_simd_mask_type<_Generation, _DesiredType_, _RegisterPolicy>;
public:
    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _Simd_mask_type<_DesiredType_> _ToMask(_VectorType_ _Vector) noexcept;

    template <
        typename _VectorType_,
        typename _DesiredType_>
    static simd_stl_always_inline _VectorType_ _ToVector(_Simd_mask_type<_DesiredType_> _Mask) noexcept;
};

#pragma endregion

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_,
    typename            _DesiredType_,
    typename            _VectorType_>
simd_stl_always_inline auto _SimdToMask(_VectorType_ _Vector) noexcept {
    _VerifyRegisterPolicy(_SimdGeneration_, _RegisterPolicy_)

    if constexpr (std::is_integral_v<_VectorType_>)
        return _Vector;
    else
        return _SimdConvertImplementation<_SimdGeneration_, _RegisterPolicy_>::template _ToMask<_DesiredType_>(_Vector);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_,
    typename            _VectorType_,
    typename            _DesiredType_,
    typename            _MaskType_>
simd_stl_always_inline _VectorType_ _SimdToVector(_MaskType_ _Mask) noexcept {
    _VerifyRegisterPolicy(_SimdGeneration_, _RegisterPolicy_)

    if constexpr (_Is_intrin_type_v<_MaskType_>)
        return _IntrinBitcast<_VectorType_>(_Mask);
    else
        return _SimdConvertImplementation<_SimdGeneration_, _RegisterPolicy_>::template _ToVector<_VectorType_, _DesiredType_>(_Mask);
}

__SIMD_STL_NUMERIC_NAMESPACE_END

#include <src/simd_stl/numeric/SimdConvert.inl>
