#pragma once 

#include <src/simd_stl/numeric/IntrinBitcast.h>
#include <simd_stl/math/IntegralTypesConversions.h>


__SIMD_STL_NUMERIC_NAMESPACE_BEGIN

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_>
class __simd_convert;

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_,
    typename            _DesiredType_,
    typename            _VectorType_>
simd_stl_always_inline auto __simd_to_mask(_VectorType_ __vector) noexcept;

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_,
    typename            _VectorType_,
    typename            _DesiredType_,
    typename            _MaskType_>
simd_stl_always_inline _VectorType_ __simd_to_vector(_MaskType_ __mask) noexcept;

#pragma region Sse2-Sse4.2 Simd convert

template <>
class __simd_convert<arch::CpuFeature::SSE2, xmm128> {
    static constexpr auto __generation  = arch::CpuFeature::SSE2;
    using __register_policy             = xmm128;

    template <typename _DesiredType_>
    using __simd_mask_type = type_traits::__deduce_simd_mask_type<__generation, _DesiredType_, __register_policy>;
public:
    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline __simd_mask_type<_DesiredType_> __to_mask(_VectorType_ __vector) noexcept;

    template <
        typename _VectorType_,
        typename _DesiredType_>
    static simd_stl_always_inline _VectorType_ __to_vector(__simd_mask_type<_DesiredType_> __mask) noexcept;
};

template <>
class __simd_convert<arch::CpuFeature::SSE3, xmm128> :
    public __simd_convert<arch::CpuFeature::SSE2, xmm128>
{};

template <>
class __simd_convert<arch::CpuFeature::SSSE3, xmm128> :
    public __simd_convert<arch::CpuFeature::SSE3, xmm128>
{
    static constexpr auto __generation  = arch::CpuFeature::SSE2;
    using __register_policy             = xmm128;

    template <typename _DesiredType_>
    using __simd_mask_type = type_traits::__deduce_simd_mask_type<__generation, _DesiredType_, __register_policy>;
public:
    template <
        typename _VectorType_,
        typename _DesiredType_>
    static simd_stl_always_inline _VectorType_ __to_vector(__simd_mask_type<_DesiredType_> __mask) noexcept;
};

template <>
class __simd_convert<arch::CpuFeature::SSE41, xmm128> :
    public __simd_convert<arch::CpuFeature::SSSE3, xmm128>
{};

template <>
class __simd_convert<arch::CpuFeature::SSE42, xmm128> :
    public __simd_convert<arch::CpuFeature::SSE41, xmm128>
{};

#pragma endregion

#pragma region Avx-Avx2 Simd convert

template <>
class __simd_convert<arch::CpuFeature::AVX2, xmm128> :
    public __simd_convert<arch::CpuFeature::SSE42, xmm128>
{};

template <>
class __simd_convert<arch::CpuFeature::AVX2, ymm256>
{
    static constexpr auto __generation  = arch::CpuFeature::AVX2;
    using __register_policy             = ymm256;

    template <typename _DesiredType_>
    using __simd_mask_type = type_traits::__deduce_simd_mask_type<__generation, _DesiredType_, __register_policy>;
public:
    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline __simd_mask_type<_DesiredType_> __to_mask(_VectorType_ _Vector) noexcept;
    
    template <
        typename _VectorType_,
        typename _DesiredType_>
    static simd_stl_always_inline _VectorType_ __to_vector(__simd_mask_type<_DesiredType_> _Mask) noexcept;
};

#pragma endregion

#pragma region Avx512 Simd convert

template <>
class __simd_convert<arch::CpuFeature::AVX512F, zmm512>
{
    static constexpr auto __generation = arch::CpuFeature::AVX512F;
    using __register_policy = zmm512;

    template <typename _DesiredType_>
    using __simd_mask_type = type_traits::__deduce_simd_mask_type<__generation, _DesiredType_, __register_policy>;

    template <
        int32 __first_,
        int32 _Second_>
    static constexpr int32 _Max() noexcept {
        return (__first_ > _Second_) ? __first_ : _Second_;
    }
public:
    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline __simd_mask_type<_DesiredType_> __to_mask(_VectorType_ _Vector) noexcept;
    
    template <
        typename _VectorType_,
        typename _DesiredType_>
    static simd_stl_always_inline _VectorType_ __to_vector(__simd_mask_type<_DesiredType_> _Mask) noexcept;
};

template <>
class __simd_convert<arch::CpuFeature::AVX512BW, zmm512>:
    public __simd_convert<arch::CpuFeature::AVX512F, zmm512>
{
    static constexpr auto __generation = arch::CpuFeature::AVX512BW;
    using __register_policy = zmm512;

    template <typename _DesiredType_>
    using __simd_mask_type = type_traits::__deduce_simd_mask_type<__generation, _DesiredType_, __register_policy>;
public:
    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline __simd_mask_type<_DesiredType_> __to_mask(_VectorType_ _Vector) noexcept;

    template <
        typename _VectorType_,
        typename _DesiredType_>
    static simd_stl_always_inline _VectorType_ __to_vector(__simd_mask_type<_DesiredType_> _Mask) noexcept;
};

template <>
class __simd_convert<arch::CpuFeature::AVX512DQ, zmm512>:
    public __simd_convert<arch::CpuFeature::AVX512F, zmm512>
{
    static constexpr auto __generation = arch::CpuFeature::AVX512DQ;
    using __register_policy = zmm512;

    template <typename _DesiredType_>
    using __simd_mask_type = type_traits::__deduce_simd_mask_type<__generation, _DesiredType_, __register_policy>;
public:
    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline __simd_mask_type<_DesiredType_> __to_mask(_VectorType_ _Vector) noexcept;

    template <
        typename _VectorType_,
        typename _DesiredType_>
    static simd_stl_always_inline _VectorType_ __to_vector(__simd_mask_type<_DesiredType_> _Mask) noexcept;
};

template <>
class __simd_convert<arch::CpuFeature::AVX512VLF, ymm256> :
	public __simd_convert<arch::CpuFeature::AVX2, ymm256>
{};

template <>
class __simd_convert<arch::CpuFeature::AVX512VLBW, ymm256> :
	public __simd_convert<arch::CpuFeature::AVX512VLF, ymm256>
{
    static constexpr auto __generation = arch::CpuFeature::AVX512VLBW;
    using __register_policy = ymm256;

    template <typename _DesiredType_>
    using __simd_mask_type = type_traits::__deduce_simd_mask_type<__generation, _DesiredType_, __register_policy>;
public:
    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline __simd_mask_type<_DesiredType_> __to_mask(_VectorType_ _Vector) noexcept;

    template <
        typename _VectorType_,
        typename _DesiredType_>
    static simd_stl_always_inline _VectorType_ __to_vector(__simd_mask_type<_DesiredType_> _Mask) noexcept;
};

template <>
class __simd_convert<arch::CpuFeature::AVX512VLDQ, ymm256> :
	public __simd_convert<arch::CpuFeature::AVX512VLBW, ymm256>
{
    static constexpr auto __generation = arch::CpuFeature::AVX512VLDQ;
    using __register_policy = ymm256;

    template <typename _DesiredType_>
    using __simd_mask_type = type_traits::__deduce_simd_mask_type<__generation, _DesiredType_, __register_policy>;
public:
    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline __simd_mask_type<_DesiredType_> __to_mask(_VectorType_ _Vector) noexcept;

    template <
        typename _VectorType_,
        typename _DesiredType_>
    static simd_stl_always_inline _VectorType_ __to_vector(__simd_mask_type<_DesiredType_> _Mask) noexcept;
};

template <>
class __simd_convert<arch::CpuFeature::AVX512VLF, xmm128> :
	public __simd_convert<arch::CpuFeature::SSE42, xmm128>
{};

template <>
class __simd_convert<arch::CpuFeature::AVX512VLBW, xmm128> :
	public __simd_convert<arch::CpuFeature::AVX512VLF, xmm128>
{
    static constexpr auto __generation = arch::CpuFeature::AVX512VLBW;
    using __register_policy = xmm128;

    template <typename _DesiredType_>
    using __simd_mask_type = type_traits::__deduce_simd_mask_type<__generation, _DesiredType_, __register_policy>;
public:
    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline __simd_mask_type<_DesiredType_> __to_mask(_VectorType_ _Vector) noexcept;

    template <
        typename _VectorType_,
        typename _DesiredType_>
    static simd_stl_always_inline _VectorType_ __to_vector(__simd_mask_type<_DesiredType_> _Mask) noexcept;
};

template <>
class __simd_convert<arch::CpuFeature::AVX512VLDQ, xmm128> :
	public __simd_convert<arch::CpuFeature::AVX512VLBW, xmm128>
{
    static constexpr auto __generation = arch::CpuFeature::AVX512VLDQ;
    using __register_policy = xmm128;

    template <typename _DesiredType_>
    using __simd_mask_type = type_traits::__deduce_simd_mask_type<__generation, _DesiredType_, __register_policy>;
public:
    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline __simd_mask_type<_DesiredType_> __to_mask(_VectorType_ _Vector) noexcept;

    template <
        typename _VectorType_,
        typename _DesiredType_>
    static simd_stl_always_inline _VectorType_ __to_vector(__simd_mask_type<_DesiredType_> _Mask) noexcept;
};


template <>
class __simd_convert<arch::CpuFeature::AVX512VLBWDQ, ymm256>
{
    static constexpr auto __generation = arch::CpuFeature::AVX512VLBWDQ;
    using __register_policy = ymm256;

    template <typename _DesiredType_>
    using __simd_mask_type = type_traits::__deduce_simd_mask_type<__generation, _DesiredType_, __register_policy>;
public:
    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline __simd_mask_type<_DesiredType_> __to_mask(_VectorType_ _Vector) noexcept;

    template <
        typename _VectorType_,
        typename _DesiredType_>
    static simd_stl_always_inline _VectorType_ __to_vector(__simd_mask_type<_DesiredType_> _Mask) noexcept;
};


template <>
class __simd_convert<arch::CpuFeature::AVX512VLBWDQ, xmm128>
{
    static constexpr auto __generation = arch::CpuFeature::AVX512VLBWDQ;
    using __register_policy = xmm128;

    template <typename _DesiredType_>
    using __simd_mask_type = type_traits::__deduce_simd_mask_type<__generation, _DesiredType_, __register_policy>;
public:
    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline __simd_mask_type<_DesiredType_> __to_mask(_VectorType_ _Vector) noexcept;

    template <
        typename _VectorType_,
        typename _DesiredType_>
    static simd_stl_always_inline _VectorType_ __to_vector(__simd_mask_type<_DesiredType_> _Mask) noexcept;
};

#pragma endregion

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_,
    typename            _DesiredType_,
    typename            _VectorType_>
simd_stl_always_inline auto __simd_to_mask(_VectorType_ __vector) noexcept {
    __verify_register_policy(_SimdGeneration_, _RegisterPolicy_)

    if constexpr (std::is_integral_v<_VectorType_>)
        return __vector;
    else
        return __simd_convert<_SimdGeneration_, _RegisterPolicy_>::template __to_mask<_DesiredType_>(__vector);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_,
    typename            _VectorType_,
    typename            _DesiredType_,
    typename            _MaskType_>
simd_stl_always_inline _VectorType_ __simd_to_vector(_MaskType_ __mask) noexcept {
    __verify_register_policy(_SimdGeneration_, _RegisterPolicy_)

    if constexpr (__is_intrin_type_v<_MaskType_>)
        return __intrin_bitcast<_VectorType_>(__mask);
    else
        return __simd_convert<_SimdGeneration_, _RegisterPolicy_>::template __to_vector<_VectorType_, _DesiredType_>(__mask);
}

__SIMD_STL_NUMERIC_NAMESPACE_END

#include <src/simd_stl/numeric/SimdConvert.inl>
