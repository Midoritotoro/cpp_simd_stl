#pragma once 

#include <src/simd_stl/numeric/SimdIntegralTypesCheck.h>
#include <src/simd_stl/numeric/SimdConvert.h>

#include <simd_stl/numeric/BasicSimdShuffleMask.h>
#include <src/simd_stl/numeric/ShuffleTables.h>

#include <src/simd_stl/numeric/MaskExpand.h>
#include <src/simd_stl/utility/Assert.h>

#include <src/simd_stl/algorithm/AdvanceBytes.h>


__SIMD_STL_NUMERIC_NAMESPACE_BEGIN

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_>
class __simd_element_wise;

#pragma region Sse2-Sse4.2 Simd element wise 

template <>
class __simd_element_wise<arch::CpuFeature::SSE2, xmm128> {
    static constexpr auto __generation  = arch::CpuFeature::SSE2;
    using __register_policy             = xmm128;

    template <typename _DesiredType_>
    using __simd_mask_type = type_traits::__deduce_simd_mask_type<__generation, _DesiredType_, __register_policy>;
public:
    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline std::pair<int32, _VectorType_> __compress(
        _VectorType_    __vector,
        _VectorType_    __mask) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline std::pair<int32, _VectorType_> __compress(
        _VectorType_                    __vector,
        __simd_mask_type<_DesiredType_> __mask) noexcept;

    template <
        typename    _DesiredType_,
        typename    _VectorType_>
    static simd_stl_always_inline _VectorType_ __blend(
        _VectorType_ __first,
        _VectorType_ _Second,
        _VectorType_ _Mask) noexcept;

    template <
        typename    _DesiredType_,
        typename    _VectorType_>
    static simd_stl_always_inline _VectorType_ __blend(
        _VectorType_                        __first,
        _VectorType_                        _Second,
        __simd_mask_type<_DesiredType_>      _Mask) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ __reverse(_VectorType_ _Vector) noexcept;
};

template <>
class __simd_element_wise<arch::CpuFeature::SSE3, xmm128> :
    public __simd_element_wise<arch::CpuFeature::SSE2, xmm128>
{};

template <>
class __simd_element_wise<arch::CpuFeature::SSSE3, xmm128> :
    public __simd_element_wise<arch::CpuFeature::SSE3, xmm128>
{
    static constexpr auto __generation  = arch::CpuFeature::SSSE3;
    using __register_policy             = xmm128;

    template <typename _DesiredType_>
    using __simd_mask_type = type_traits::__deduce_simd_mask_type<__generation, _DesiredType_, __register_policy>;
public:
    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline std::pair<int32, _VectorType_> __compress(
        _VectorType_    __vector,
        _VectorType_    __mask) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline std::pair<int32, _VectorType_> __compress(
        _VectorType_                    __vector,
        __simd_mask_type<_DesiredType_> __mask) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ __reverse(_VectorType_ _Vector) noexcept;
};

template <>
class __simd_element_wise<arch::CpuFeature::SSE41, xmm128> :
    public __simd_element_wise<arch::CpuFeature::SSSE3, xmm128>
{
    static constexpr auto __generation   = arch::CpuFeature::SSE41;
    using __register_policy = xmm128;

    template <typename _DesiredType_>
    using __simd_mask_type = type_traits::__deduce_simd_mask_type<__generation, _DesiredType_, __register_policy>;
public:
    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline std::pair<int32, _VectorType_> __compress(
        _VectorType_    __vector,
        _VectorType_    __mask) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline std::pair<int32, _VectorType_> __compress(
        _VectorType_                    __vector,
        __simd_mask_type<_DesiredType_> __mask) noexcept;

    template <
        typename    _DesiredType_,
        typename    _VectorType_>
    static simd_stl_always_inline _VectorType_ __blend(
        _VectorType_ __first,
        _VectorType_ _Second,
        _VectorType_ _Mask) noexcept;

    template <
        typename    _DesiredType_,
        typename    _VectorType_>
    static simd_stl_always_inline _VectorType_ __blend(
        _VectorType_                        __first,
        _VectorType_                        _Second,
        __simd_mask_type<_DesiredType_>      _Mask) noexcept;
};

template <>
class __simd_element_wise<arch::CpuFeature::SSE42, xmm128> :
    public __simd_element_wise<arch::CpuFeature::SSE41, xmm128>
{};

#pragma endregion

#pragma region Avx-Avx2 Simd element wise

template <>
class __simd_element_wise<arch::CpuFeature::AVX2, xmm128>:
    public __simd_element_wise<arch::CpuFeature::SSE42, xmm128>
{};

template <>
class __simd_element_wise<arch::CpuFeature::AVX2, ymm256>
{
    static constexpr auto __generation  = arch::CpuFeature::AVX2;
    using __register_policy             = ymm256;

    template <typename _DesiredType_>
    using __simd_mask_type = type_traits::__deduce_simd_mask_type<__generation, _DesiredType_, __register_policy>;
public:
    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline std::pair<int32, _VectorType_> __compress(
        _VectorType_    __vector,
        _VectorType_    __mask) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline std::pair<int32, _VectorType_> __compress(
        _VectorType_                    __vector,
        __simd_mask_type<_DesiredType_> __mask) noexcept;

    template <
        typename    _DesiredType_,
        typename    _VectorType_>
    static simd_stl_always_inline _VectorType_ __blend(
        _VectorType_ __first,
        _VectorType_ __second,
        _VectorType_ __mask) noexcept;

    template <
        typename    _DesiredType_,
        typename    _VectorType_>
    static simd_stl_always_inline _VectorType_ __blend(
        _VectorType_                        __first,
        _VectorType_                        __second,
        __simd_mask_type<_DesiredType_>     __mask) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ __reverse(_VectorType_ __vector) noexcept;
};

#pragma endregion

#pragma region Avx512 Simd element wise

template <>
class __simd_element_wise<arch::CpuFeature::AVX512F, zmm512>
{
    static constexpr auto __generation   = arch::CpuFeature::AVX512F;
    using __register_policy = zmm512;

    template <typename _DesiredType_>
    using __simd_mask_type = type_traits::__deduce_simd_mask_type<__generation, _DesiredType_, __register_policy>;
public:
    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline std::pair<int32, _VectorType_> __compress(
        _VectorType_    __vector,
        _VectorType_    __mask) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline std::pair<int32, _VectorType_> __compress(
        _VectorType_                    __vector,
        __simd_mask_type<_DesiredType_> __mask) noexcept;

    template <
        typename    _DesiredType_,
        typename    _VectorType_>
    static simd_stl_always_inline _VectorType_ __blend(
        _VectorType_ __first,
        _VectorType_ __second,
        _VectorType_ __mask) noexcept;

    template <
        typename    _DesiredType_,
        typename    _VectorType_>
    static simd_stl_always_inline _VectorType_ __blend(
        _VectorType_                        __first,
        _VectorType_                        __second,
        __simd_mask_type<_DesiredType_>     __mask) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ __reverse(_VectorType_ __vector) noexcept;
};

template <>
class __simd_element_wise<arch::CpuFeature::AVX512BW, zmm512>:
    public __simd_element_wise<arch::CpuFeature::AVX512F, zmm512>
{
    static constexpr auto __generation   = arch::CpuFeature::AVX512BW;
    using __register_policy = zmm512;

    template <typename _DesiredType_>
    using __simd_mask_type = type_traits::__deduce_simd_mask_type<__generation, _DesiredType_, __register_policy>;
public:
    template <
        typename    _DesiredType_,
        typename    _VectorType_>
    static simd_stl_always_inline _VectorType_ __blend(
        _VectorType_ __first,
        _VectorType_ __second,
        _VectorType_ __mask) noexcept;

    template <
        typename    _DesiredType_,
        typename    _VectorType_>
    static simd_stl_always_inline _VectorType_ __blend(
        _VectorType_                        __first,
        _VectorType_                        __second,
        __simd_mask_type<_DesiredType_>     __mask) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ __reverse(_VectorType_ __vector) noexcept;
};

template <>
class __simd_element_wise<arch::CpuFeature::AVX512DQ, zmm512> :
    public __simd_element_wise<arch::CpuFeature::AVX512F, zmm512>
{};

template <>
class __simd_element_wise<arch::CpuFeature::AVX512BWDQ, zmm512> :
    public __simd_element_wise<arch::CpuFeature::AVX512BW, zmm512>
{};

template <>
class __simd_element_wise<arch::CpuFeature::AVX512VLF, ymm256> :
    public __simd_element_wise<arch::CpuFeature::AVX2, ymm256>
{
    static constexpr auto __generation  = arch::CpuFeature::AVX512VLF;
    using __register_policy             = ymm256;

    template <typename _DesiredType_>
    using __simd_mask_type = type_traits::__deduce_simd_mask_type<__generation, _DesiredType_, __register_policy>;
public:
    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline std::pair<int32, _VectorType_> __compress(
        _VectorType_    __vector,
        _VectorType_    __mask) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline std::pair<int32, _VectorType_> __compress(
        _VectorType_                    __vector,
        __simd_mask_type<_DesiredType_> __mask) noexcept;
};

template <>
class __simd_element_wise<arch::CpuFeature::AVX512VLBW, ymm256> :
    public __simd_element_wise<arch::CpuFeature::AVX512VLF, ymm256>
{};

template <>
class __simd_element_wise<arch::CpuFeature::AVX512VLDQ, ymm256> :
    public __simd_element_wise<arch::CpuFeature::AVX512VLF, ymm256>
{};

template <>
class __simd_element_wise<arch::CpuFeature::AVX512VLBWDQ, ymm256> :
    public __simd_element_wise<arch::CpuFeature::AVX512VLBW, ymm256>
{};

template <>
class __simd_element_wise<arch::CpuFeature::AVX512VLF, xmm128> :
    public __simd_element_wise<arch::CpuFeature::SSE42, xmm128>
{
    static constexpr auto __generation  = arch::CpuFeature::AVX512VLF;
    using __register_policy             = xmm128;

    template <typename _DesiredType_>
    using __simd_mask_type = type_traits::__deduce_simd_mask_type<__generation, _DesiredType_, __register_policy>;
public:
    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline std::pair<int32, _VectorType_> __compress(
        _VectorType_    __vector,
        _VectorType_    __mask) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline std::pair<int32, _VectorType_> __compress(
        _VectorType_                    __vector,
        __simd_mask_type<_DesiredType_> __mask) noexcept;
};

template <>
class __simd_element_wise<arch::CpuFeature::AVX512VLBW, xmm128> :
    public __simd_element_wise<arch::CpuFeature::AVX512VLF, xmm128>
{};

template <>
class __simd_element_wise<arch::CpuFeature::AVX512VLDQ, xmm128> :
    public __simd_element_wise<arch::CpuFeature::AVX512VLF, xmm128>
{};

template <>
class __simd_element_wise<arch::CpuFeature::AVX512VLBWDQ, xmm128> :
    public __simd_element_wise<arch::CpuFeature::AVX512VLBW, xmm128>
{};

#pragma endregion

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_,
    typename            _DesiredType_,
    typename            _VectorType_>
simd_stl_always_inline _VectorType_ __simd_reverse(_VectorType_ __vector) noexcept {
    __verify_register_policy(_SimdGeneration_, _RegisterPolicy_);
    return __simd_element_wise<_SimdGeneration_, _RegisterPolicy_>::template __reverse<_DesiredType_>(__vector);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_,
    typename            _DesiredType_,
    typename            _VectorType_>
simd_stl_always_inline _VectorType_ __simd_blend(
    _VectorType_                            __first,
    _VectorType_                            __second,
    type_traits::__deduce_simd_mask_type<_SimdGeneration_,
        _DesiredType_, _RegisterPolicy_>    __mask) noexcept
{
    __verify_register_policy(_SimdGeneration_, _RegisterPolicy_);
    return __simd_element_wise<_SimdGeneration_, _RegisterPolicy_>::template __blend<_DesiredType_>(__first, __second, __mask);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_,
    typename            _DesiredType_,
    typename            _VectorType_>
simd_stl_always_inline _VectorType_ __simd_blend(
    _VectorType_    __first,
    _VectorType_    __second,
    _VectorType_    __mask) noexcept
{
    __verify_register_policy(_SimdGeneration_, _RegisterPolicy_);
    return __simd_element_wise<_SimdGeneration_, _RegisterPolicy_>::template __blend<_DesiredType_>(__first, __second, __mask);
}


template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_,
    typename            _DesiredType_,
    typename            _VectorType_>
simd_stl_always_inline std::pair<int32, _VectorType_> __simd_compress(
    _VectorType_    __vector,
    _VectorType_    __mask) noexcept
{
    __verify_register_policy(_SimdGeneration_, _RegisterPolicy_);
    return  __simd_element_wise<_SimdGeneration_, _RegisterPolicy_>::template __compress<_DesiredType_>(__vector, __mask);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_,
    typename            _DesiredType_,
    typename            _VectorType_>
simd_stl_always_inline std::pair<int32, _VectorType_> __simd_compress(
    _VectorType_                            __vector,
    type_traits::__deduce_simd_mask_type<_SimdGeneration_,
        _DesiredType_, _RegisterPolicy_>    __mask) noexcept
{
    __verify_register_policy(_SimdGeneration_, _RegisterPolicy_);
    return  __simd_element_wise<_SimdGeneration_, _RegisterPolicy_>::template __compress<_DesiredType_>(__vector, __mask);
}


__SIMD_STL_NUMERIC_NAMESPACE_END

#include <src/simd_stl/numeric/SimdElementWise.inl>