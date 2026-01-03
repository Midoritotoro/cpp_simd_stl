#pragma once 

#include <src/simd_stl/numeric/IntrinBitcast.h>
#include <src/simd_stl/numeric/SimdConvert.h>
#include <src/simd_stl/type_traits/OperatorWrappers.h>


__SIMD_STL_NUMERIC_NAMESPACE_BEGIN

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_,
    typename            _VectorType_>
simd_stl_always_inline _VectorType_ _SimdBitNot(_VectorType_ _Vector) noexcept;

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_>
class _SimdCompareImplementation;

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_,
    class               _DesiredType_,
    class               _CompareType_,
    class               _VectorType_>
simd_stl_nodiscard simd_stl_always_inline _VectorType_ _SimdCompare(
    _VectorType_ _Left,
    _VectorType_ _Right) noexcept;

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_,
    class               _DesiredType_,
    class               _CompareType_,
    class               _VectorType_>
simd_stl_nodiscard simd_stl_always_inline type_traits::__deduce_simd_mask_type<_SimdGeneration_,
    _DesiredType_, _RegisterPolicy_> _SimdMaskCompare(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept;

#pragma region Sse2-Sse4.2 Simd compare

template <>
class _SimdCompareImplementation<arch::CpuFeature::SSE2, xmm128> {
    static constexpr auto _Generation   = arch::CpuFeature::SSE2;
    using _RegisterPolicy               = xmm128;

    template <class _DesiredType_>
    using _Simd_mask_type = type_traits::__deduce_simd_mask_type<_Generation, _DesiredType_, _RegisterPolicy>;
public:
    template <
        class   _DesiredType_,
        class   _CompareType_,
        class   _VectorType_>
    static simd_stl_always_inline _Simd_mask_type<_DesiredType_> _MaskCompare(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept;

    template <
        typename _DesiredType_,
        class    _CompareType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _Compare(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept;

    template <
        typename _DesiredType_,
        class    _CompareType_,
        typename _VectorType_>
    static simd_stl_always_inline auto _NativeCompare(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _CompareEqual(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _CompareLess(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _CompareGreater(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept;
};

template <>
class _SimdCompareImplementation<arch::CpuFeature::SSE3, xmm128> :
    public _SimdCompareImplementation<arch::CpuFeature::SSE2, xmm128>
{};

template <>
class _SimdCompareImplementation<arch::CpuFeature::SSSE3, xmm128> :
    public _SimdCompareImplementation<arch::CpuFeature::SSE3, xmm128>
{
    static constexpr auto _Generation   = arch::CpuFeature::SSSE3;
    using _RegisterPolicy               = xmm128;

    template <class _DesiredType_>
    using _Simd_mask_type = type_traits::__deduce_simd_mask_type<_Generation, _DesiredType_, _RegisterPolicy>;
public:
    template <
        class   _DesiredType_,
        class   _CompareType_,
        class   _VectorType_>
    static simd_stl_always_inline _Simd_mask_type<_DesiredType_> _MaskCompare(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept;
};

template <>
class _SimdCompareImplementation<arch::CpuFeature::SSE41, xmm128> :
    public _SimdCompareImplementation<arch::CpuFeature::SSSE3, xmm128>
{
    static constexpr auto _Generation   = arch::CpuFeature::SSE41;
    using _RegisterPolicy               = xmm128;

    template <class _DesiredType_>
    using _Simd_mask_type = type_traits::__deduce_simd_mask_type<_Generation, _DesiredType_, _RegisterPolicy>;
public:
    template <
        class   _DesiredType_,
        class   _CompareType_,
        class   _VectorType_>
    static simd_stl_always_inline _Simd_mask_type<_DesiredType_> _MaskCompare(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept;

    template <
        typename _DesiredType_,
        class    _CompareType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _Compare(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept;

    template <
        typename _DesiredType_,
        class    _CompareType_,
        typename _VectorType_>
    static simd_stl_always_inline auto _NativeCompare(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _CompareEqual(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept;
};

template <>
class _SimdCompareImplementation<arch::CpuFeature::SSE42, xmm128>:
    public _SimdCompareImplementation<arch::CpuFeature::SSE41, xmm128>
{
    static constexpr auto _Generation   = arch::CpuFeature::SSE42;
    using _RegisterPolicy               = xmm128;

    template <class _DesiredType_>
    using _Simd_mask_type = type_traits::__deduce_simd_mask_type<_Generation, _DesiredType_, _RegisterPolicy>;
public:
    template <
        class   _DesiredType_,
        class   _CompareType_,
        class   _VectorType_>
    static simd_stl_always_inline _Simd_mask_type<_DesiredType_> _MaskCompare(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept;

    template <
        typename _DesiredType_,
        class    _CompareType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _Compare(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept;

    template <
        typename _DesiredType_,
        class    _CompareType_,
        typename _VectorType_>
    static simd_stl_always_inline auto _NativeCompare(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _CompareGreater(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _CompareLess(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept;
};

#pragma endregion

#pragma region Avx Simd compare

template <>
class _SimdCompareImplementation<arch::CpuFeature::AVX, ymm256> {
    static constexpr auto _Generation   = arch::CpuFeature::AVX;
    using _RegisterPolicy               = ymm256;

    template <class _DesiredType_>
    using _Simd_mask_type = type_traits::__deduce_simd_mask_type<_Generation, _DesiredType_, _RegisterPolicy>;
public:
    template <
        class   _DesiredType_,
        class   _CompareType_,
        class   _VectorType_>
    static simd_stl_always_inline _Simd_mask_type<_DesiredType_> _MaskCompare(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept;

    template <
        typename _DesiredType_,
        class    _CompareType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _Compare(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept;

    template <
        typename _DesiredType_,
        class    _CompareType_,
        typename _VectorType_>
    static simd_stl_always_inline auto _NativeCompare(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _CompareEqual(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _CompareLess(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _CompareGreater(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept;
};

template <>
class _SimdCompareImplementation<arch::CpuFeature::AVX2, ymm256>:
    public _SimdCompareImplementation<arch::CpuFeature::AVX, ymm256>
{
    static constexpr auto _Generation   = arch::CpuFeature::AVX2;
    using _RegisterPolicy               = ymm256;

    template <class _DesiredType_>
    using _Simd_mask_type = type_traits::__deduce_simd_mask_type<_Generation, _DesiredType_, _RegisterPolicy>;
public:
    template <
        class   _DesiredType_,
        class   _CompareType_,
        class   _VectorType_>
    static simd_stl_always_inline _Simd_mask_type<_DesiredType_> _MaskCompare(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept;

    template <
        typename _DesiredType_,
        class    _CompareType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _Compare(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept;

    template <
        typename _DesiredType_,
        class    _CompareType_,
        typename _VectorType_>
    static simd_stl_always_inline auto _NativeCompare(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _CompareEqual(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _CompareLess(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _CompareGreater(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept;
};


#pragma endregion

#pragma region Avx512 Simd compare

template <>
class _SimdCompareImplementation<arch::CpuFeature::AVX512F, zmm512> {
    static constexpr auto _Generation = arch::CpuFeature::AVX512F;
    using _RegisterPolicy = zmm512;

    template <class _DesiredType_>
    using _Simd_mask_type = type_traits::__deduce_simd_mask_type<_Generation, _DesiredType_, _RegisterPolicy>;

    template <
        class _DesiredType_,
        class _CompareType_,
        class _VectorType_>
    static simd_stl_always_inline _VectorType_ _BlockwiseCompare(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept;
public:
    template <
        class   _DesiredType_,
        class   _CompareType_,
        class   _VectorType_>
    static simd_stl_always_inline _Simd_mask_type<_DesiredType_> _MaskCompare(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept;

    template <
        typename _DesiredType_,
        class    _CompareType_,
        typename _VectorType_>
    static simd_stl_always_inline auto _NativeCompare(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept;

    template <
        typename _DesiredType_,
        class    _CompareType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _Compare(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _Simd_mask_type<_DesiredType_> _MaskCompareEqual(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _Simd_mask_type<_DesiredType_> _MaskCompareLess(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _Simd_mask_type<_DesiredType_> _MaskCompareGreater(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept;
};

template <>
class _SimdCompareImplementation<arch::CpuFeature::AVX512BW, zmm512>:
    public _SimdCompareImplementation<arch::CpuFeature::AVX512F, zmm512> 
{
    static constexpr auto _Generation   = arch::CpuFeature::AVX512BW;
    using _RegisterPolicy               = zmm512;

    template <class _DesiredType_>
    using _Simd_mask_type = type_traits::__deduce_simd_mask_type<_Generation, _DesiredType_, _RegisterPolicy>;
public:
    template <
        class   _DesiredType_,
        class   _CompareType_,
        class   _VectorType_>
    static simd_stl_always_inline _Simd_mask_type<_DesiredType_> _MaskCompare(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept;

    template <
        typename _DesiredType_,
        class    _CompareType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _Compare(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _Simd_mask_type<_DesiredType_> _MaskCompareEqual(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept;

    template <
        typename _DesiredType_,
        class    _CompareType_,
        typename _VectorType_>
    static simd_stl_always_inline auto _NativeCompare(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _Simd_mask_type<_DesiredType_> _MaskCompareLess(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _Simd_mask_type<_DesiredType_> _MaskCompareGreater(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept;
};

template <>
class _SimdCompareImplementation<arch::CpuFeature::AVX512DQ, zmm512> :
    public _SimdCompareImplementation<arch::CpuFeature::AVX512BW, zmm512>
{};

template <>
class _SimdCompareImplementation<arch::CpuFeature::AVX512VLF, ymm256> :
	public _SimdCompareImplementation<arch::CpuFeature::AVX2, ymm256>
{
    static constexpr auto _Generation = arch::CpuFeature::AVX512VLF;
    using _RegisterPolicy = ymm256;

    template <class _DesiredType_>
    using _Simd_mask_type = type_traits::__deduce_simd_mask_type<_Generation, _DesiredType_, _RegisterPolicy>;
public:
    template <
        class   _DesiredType_,
        class   _CompareType_,
        class   _VectorType_>
    static simd_stl_always_inline _Simd_mask_type<_DesiredType_> _MaskCompare(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _Simd_mask_type<_DesiredType_> _MaskCompareEqual(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept;

    template <
        typename _DesiredType_,
        class    _CompareType_,
        typename _VectorType_>
    static simd_stl_always_inline auto _NativeCompare(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _Simd_mask_type<_DesiredType_> _MaskCompareLess(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _Simd_mask_type<_DesiredType_> _MaskCompareGreater(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept;
};

template <>
class _SimdCompareImplementation<arch::CpuFeature::AVX512VLBW, ymm256> :
	public _SimdCompareImplementation<arch::CpuFeature::AVX512VLF, ymm256>
{
    static constexpr auto _Generation = arch::CpuFeature::AVX512VLBW;
    using _RegisterPolicy = ymm256;

    template <class _DesiredType_>
    using _Simd_mask_type = type_traits::__deduce_simd_mask_type<_Generation, _DesiredType_, _RegisterPolicy>;
public:
    template <
        class   _DesiredType_,
        class   _CompareType_,
        class   _VectorType_>
    static simd_stl_always_inline _Simd_mask_type<_DesiredType_> _MaskCompare(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _Simd_mask_type<_DesiredType_> _MaskCompareEqual(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept;

    template <
        typename _DesiredType_,
        class    _CompareType_,
        typename _VectorType_>
    static simd_stl_always_inline auto _NativeCompare(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _Simd_mask_type<_DesiredType_> _MaskCompareLess(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _Simd_mask_type<_DesiredType_> _MaskCompareGreater(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept;
};

template <>
class _SimdCompareImplementation<arch::CpuFeature::AVX512VLDQ, ymm256> :
	public _SimdCompareImplementation<arch::CpuFeature::AVX512VLF, ymm256>
{};

template <>
class _SimdCompareImplementation<arch::CpuFeature::AVX512VLBWDQ, ymm256> :
	public _SimdCompareImplementation<arch::CpuFeature::AVX512VLBW, ymm256>
{};

template <>
class _SimdCompareImplementation<arch::CpuFeature::AVX512VLF, xmm128> :
	public _SimdCompareImplementation<arch::CpuFeature::SSE42, xmm128>
{
    static constexpr auto _Generation = arch::CpuFeature::AVX512VLF;
    using _RegisterPolicy = xmm128;

    template <class _DesiredType_>
    using _Simd_mask_type = type_traits::__deduce_simd_mask_type<_Generation, _DesiredType_, _RegisterPolicy>;
public:
    template <
        class   _DesiredType_,
        class   _CompareType_,
        class   _VectorType_>
    static simd_stl_always_inline _Simd_mask_type<_DesiredType_> _MaskCompare(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _Simd_mask_type<_DesiredType_> _MaskCompareEqual(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept;

    template <
        typename _DesiredType_,
        class    _CompareType_,
        typename _VectorType_>
    static simd_stl_always_inline auto _NativeCompare(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _Simd_mask_type<_DesiredType_> _MaskCompareLess(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _Simd_mask_type<_DesiredType_> _MaskCompareGreater(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept;
};

template <>
class _SimdCompareImplementation<arch::CpuFeature::AVX512VLBW, xmm128> :
	public _SimdCompareImplementation<arch::CpuFeature::AVX512VLF, xmm128>
{
    static constexpr auto _Generation = arch::CpuFeature::AVX512VLBW;
    using _RegisterPolicy = xmm128;

    template <class _DesiredType_>
    using _Simd_mask_type = type_traits::__deduce_simd_mask_type<_Generation, _DesiredType_, _RegisterPolicy>;
public:
    template <
        class   _DesiredType_,
        class   _CompareType_,
        class   _VectorType_>
    static simd_stl_always_inline _Simd_mask_type<_DesiredType_> _MaskCompare(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _Simd_mask_type<_DesiredType_> _MaskCompareEqual(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept;

    template <
        typename _DesiredType_,
        class    _CompareType_,
        typename _VectorType_>
    static simd_stl_always_inline auto _NativeCompare(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _Simd_mask_type<_DesiredType_> _MaskCompareLess(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _Simd_mask_type<_DesiredType_> _MaskCompareGreater(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept;
};

template <>
class _SimdCompareImplementation<arch::CpuFeature::AVX512VLDQ, xmm128> :
	public _SimdCompareImplementation<arch::CpuFeature::AVX512VLF, xmm128>
{};

template <>
class _SimdCompareImplementation<arch::CpuFeature::AVX512VLBWDQ, xmm128> :
	public _SimdCompareImplementation<arch::CpuFeature::AVX512VLBW, xmm128>
{};

#pragma endregion 

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_,
    class               _DesiredType_,
    class               _CompareType_,
    class               _VectorType_>
simd_stl_nodiscard simd_stl_always_inline _VectorType_ _SimdCompare(
    _VectorType_ _Left,
    _VectorType_ _Right) noexcept
{
    _VerifyRegisterPolicy(_SimdGeneration_, _RegisterPolicy_);
    return _SimdCompareImplementation<_SimdGeneration_, _RegisterPolicy_>
        ::template _Compare<_DesiredType_, _CompareType_>(_Left, _Right);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_,
    class               _DesiredType_,
    class               _CompareType_,
    class               _VectorType_>
simd_stl_nodiscard simd_stl_always_inline type_traits::__deduce_simd_mask_type<_SimdGeneration_,
    _DesiredType_, _RegisterPolicy_> _SimdMaskCompare(
        _VectorType_ _Left,
        _VectorType_ _Right) noexcept
{
    _VerifyRegisterPolicy(_SimdGeneration_, _RegisterPolicy_);
    return _SimdCompareImplementation<_SimdGeneration_, _RegisterPolicy_>::template
        _MaskCompare<_DesiredType_, _CompareType_>(_Left, _Right);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_,
    class               _DesiredType_,
    class               _CompareType_,
    class               _VectorType_>
simd_stl_nodiscard simd_stl_always_inline auto _SimdNativeCompare(
    _VectorType_ _Left,
    _VectorType_ _Right) noexcept
{
    _VerifyRegisterPolicy(_SimdGeneration_, _RegisterPolicy_);
    return _SimdCompareImplementation<_SimdGeneration_, _RegisterPolicy_>
        ::template _NativeCompare<_DesiredType_, _CompareType_>(_Left, _Right);
}

template <
    class       _BasicSimd_, 
    typename    _DesiredType_,
    class       _CompareType_>
using _Native_compare_return_type = _Native_compare_return_type_helper<_BasicSimd_,
    type_traits::invoke_result_type<decltype(_SimdNativeCompare<_BasicSimd_::_Generation,
        typename _BasicSimd_::policy_type, _DesiredType_, _CompareType_, typename _BasicSimd_::vector_type>),
    typename _BasicSimd_::vector_type, typename _BasicSimd_::vector_type>, _DesiredType_>;

__SIMD_STL_NUMERIC_NAMESPACE_END

#include <src/simd_stl/numeric/SimdCompare.inl>
