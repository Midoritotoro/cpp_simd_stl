#pragma once 

#include <src/simd_stl/numeric/SimdElementWise.h>
#include <simd_stl/numeric/BasicSimdMask.h>

#include <src/simd_stl/algorithm/AdvanceBytes.h>
#include <src/simd_stl/numeric/SimdBroadcast.h>

__SIMD_STL_NUMERIC_NAMESPACE_BEGIN

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_>
class __simd_memory_access;

template <
    arch::CpuFeature	_SimdGeneration_,
    class				_RegisterPolicy_,
    typename            _DesiredType_,
    typename            _VectorType_>
simd_stl_always_inline _DesiredType_* __simd_compress_store_unaligned(
    _DesiredType_*                          __address,
    type_traits::__deduce_simd_mask_type<_SimdGeneration_,
        _DesiredType_, _RegisterPolicy_>    __mask,
    _VectorType_                            __vector) noexcept;

template <
    arch::CpuFeature	_SimdGeneration_,
    class				_RegisterPolicy_,
    typename            _DesiredType_,
    typename            _VectorType_>
simd_stl_always_inline _DesiredType_* __simd_compress_store_aligned(
    _DesiredType_*                          __address,
    type_traits::__deduce_simd_mask_type<_SimdGeneration_,
        _DesiredType_, _RegisterPolicy_>    __mask,
    const _VectorType_                      __vector) noexcept;

#pragma region Sse2-Sse4.2 memory access 

template <>
class __simd_memory_access<arch::CpuFeature::SSE2, numeric::xmm128> {
    static constexpr auto __generation   = arch::CpuFeature::SSE2;
    using __register_policy = numeric::xmm128;

    template <class _DesiredType_>
    using __simd_mask_type = type_traits::__deduce_simd_mask_type<__generation, _DesiredType_, __register_policy>;
public:
    template <sizetype _TypeSize_>
    static constexpr auto __native_mask_load_supported = false;

    template <sizetype _TypeSize_>
    static constexpr auto __native_mask_store_supported = false;

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ __load_upper_half(const void* __address) noexcept;

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ __load_lower_half(const void* __address) noexcept;

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ __non_temporal_load(const void* __address) noexcept;

    template <typename _VectorType_>
    static simd_stl_always_inline void __non_temporal_store(
        void*           __address,
        _VectorType_    __vector) noexcept;

    static simd_stl_always_inline void __streaming_fence() noexcept;

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ __load_unaligned(const void* __address) noexcept;

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ __load_aligned(const void* __address) noexcept;

    template <typename _VectorType_>
    static simd_stl_always_inline void __store_upper_half(
        void*           __address,
        _VectorType_    __vector) noexcept;

    template <typename _VectorType_>
    static simd_stl_always_inline void __store_lower_half(
        void*           __address,
        _VectorType_    __vector) noexcept;

    template <typename _VectorType_>
    static simd_stl_always_inline void __store_unaligned(
        void*           __address,
        _VectorType_    __vector) noexcept;

    template <typename _VectorType_>
    static simd_stl_always_inline void __store_aligned(
        void*           __address,
        _VectorType_    __vector) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline void __mask_store_unaligned(
        void*                                   __address,
        const __simd_mask_type<_DesiredType_>   __mask,
        const _VectorType_                      __vector) noexcept;
    
    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline void __mask_store_aligned(
        void*                                   __address,
        const __simd_mask_type<_DesiredType_>   __mask,
        const _VectorType_                      __vector) noexcept;

    template <
        typename _DesiredType_,
        typename _MaskVectorType_,
        typename _VectorType_,
        std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int> = 0>
    static simd_stl_always_inline void __mask_store_unaligned(
        void*                   __address,
        const _MaskVectorType_  __mask,
        const _VectorType_      __vector) noexcept;

    template <
        typename _DesiredType_,
        typename _MaskVectorType_,
        typename _VectorType_,
        std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int> = 0>
    static simd_stl_always_inline void __mask_store_aligned(
        void*                   __address,
        const _MaskVectorType_  __mask,
        const _VectorType_      __vector) noexcept;

    template <
        typename _VectorType_,
        typename _DesiredType_>
    static simd_stl_always_inline _VectorType_ __mask_load_unaligned(
        const void* __address,
        const __simd_mask_type<_DesiredType_>    __mask) noexcept;

    template <
        typename _VectorType_,
        typename _DesiredType_>
    static simd_stl_always_inline _VectorType_ __mask_load_aligned(
        const void*                             __address,
        const __simd_mask_type<_DesiredType_>   __mask) noexcept;

    template <
        typename _VectorType_,
        typename _DesiredType_,
        typename _MaskVectorType_,
        std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int> = 0>
    static simd_stl_always_inline _VectorType_ __mask_load_unaligned(
        const void*             __address,
        const _MaskVectorType_  __mask) noexcept;

    template <
        typename _VectorType_,
        typename _DesiredType_,
        typename _MaskVectorType_,
        std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int> = 0>
    static simd_stl_always_inline _VectorType_ __mask_load_aligned(
        const void*             __address,
        const _MaskVectorType_  __mask) noexcept;

    template <
        typename _VectorType_,
        typename _DesiredType_>
    static simd_stl_always_inline _VectorType_ __mask_load_unaligned(
        const void*                             __address,
        const __simd_mask_type<_DesiredType_>   __mask,
        _VectorType_                            __additional_source) noexcept;

    template <
        typename _VectorType_,
        typename _DesiredType_>
    static simd_stl_always_inline _VectorType_ __mask_load_aligned(
        const void*                             __address,
        const __simd_mask_type<_DesiredType_>   __mask,
        _VectorType_                            __additional_source) noexcept;

    template <
        typename _VectorType_,
        typename _DesiredType_,
        typename _MaskVectorType_,
        std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int> = 0>
    static simd_stl_always_inline _VectorType_ __mask_load_unaligned(
        const void*             __address,
        const _MaskVectorType_  __mask,
        _VectorType_            __additional_source) noexcept;

    template <
        typename _VectorType_,
        typename _DesiredType_,
        typename _MaskVectorType_,
        std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int> = 0>
    static simd_stl_always_inline _VectorType_ __mask_load_aligned(
        const void*             __address,
        const _MaskVectorType_  __mask,
        _VectorType_            __additional_source) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _DesiredType_* __compress_store_unaligned(
        _DesiredType_*                  __address,
        __simd_mask_type<_DesiredType_> __mask,
        _VectorType_                    __vector) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _DesiredType_* __compress_store_aligned(
        _DesiredType_*                  __address,
        __simd_mask_type<_DesiredType_> __mask,
        _VectorType_                    __vector) noexcept;

    template <typename _Type_>
    static simd_stl_always_inline __m128i __make_tail_mask(uint32 __bytes) noexcept;
};

template <>
class __simd_memory_access<arch::CpuFeature::SSE3, numeric::xmm128> :
    public __simd_memory_access<arch::CpuFeature::SSE2, numeric::xmm128>
{
public:
    template <sizetype _TypeSize_>
    static constexpr auto __native_mask_load_supported = false;

    template <sizetype _TypeSize_>
    static constexpr auto __native_mask_store_supported = false;
};

template <>
class __simd_memory_access<arch::CpuFeature::SSSE3, numeric::xmm128> :
    public __simd_memory_access<arch::CpuFeature::SSE3, numeric::xmm128>
{
    static constexpr auto __generation   = arch::CpuFeature::SSSE3;
    using __register_policy = numeric::xmm128;

    template <class _DesiredType_>
    using __simd_mask_type = type_traits::__deduce_simd_mask_type<__generation, _DesiredType_, __register_policy>;
public:
    template <sizetype _TypeSize_>
    static constexpr auto __native_mask_load_supported = false;

    template <sizetype _TypeSize_>
    static constexpr auto __native_mask_store_supported = false;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _DesiredType_* __compress_store_lower_half(
        _DesiredType_*                          __address,
        const __simd_mask_type<_DesiredType_>   __mask,
        const _VectorType_                      __vector) noexcept;
    
    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _DesiredType_* __compress_store_upper_half(
        _DesiredType_*                          __address,
        const __simd_mask_type<_DesiredType_>   __mask,
        const _VectorType_                      __vector) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _DesiredType_* __compress_store_unaligned(
        _DesiredType_*                          __address,
        const __simd_mask_type<_DesiredType_>   __mask,
        const _VectorType_                      __vector) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _DesiredType_* __compress_store_aligned(
        _DesiredType_*                          __address,
        const __simd_mask_type<_DesiredType_>   __mask,
        const _VectorType_                      __vector) noexcept;
};

template <>
class __simd_memory_access<arch::CpuFeature::SSE41, numeric::xmm128> :
    public __simd_memory_access<arch::CpuFeature::SSSE3, numeric::xmm128>
{    
    static constexpr auto __generation   = arch::CpuFeature::SSE41;
    using __register_policy = numeric::xmm128;

    template <class _DesiredType_>
    using __simd_mask_type = type_traits::__deduce_simd_mask_type<__generation, _DesiredType_, __register_policy>;
public:
    template <sizetype _TypeSize_>
    static constexpr auto __native_mask_load_supported = false;

    template <sizetype _TypeSize_>
    static constexpr auto __native_mask_store_supported = false;

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ __non_temporal_load(const void* __address) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline void __mask_store_unaligned(
        void*                                   __address,
        const __simd_mask_type<_DesiredType_>   __mask,
        const _VectorType_                      __vector) noexcept;
    
    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline void __mask_store_aligned(
        void*                                   __address,
        const __simd_mask_type<_DesiredType_>   __mask,
        const _VectorType_                      __vector) noexcept;

    template <
        typename _DesiredType_,
        typename _MaskVectorType_,
        typename _VectorType_,
        std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int> = 0>
    static simd_stl_always_inline void __mask_store_unaligned(
        void*                   __address,
        const _MaskVectorType_  __mask,
        const _VectorType_      __vector) noexcept;
    
    template <
        typename _DesiredType_,
        typename _MaskVectorType_,
        typename _VectorType_,
        std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int> = 0>
    static simd_stl_always_inline void __mask_store_aligned(
        void*                   __address,
        const _MaskVectorType_  __mask,
        const _VectorType_      __vector) noexcept;

    template <
        typename _VectorType_,
        typename _DesiredType_>
    static simd_stl_always_inline _VectorType_ __mask_load_unaligned(
        const void*                             __address,
        const __simd_mask_type<_DesiredType_>   __mask) noexcept;

    template <
        typename _VectorType_,
        typename _DesiredType_>
    static simd_stl_always_inline _VectorType_ __mask_load_aligned(
        const void*                             __address,
        const __simd_mask_type<_DesiredType_>   __mask) noexcept;

    template <
        typename _VectorType_,
        typename _DesiredType_,
        typename _MaskVectorType_,
        std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int> = 0>
    static simd_stl_always_inline _VectorType_ __mask_load_unaligned(
        const void*             __address,
        const _MaskVectorType_  __mask) noexcept;

    template <
        typename _VectorType_,
        typename _DesiredType_,
        typename _MaskVectorType_,
        std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int> = 0>
    static simd_stl_always_inline _VectorType_ __mask_load_aligned(
        const void*             __address,
        const _MaskVectorType_  __mask) noexcept;

    template <
        typename _VectorType_,
        typename _DesiredType_>
    static simd_stl_always_inline _VectorType_ __mask_load_unaligned(
        const void*                             __address,
        const __simd_mask_type<_DesiredType_>   __mask,
        _VectorType_                            __additional_source) noexcept;

    template <
        typename _VectorType_,
        typename _DesiredType_>
    static simd_stl_always_inline _VectorType_ __mask_load_aligned(
        const void*                             __address,
        const __simd_mask_type<_DesiredType_>    __mask,
        _VectorType_                            __additional_source) noexcept;

    template <
        typename _VectorType_,
        typename _DesiredType_,
        typename _MaskVectorType_,
        std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int> = 0>
    static simd_stl_always_inline _VectorType_ __mask_load_unaligned(
        const void*             __address,
        const _MaskVectorType_  __mask,
        _VectorType_            __additional_source) noexcept;

    template <
        typename _VectorType_,
        typename _DesiredType_,
        typename _MaskVectorType_,
        std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int> = 0>
    static simd_stl_always_inline _VectorType_ __mask_load_aligned(
        const void*             __address,
        const _MaskVectorType_  __mask,
        _VectorType_            __additional_source) noexcept;
};

template <>
class __simd_memory_access<arch::CpuFeature::SSE42, numeric::xmm128> :
    public __simd_memory_access<arch::CpuFeature::SSE41, numeric::xmm128>
{
};

#pragma endregion

#pragma region Avx-Avx2 memory access

template <>
class __simd_memory_access<arch::CpuFeature::AVX, numeric::ymm256>
{
    static constexpr auto __generation   = arch::CpuFeature::AVX;
    using __register_policy = numeric::ymm256;

    template <class _DesiredType_>
    using __simd_mask_type = type_traits::__deduce_simd_mask_type<__generation, _DesiredType_, __register_policy>;

    template <sizetype _TypeSize_>
    struct __native_mask_load_support:
        std::bool_constant<false> 
    {};

    template <>
    struct __native_mask_load_support<4>:
        std::bool_constant<true> 
    {}; 

    template <>
    struct __native_mask_load_support<8>:
        std::bool_constant<true>
    {}; 
public:
    template <sizetype _TypeSize_>
    static constexpr auto __native_mask_load_supported = __native_mask_load_support<_TypeSize_>::value;

    template <sizetype _TypeSize_>
    static constexpr auto __native_mask_store_supported = __native_mask_load_support<_TypeSize_>::value;

    template <typename _Type_>
    static simd_stl_always_inline auto __make_tail_mask(uint32 __bytes) noexcept;

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ __load_upper_half(const void* __address) noexcept;

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ __load_lower_half(const void* __address) noexcept;

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ __non_temporal_load(const void* __address) noexcept;

    template <typename _VectorType_>
    static simd_stl_always_inline void __non_temporal_store(
        void*           __address,
        _VectorType_    __vector) noexcept;

    static simd_stl_always_inline void __streaming_fence() noexcept;

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ __load_unaligned(const void* __address) noexcept;

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ __load_aligned(const void* __address) noexcept;

    template <typename _VectorType_>
    static simd_stl_always_inline void __store_upper_half(
        void*           __address,
        _VectorType_    __vector) noexcept;

    template <typename _VectorType_>
    static simd_stl_always_inline void __store_lower_half(
        void*           __address,
        _VectorType_    __vector) noexcept;

    template <typename _VectorType_>
    static simd_stl_always_inline void __store_unaligned(
        void*           __address,
        _VectorType_    __vector) noexcept;

    template <typename _VectorType_>
    static simd_stl_always_inline void __store_aligned(
        void*           __address,
        _VectorType_    __vector) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline void __mask_store_unaligned(
        void*                                   __address,
        const __simd_mask_type<_DesiredType_>   __mask,
        const _VectorType_                      __vector) noexcept;
    
    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline void __mask_store_aligned(
        void*                                   __address,
        const __simd_mask_type<_DesiredType_>   __mask,
        const _VectorType_                      __vector) noexcept;

     template <
        typename _DesiredType_,
        typename _MaskVectorType_,
        typename _VectorType_,
        std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int> = 0>
    static simd_stl_always_inline void __mask_store_unaligned(
        void*                   __address,
        const _MaskVectorType_  __mask,
        const _VectorType_      __vector) noexcept;
    
    template <
        typename _DesiredType_,
        typename _MaskVectorType_,
        typename _VectorType_,
        std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int> = 0>
    static simd_stl_always_inline void __mask_store_aligned(
        void*                   __address,
        const _MaskVectorType_  __mask,
        const _VectorType_      __vector) noexcept;

    template <
        typename _VectorType_,
        typename _DesiredType_>
    static simd_stl_always_inline _VectorType_ __mask_load_unaligned(
        const void*                             __address,
        const __simd_mask_type<_DesiredType_>   __mask) noexcept;

    template <
        typename _VectorType_,
        typename _DesiredType_>
    static simd_stl_always_inline _VectorType_ __mask_load_aligned(
        const void*                             __address,
        const __simd_mask_type<_DesiredType_>   __mask) noexcept;

    template <
        typename _VectorType_,
        typename _DesiredType_,
        typename _MaskVectorType_,
        std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int> = 0>
    static simd_stl_always_inline _VectorType_ __mask_load_unaligned(
        const void*             __address,
        const _MaskVectorType_  __mask) noexcept;

    template <
        typename _VectorType_,
        typename _DesiredType_,
        typename _MaskVectorType_,
        std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int> = 0>
    static simd_stl_always_inline _VectorType_ __mask_load_aligned(
        const void*             __address,
        const _MaskVectorType_  __mask) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _DesiredType_* __compress_store_lower_half(
        _DesiredType_*                  __address,
        __simd_mask_type<_DesiredType_> __mask,
        _VectorType_                    __vector) noexcept;
    
    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _DesiredType_* __compress_store_upper_half(
        _DesiredType_*                  __address,
        __simd_mask_type<_DesiredType_> __mask,
        _VectorType_                    __vector) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _DesiredType_* __compress_store_unaligned(
        _DesiredType_*                      __address,
        __simd_mask_type<_DesiredType_>     __mask,
        _VectorType_                        __vector) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _DesiredType_* __compress_store_aligned(
        _DesiredType_*                      __address,
        __simd_mask_type<_DesiredType_>     __mask,
        _VectorType_                        __vector) noexcept;
};

template <>
class __simd_memory_access<arch::CpuFeature::AVX2, numeric::ymm256>:
    public __simd_memory_access<arch::CpuFeature::AVX, numeric::ymm256>
{
    static constexpr auto __generation   = arch::CpuFeature::AVX2;
    using __register_policy = numeric::ymm256;

    template <class _DesiredType_>
    using __simd_mask_type = type_traits::__deduce_simd_mask_type<__generation, _DesiredType_, __register_policy>;
    
    template <sizetype _TypeSize_>
    struct __native_mask_load_support :
        std::bool_constant<false>
    {};

    template <>
    struct __native_mask_load_support<4> :
        std::bool_constant<true>
    {};

    template <>
    struct __native_mask_load_support<8> :
        std::bool_constant<true>
    {};
public:
    template <sizetype _TypeSize_>
    static constexpr auto __native_mask_load_supported = __native_mask_load_support<_TypeSize_>::value;

    template <sizetype _TypeSize_>
    static constexpr auto __native_mask_store_supported = __native_mask_load_support<_TypeSize_>::value;

    template <
        typename _VectorType_,
        typename _DesiredType_>
    static simd_stl_always_inline _VectorType_ __mask_load_unaligned(
        const void*                             __address,
        const __simd_mask_type<_DesiredType_>   __mask) noexcept;

    template <
        typename _VectorType_,
        typename _DesiredType_>
    static simd_stl_always_inline _VectorType_ __mask_load_aligned(
        const void*                             __address,
        const __simd_mask_type<_DesiredType_>   __mask) noexcept;

    template <
        typename _VectorType_,
        typename _DesiredType_,
        typename _MaskVectorType_,
        std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int> = 0>
    static simd_stl_always_inline _VectorType_ __mask_load_unaligned(
        const void*             __address,
        const _MaskVectorType_  __mask) noexcept;

    template <
        typename _VectorType_,
        typename _DesiredType_,
        typename _MaskVectorType_,
        std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int> = 0>
    static simd_stl_always_inline _VectorType_ __mask_load_aligned(
        const void*             __address,
        const _MaskVectorType_  __mask) noexcept;

    template <
        typename _VectorType_,
        typename _DesiredType_>
    static simd_stl_always_inline _VectorType_ __mask_load_unaligned(
        const void*                             __address,
        const __simd_mask_type<_DesiredType_>   __mask,
        _VectorType_                            __additional_source) noexcept;

    template <
        typename _VectorType_,
        typename _DesiredType_>
    static simd_stl_always_inline _VectorType_ __mask_load_aligned(
        const void*                             __address,
        const __simd_mask_type<_DesiredType_>   __mask,
        _VectorType_                            __additional_source) noexcept;

    template <
        typename _VectorType_,
        typename _DesiredType_,
        typename _MaskVectorType_,
        std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int> = 0>
    static simd_stl_always_inline _VectorType_ __mask_load_unaligned(
        const void*             __address,
        const _MaskVectorType_  __mask,
        _VectorType_            __additional_source) noexcept;

    template <
        typename _VectorType_,
        typename _DesiredType_,
        typename _MaskVectorType_,
        std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int> = 0>
    static simd_stl_always_inline _VectorType_ __mask_load_aligned(
        const void*             __address,
        const _MaskVectorType_  __mask,
        _VectorType_            __additional_source) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline void __mask_store_unaligned(
        void*                                   __address,
        const __simd_mask_type<_DesiredType_>   __mask,
        const _VectorType_                      __vector) noexcept;
    
    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline void __mask_store_aligned(
        void*                                   __address,
        const __simd_mask_type<_DesiredType_>   __mask,
        const _VectorType_                      __vector) noexcept;

    template <
        typename _DesiredType_,
        typename _MaskVectorType_,
        typename _VectorType_,
        std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int> = 0>
    static simd_stl_always_inline void __mask_store_unaligned(
        void*                   __address,
        const _MaskVectorType_  __mask,
        const _VectorType_      __vector) noexcept;
    
    template <
        typename _DesiredType_,
        typename _MaskVectorType_,
        typename _VectorType_,
        std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int> = 0>
    static simd_stl_always_inline void __mask_store_aligned(
        void*                   __address,
        const _MaskVectorType_  __mask,
        const _VectorType_      __vector) noexcept;

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ __non_temporal_load(const void* __address) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _DesiredType_* __compress_store_unaligned(
        _DesiredType_*                      __address,
        __simd_mask_type<_DesiredType_>     __mask,
        _VectorType_                        __vector) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _DesiredType_* __compress_store_aligned(
        _DesiredType_*                      __address,
        __simd_mask_type<_DesiredType_>     __mask,
        _VectorType_                        __vector) noexcept;
};

#pragma endregion

#pragma region Avx512 memory access

template <>
class __simd_memory_access<arch::CpuFeature::AVX512F, zmm512>
{
    static constexpr auto __generation   = arch::CpuFeature::AVX512F;
    using __register_policy = zmm512;

    template <class _DesiredType_>
    using __simd_mask_type = type_traits::__deduce_simd_mask_type<__generation, _DesiredType_, __register_policy>;

    template <sizetype _TypeSize_>
    struct __native_mask_load_support :
        std::bool_constant<false>
    {};

    template <>
    struct __native_mask_load_support<4> :
        std::bool_constant<true>
    {};

    template <>
    struct __native_mask_load_support<8> :
        std::bool_constant<true>
    {};
public:
    template <sizetype _TypeSize_>
    static constexpr auto __native_mask_load_supported = __native_mask_load_support<_TypeSize_>::value;

    template <sizetype _TypeSize_>
    static constexpr auto __native_mask_store_supported = __native_mask_load_support<_TypeSize_>::value;

    template <typename _Type_>
    static simd_stl_always_inline auto __make_tail_mask(uint32 __bytes) noexcept;

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ __load_upper_half(const void* __address) noexcept;

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ __load_lower_half(const void* __address) noexcept;

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ __non_temporal_load(const void* __address) noexcept;

    template <typename _VectorType_>
    static simd_stl_always_inline void __non_temporal_store(
        void*           __address,
        _VectorType_    __vector) noexcept;

    static simd_stl_always_inline void __streaming_fence() noexcept;

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ __load_unaligned(const void* __address) noexcept;

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ __load_aligned(const void* __address) noexcept;

    template <typename _VectorType_>
    static simd_stl_always_inline void __store_upper_half(
        void*           __address,
        _VectorType_    __vector) noexcept;

    template <typename _VectorType_>
    static simd_stl_always_inline void __store_lower_half(
        void*           __address,
        _VectorType_    __vector) noexcept;

    template <typename _VectorType_>
    static simd_stl_always_inline void __store_unaligned(
        void*           __address,
        _VectorType_    __vector) noexcept;

    template <typename _VectorType_>
    static simd_stl_always_inline void __store_aligned(
        void*           __address,
        _VectorType_    __vector) noexcept;

     template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline void __mask_store_unaligned(
        void*                                   __address,
        const __simd_mask_type<_DesiredType_>   __mask,
        const _VectorType_                      __vector) noexcept;
    
    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline void __mask_store_aligned(
        void*                                   __address,
        const __simd_mask_type<_DesiredType_>   __mask,
        const _VectorType_                      __vector) noexcept;

    template <
        typename _DesiredType_,
        typename _MaskVectorType_,
        typename _VectorType_,
        std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int> = 0>
    static simd_stl_always_inline void __mask_store_unaligned(
        void*                   __address,
        const _MaskVectorType_  __mask,
        const _VectorType_      __vector) noexcept;
    
    template <
        typename _DesiredType_,
        typename _MaskVectorType_,
        typename _VectorType_,
        std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int> = 0>
    static simd_stl_always_inline void __mask_store_aligned(
        void*                   __address,
        const _MaskVectorType_  __mask,
        const _VectorType_      __vector) noexcept;

    template <
        typename _VectorType_,
        typename _DesiredType_>
    static simd_stl_always_inline _VectorType_ __mask_load_unaligned(
        const void*                             __address,
        const __simd_mask_type<_DesiredType_>   __mask) noexcept;

    template <
        typename _VectorType_,
        typename _DesiredType_>
    static simd_stl_always_inline _VectorType_ __mask_load_aligned(
        const void*                             __address,
        const __simd_mask_type<_DesiredType_>   __mask) noexcept;

    template <
        typename _VectorType_,
        typename _DesiredType_,
        typename _MaskVectorType_,
        std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int> = 0>
    static simd_stl_always_inline _VectorType_ __mask_load_unaligned(
        const void*             __address,
        const _MaskVectorType_  __mask) noexcept;

    template <
        typename _VectorType_,
        typename _DesiredType_,
        typename _MaskVectorType_,
        std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int> = 0>
    static simd_stl_always_inline _VectorType_ __mask_load_aligned(
        const void*             __address,
        const _MaskVectorType_  __mask) noexcept;

    template <
        typename _VectorType_,
        typename _DesiredType_>
    static simd_stl_always_inline _VectorType_ __mask_load_unaligned(
        const void*                             __address,
        const __simd_mask_type<_DesiredType_>   __mask,
        _VectorType_                            __additional_source) noexcept;

    template <
        typename _VectorType_,
        typename _DesiredType_>
    static simd_stl_always_inline _VectorType_ __mask_load_aligned(
        const void*                             __address,
        const __simd_mask_type<_DesiredType_>   __mask,
        _VectorType_                            __additional_source) noexcept;

    template <
        typename _VectorType_,
        typename _DesiredType_,
        typename _MaskVectorType_,
        std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int> = 0>
    static simd_stl_always_inline _VectorType_ __mask_load_unaligned(
        const void*             __address,
        const _MaskVectorType_  __mask,
        _VectorType_            __additional_source) noexcept;

    template <
        typename _VectorType_,
        typename _DesiredType_,
        typename _MaskVectorType_,
        std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int> = 0>
    static simd_stl_always_inline _VectorType_ __mask_load_aligned(
        const void*             __address,
        const _MaskVectorType_  __mask,
        _VectorType_            __additional_source) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _DesiredType_* __compress_store_unaligned(
        _DesiredType_*                      __address,
        __simd_mask_type<_DesiredType_>     __mask,
        _VectorType_                        __vector) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _DesiredType_* __compress_store_aligned(
        _DesiredType_*                      __address,
        __simd_mask_type<_DesiredType_>     __mask,
        _VectorType_                        __vector) noexcept;
};

template <>
class __simd_memory_access<arch::CpuFeature::AVX512BW, zmm512>:
    public __simd_memory_access<arch::CpuFeature::AVX512F, zmm512>
{
    static constexpr auto __generation = arch::CpuFeature::AVX512BW;
    using __register_policy = zmm512;

    template <class _DesiredType_>
    using __simd_mask_type = type_traits::__deduce_simd_mask_type<__generation, _DesiredType_, __register_policy>;
public:
    template <sizetype _TypeSize_>
    static constexpr auto __native_mask_load_supported = true;

    template <sizetype _TypeSize_>
    static constexpr auto __native_mask_store_supported = true;

    template <typename _Type_>
    static simd_stl_always_inline auto __make_tail_mask(uint32 __bytes) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline void __mask_store_unaligned(
        void*                                   __address,
        const __simd_mask_type<_DesiredType_>   __mask,
        const _VectorType_                      __vector) noexcept;
    
    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline void __mask_store_aligned(
        void*                                   __address,
        const __simd_mask_type<_DesiredType_>   __mask,
        const _VectorType_                      __vector) noexcept;

    template <
        typename _DesiredType_,
        typename _MaskVectorType_,
        typename _VectorType_,
        std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int> = 0>
    static simd_stl_always_inline void __mask_store_unaligned(
        void*                   __address,
        const _MaskVectorType_  __mask,
        const _VectorType_      __vector) noexcept;
    
    template <
        typename _DesiredType_,
        typename _MaskVectorType_,
        typename _VectorType_,
        std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int> = 0>
    static simd_stl_always_inline void __mask_store_aligned(
        void*                   __address,
        const _MaskVectorType_  __mask,
        const _VectorType_      __vector) noexcept;

    template <
        typename _VectorType_,
        typename _DesiredType_>
    static simd_stl_always_inline _VectorType_ __mask_load_unaligned(
        const void*                             __address,
        const __simd_mask_type<_DesiredType_>   __mask) noexcept;

    template <
        typename _VectorType_,
        typename _DesiredType_>
    static simd_stl_always_inline _VectorType_ __mask_load_aligned(
        const void*                             __address,
        const __simd_mask_type<_DesiredType_>   __mask) noexcept;

    template <
        typename _VectorType_,
        typename _DesiredType_,
        typename _MaskVectorType_,
        std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int> = 0>
    static simd_stl_always_inline _VectorType_ __mask_load_unaligned(
        const void*             __address,
        const _MaskVectorType_  __mask) noexcept;

    template <
        typename _VectorType_,
        typename _DesiredType_,
        typename _MaskVectorType_,
        std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int> = 0>
    static simd_stl_always_inline _VectorType_ __mask_load_aligned(
        const void*             __address,
        const _MaskVectorType_  __mask) noexcept;

    template <
        typename _VectorType_,
        typename _DesiredType_>
    static simd_stl_always_inline _VectorType_ __mask_load_unaligned(
        const void*                             __address,
        const __simd_mask_type<_DesiredType_>   __mask,
        _VectorType_                            __additional_source) noexcept;

    template <
        typename _VectorType_,
        typename _DesiredType_>
    static simd_stl_always_inline _VectorType_ __mask_load_aligned(
        const void*                             __address,
        const __simd_mask_type<_DesiredType_>   __mask,
        _VectorType_                            __additional_source) noexcept;

    template <
        typename _VectorType_,
        typename _DesiredType_,
        typename _MaskVectorType_,
        std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int> = 0>
    static simd_stl_always_inline _VectorType_ __mask_load_unaligned(
        const void*             __address,
        const _MaskVectorType_  __mask,
        _VectorType_            __additional_source) noexcept;

    template <
        typename _VectorType_,
        typename _DesiredType_,
        typename _MaskVectorType_,
        std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int> = 0>
    static simd_stl_always_inline _VectorType_ __mask_load_aligned(
        const void*             __address,
        const _MaskVectorType_  __mask,
        _VectorType_            __additional_source) noexcept;
};

template <>
class __simd_memory_access<arch::CpuFeature::AVX512DQ, zmm512> :
    public __simd_memory_access<arch::CpuFeature::AVX512F, zmm512>
{};


template <>
class __simd_memory_access<arch::CpuFeature::AVX512VLF, ymm256> :
    public __simd_memory_access<arch::CpuFeature::AVX2, ymm256>
{
    static constexpr auto __generation   = arch::CpuFeature::AVX512VLF;
    using __register_policy = ymm256;

    template <class _DesiredType_>
    using __simd_mask_type = type_traits::__deduce_simd_mask_type<__generation, _DesiredType_, __register_policy>;

    template <sizetype _TypeSize_>
    struct __native_mask_load_support :
        std::bool_constant<false>
    {};

    template <>
    struct __native_mask_load_support<4> :
        std::bool_constant<true>
    {};

    template <>
    struct __native_mask_load_support<8> :
        std::bool_constant<true>
    {};
public:
    template <sizetype _TypeSize_>
    static constexpr auto __native_mask_load_supported = __native_mask_load_support<_TypeSize_>::value;

    template <sizetype _TypeSize_>
    static constexpr auto __native_mask_store_supported = __native_mask_load_support<_TypeSize_>::value;

    template <typename _Type_>
    static simd_stl_always_inline auto __make_tail_mask(uint32 __bytes) noexcept;

     template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline void __mask_store_unaligned(
        void*                                   __address,
        const __simd_mask_type<_DesiredType_>   __mask,
        const _VectorType_                      __vector) noexcept;
    
    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline void __mask_store_aligned(
        void*                                   __address,
        const __simd_mask_type<_DesiredType_>   __mask,
        const _VectorType_                      __vector) noexcept;

    template <
        typename _DesiredType_,
        typename _MaskVectorType_,
        typename _VectorType_,
        std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int> = 0>
    static simd_stl_always_inline void __mask_store_unaligned(
        void*                   __address,
        const _MaskVectorType_  __mask,
        const _VectorType_      __vector) noexcept;
    
    template <
        typename _DesiredType_,
        typename _MaskVectorType_,
        typename _VectorType_,
        std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int> = 0>
    static simd_stl_always_inline void __mask_store_aligned(
        void*                   __address,
        const _MaskVectorType_  __mask,
        const _VectorType_      __vector) noexcept;

    template <
        typename _VectorType_,
        typename _DesiredType_>
    static simd_stl_always_inline _VectorType_ __mask_load_unaligned(
        const void*                             __address,
        const __simd_mask_type<_DesiredType_>   __mask) noexcept;

    template <
        typename _VectorType_,
        typename _DesiredType_>
    static simd_stl_always_inline _VectorType_ __mask_load_aligned(
        const void*                             __address,
        const __simd_mask_type<_DesiredType_>   __mask) noexcept;

    template <
        typename _VectorType_,
        typename _DesiredType_,
        typename _MaskVectorType_,
        std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int> = 0>
    static simd_stl_always_inline _VectorType_ __mask_load_unaligned(
        const void*             __address,
        const _MaskVectorType_  __mask) noexcept;

    template <
        typename _VectorType_,
        typename _DesiredType_,
        typename _MaskVectorType_,
        std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int> = 0>
    static simd_stl_always_inline _VectorType_ __mask_load_aligned(
        const void*             __address,
        const _MaskVectorType_  __mask) noexcept;

    template <
        typename _VectorType_,
        typename _DesiredType_>
    static simd_stl_always_inline _VectorType_ __mask_load_unaligned(
        const void*                             __address,
        const __simd_mask_type<_DesiredType_>   __mask,
        _VectorType_                            __additional_source) noexcept;

    template <
        typename _VectorType_,
        typename _DesiredType_>
    static simd_stl_always_inline _VectorType_ __mask_load_aligned(
        const void*                             __address,
        const __simd_mask_type<_DesiredType_>   __mask,
        _VectorType_                            __additional_source) noexcept;

    template <
        typename _VectorType_,
        typename _DesiredType_,
        typename _MaskVectorType_,
        std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int> = 0>
    static simd_stl_always_inline _VectorType_ __mask_load_unaligned(
        const void*             __address,
        const _MaskVectorType_  __mask,
        _VectorType_            __additional_source) noexcept;

    template <
        typename _VectorType_,
        typename _DesiredType_,
        typename _MaskVectorType_,
        std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int> = 0>
    static simd_stl_always_inline _VectorType_ __mask_load_aligned(
        const void*             __address,
        const _MaskVectorType_  __mask,
        _VectorType_            __additional_source) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _DesiredType_* __compress_store_unaligned(
        _DesiredType_*                      __address,
        __simd_mask_type<_DesiredType_>     __mask,
        _VectorType_                        __vector) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _DesiredType_* __compress_store_aligned(
        _DesiredType_*                      __address,
        __simd_mask_type<_DesiredType_>     __mask,
        _VectorType_                        __vector) noexcept;
};

template <>
class __simd_memory_access<arch::CpuFeature::AVX512VLBW, ymm256> :
    public __simd_memory_access<arch::CpuFeature::AVX512VLF, ymm256>
{
    static constexpr auto __generation   = arch::CpuFeature::AVX512VLBW;
    using __register_policy = ymm256;

    template <class _DesiredType_>
    using __simd_mask_type = type_traits::__deduce_simd_mask_type<__generation, _DesiredType_, __register_policy>;
public:
    template <sizetype _TypeSize_>
    static constexpr auto __native_mask_load_supported = true;

    template <sizetype _TypeSize_>
    static constexpr auto __native_mask_store_supported = true;

    template <typename _Type_>
    static simd_stl_always_inline auto __make_tail_mask(uint32 __bytes) noexcept;

     template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline void __mask_store_unaligned(
        void*                                   __address,
        const __simd_mask_type<_DesiredType_>   __mask,
        const _VectorType_                      __vector) noexcept;
    
    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline void __mask_store_aligned(
        void*                                   __address,
        const __simd_mask_type<_DesiredType_>   __mask,
        const _VectorType_                      __vector) noexcept;

    template <
        typename _DesiredType_,
        typename _MaskVectorType_,
        typename _VectorType_,
        std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int> = 0>
    static simd_stl_always_inline void __mask_store_unaligned(
        void*                   __address,
        const _MaskVectorType_  __mask,
        const _VectorType_      __vector) noexcept;
    
    template <
        typename _DesiredType_,
        typename _MaskVectorType_,
        typename _VectorType_,
        std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int> = 0>
    static simd_stl_always_inline void __mask_store_aligned(
        void*                   __address,
        const _MaskVectorType_  __mask,
        const _VectorType_      __vector) noexcept;

    template <
        typename _VectorType_,
        typename _DesiredType_>
    static simd_stl_always_inline _VectorType_ __mask_load_unaligned(
        const void*                             __address,
        const __simd_mask_type<_DesiredType_>   __mask) noexcept;

    template <
        typename _VectorType_,
        typename _DesiredType_>
    static simd_stl_always_inline _VectorType_ __mask_load_aligned(
        const void*                             __address,
        const __simd_mask_type<_DesiredType_>   __mask) noexcept;

    template <
        typename _VectorType_,
        typename _DesiredType_,
        typename _MaskVectorType_,
        std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int> = 0>
    static simd_stl_always_inline _VectorType_ __mask_load_unaligned(
        const void*             __address,
        const _MaskVectorType_  __mask) noexcept;

    template <
        typename _VectorType_,
        typename _DesiredType_,
        typename _MaskVectorType_,
        std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int> = 0>
    static simd_stl_always_inline _VectorType_ __mask_load_aligned(
        const void*             __address,
        const _MaskVectorType_  __mask) noexcept;

    template <
        typename _VectorType_,
        typename _DesiredType_>
    static simd_stl_always_inline _VectorType_ __mask_load_unaligned(
        const void*                             __address,
        const __simd_mask_type<_DesiredType_>   __mask,
        _VectorType_                            __additional_source) noexcept;

    template <
        typename _VectorType_,
        typename _DesiredType_>
    static simd_stl_always_inline _VectorType_ __mask_load_aligned(
        const void*                             __address,
        const __simd_mask_type<_DesiredType_>   __mask,
        _VectorType_                            __additional_source) noexcept;

    template <
        typename _VectorType_,
        typename _DesiredType_,
        typename _MaskVectorType_,
        std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int> = 0>
    static simd_stl_always_inline _VectorType_ __mask_load_unaligned(
        const void*             __address,
        const _MaskVectorType_  __mask,
        _VectorType_            __additional_source) noexcept;

    template <
        typename _VectorType_,
        typename _DesiredType_,
        typename _MaskVectorType_,
        std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int> = 0>
    static simd_stl_always_inline _VectorType_ __mask_load_aligned(
        const void*             __address,
        const _MaskVectorType_  __mask,
        _VectorType_            __additional_source) noexcept;
};

template <>
class __simd_memory_access<arch::CpuFeature::AVX512VLDQ, ymm256> :
    public __simd_memory_access<arch::CpuFeature::AVX512VLF, ymm256>
{
};

template <>
class __simd_memory_access<arch::CpuFeature::AVX512VLBWDQ, ymm256> :
    public __simd_memory_access<arch::CpuFeature::AVX512VLBW, ymm256>
{
};

template <>
class __simd_memory_access<arch::CpuFeature::AVX512VLF, xmm128> :
    public __simd_memory_access<arch::CpuFeature::SSE42, xmm128>
{
    static constexpr auto __generation   = arch::CpuFeature::AVX512VLF;
    using __register_policy = xmm128;

    template <class _DesiredType_>
    using __simd_mask_type = type_traits::__deduce_simd_mask_type<__generation, _DesiredType_, __register_policy>;

    template <sizetype _TypeSize_>
    struct __native_mask_load_support :
        std::bool_constant<false>
    {};

    template <>
    struct __native_mask_load_support<4> :
        std::bool_constant<true>
    {};

    template <>
    struct __native_mask_load_support<8> :
        std::bool_constant<true>
    {};
public:
    template <sizetype _TypeSize_>
    static constexpr auto __native_mask_load_supported = __native_mask_load_support<_TypeSize_>::value;

    template <sizetype _TypeSize_>
    static constexpr auto __native_mask_store_supported = __native_mask_load_support<_TypeSize_>::value;

    template <typename _Type_>
    static simd_stl_always_inline auto __make_tail_mask(uint32 __bytes) noexcept;

     template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline void __mask_store_unaligned(
        void*                                   __address,
        const __simd_mask_type<_DesiredType_>   __mask,
        const _VectorType_                      __vector) noexcept;
    
    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline void __mask_store_aligned(
        void*                                   __address,
        const __simd_mask_type<_DesiredType_>   __mask,
        const _VectorType_                      __vector) noexcept;

    template <
        typename _DesiredType_,
        typename _MaskVectorType_,
        typename _VectorType_,
        std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int> = 0>
    static simd_stl_always_inline void __mask_store_unaligned(
        void*                   __address,
        const _MaskVectorType_  __mask,
        const _VectorType_      __vector) noexcept;
    
    template <
        typename _DesiredType_,
        typename _MaskVectorType_,
        typename _VectorType_,
        std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int> = 0>
    static simd_stl_always_inline void __mask_store_aligned(
        void*                   __address,
        const _MaskVectorType_  __mask,
        const _VectorType_      __vector) noexcept;

    template <
        typename _VectorType_,
        typename _DesiredType_>
    static simd_stl_always_inline _VectorType_ __mask_load_unaligned(
        const void*                             __address,
        const __simd_mask_type<_DesiredType_>   __mask) noexcept;

    template <
        typename _VectorType_,
        typename _DesiredType_>
    static simd_stl_always_inline _VectorType_ __mask_load_aligned(
        const void*                             __address,
        const __simd_mask_type<_DesiredType_>   __mask) noexcept;

    template <
        typename _VectorType_,
        typename _DesiredType_,
        typename _MaskVectorType_,
        std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int> = 0>
    static simd_stl_always_inline _VectorType_ __mask_load_unaligned(
        const void*             __address,
        const _MaskVectorType_  __mask) noexcept;

    template <
        typename _VectorType_,
        typename _DesiredType_,
        typename _MaskVectorType_,
        std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int> = 0>
    static simd_stl_always_inline _VectorType_ __mask_load_aligned(
        const void*             __address,
        const _MaskVectorType_  __mask) noexcept;

    template <
        typename _VectorType_,
        typename _DesiredType_>
    static simd_stl_always_inline _VectorType_ __mask_load_unaligned(
        const void*                             __address,
        const __simd_mask_type<_DesiredType_>   __mask,
        _VectorType_                            __additional_source) noexcept;

    template <
        typename _VectorType_,
        typename _DesiredType_>
    static simd_stl_always_inline _VectorType_ __mask_load_aligned(
        const void*                             __address,
        const __simd_mask_type<_DesiredType_>   __mask,
        _VectorType_                            __additional_source) noexcept;

    template <
        typename _VectorType_,
        typename _DesiredType_,
        typename _MaskVectorType_,
        std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int> = 0>
    static simd_stl_always_inline _VectorType_ __mask_load_unaligned(
        const void*             __address,
        const _MaskVectorType_  __mask,
        _VectorType_            __additional_source) noexcept;

    template <
        typename _VectorType_,
        typename _DesiredType_,
        typename _MaskVectorType_,
        std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int> = 0>
    static simd_stl_always_inline _VectorType_ __mask_load_aligned(
        const void*             __address,
        const _MaskVectorType_  __mask,
        _VectorType_            __additional_source) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _DesiredType_* __compress_store_unaligned(
        _DesiredType_*                      __address,
        __simd_mask_type<_DesiredType_>     __mask,
        _VectorType_                        __vector) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _DesiredType_* __compress_store_aligned(
        _DesiredType_*                      __address,
        __simd_mask_type<_DesiredType_>     __mask,
        _VectorType_                        __vector) noexcept;
};

template <>
class __simd_memory_access<arch::CpuFeature::AVX512VLBW, xmm128> :
    public __simd_memory_access<arch::CpuFeature::AVX512VLF, xmm128>
{
    static constexpr auto __generation   = arch::CpuFeature::AVX512VLBW;
    using __register_policy               = xmm128;

    template <class _DesiredType_>
    using __simd_mask_type = type_traits::__deduce_simd_mask_type<__generation, _DesiredType_, __register_policy>;
public:
    template <sizetype _TypeSize_>
    static constexpr auto __native_mask_load_supported = true;

    template <sizetype _TypeSize_>
    static constexpr auto __native_mask_store_supported = true;

    template <typename _Type_>
    static simd_stl_always_inline auto __make_tail_mask(uint32 __bytes) noexcept;

     template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline void __mask_store_unaligned(
        void*                                   __address,
        const __simd_mask_type<_DesiredType_>   __mask,
        const _VectorType_                      __vector) noexcept;
    
    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline void __mask_store_aligned(
        void*                                   __address,
        const __simd_mask_type<_DesiredType_>   __mask,
        const _VectorType_                      __vector) noexcept;

    template <
        typename _DesiredType_,
        typename _MaskVectorType_,
        typename _VectorType_,
        std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int> = 0>
    static simd_stl_always_inline void __mask_store_unaligned(
        void*                   __address,
        const _MaskVectorType_  __mask,
        const _VectorType_      __vector) noexcept;
    
    template <
        typename _DesiredType_,
        typename _MaskVectorType_,
        typename _VectorType_,
        std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int> = 0>
    static simd_stl_always_inline void __mask_store_aligned(
        void*                   __address,
        const _MaskVectorType_  __mask,
        const _VectorType_      __vector) noexcept;

    template <
        typename _VectorType_,
        typename _DesiredType_>
    static simd_stl_always_inline _VectorType_ __mask_load_unaligned(
        const void*                             __address,
        const __simd_mask_type<_DesiredType_>   __mask) noexcept;

    template <
        typename _VectorType_,
        typename _DesiredType_>
    static simd_stl_always_inline _VectorType_ __mask_load_aligned(
        const void*                             __address,
        const __simd_mask_type<_DesiredType_>   __mask) noexcept;

    template <
        typename _VectorType_,
        typename _DesiredType_,
        typename _MaskVectorType_,
        std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int> = 0>
    static simd_stl_always_inline _VectorType_ __mask_load_unaligned(
        const void*             __address,
        const _MaskVectorType_  __mask) noexcept;

    template <
        typename _VectorType_,
        typename _DesiredType_,
        typename _MaskVectorType_,
        std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int> = 0>
    static simd_stl_always_inline _VectorType_ __mask_load_aligned(
        const void*             __address,
        const _MaskVectorType_  __mask) noexcept;

    template <
        typename _VectorType_,
        typename _DesiredType_>
    static simd_stl_always_inline _VectorType_ __mask_load_unaligned(
        const void*                             __address,
        const __simd_mask_type<_DesiredType_>   __mask,
        _VectorType_                            __additional_source) noexcept;

    template <
        typename _VectorType_,
        typename _DesiredType_>
    static simd_stl_always_inline _VectorType_ __mask_load_aligned(
        const void*                             __address,
        const __simd_mask_type<_DesiredType_>   __mask,
        _VectorType_                            __additional_source) noexcept;

    template <
        typename _VectorType_,
        typename _DesiredType_,
        typename _MaskVectorType_,
        std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int> = 0>
    static simd_stl_always_inline _VectorType_ __mask_load_unaligned(
        const void*             __address,
        const _MaskVectorType_  __mask,
        _VectorType_            __additional_source) noexcept;

    template <
        typename _VectorType_,
        typename _DesiredType_,
        typename _MaskVectorType_,
        std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int> = 0>
    static simd_stl_always_inline _VectorType_ __mask_load_aligned(
        const void*             __address,
        const _MaskVectorType_  __mask,
        _VectorType_            __additional_source) noexcept;
};

template <>
class __simd_memory_access<arch::CpuFeature::AVX512VLDQ, xmm128> :
    public __simd_memory_access<arch::CpuFeature::AVX512VLF, xmm128>
{
};

template <>
class __simd_memory_access<arch::CpuFeature::AVX512VLBWDQ, xmm128> :
    public __simd_memory_access<arch::CpuFeature::AVX512VLBW, xmm128>
{
};


#pragma endregion

template <
    arch::CpuFeature	_SimdGeneration_,
    class				_RegisterPolicy_,
    class				_VectorType_>
simd_stl_nodiscard simd_stl_always_inline _VectorType_ __simd_load_unaligned(const void* __address) noexcept {
    __verify_register_policy(_SimdGeneration_, _RegisterPolicy_);
    return __simd_memory_access<_SimdGeneration_, _RegisterPolicy_>::template __load_unaligned<_VectorType_>(__address);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    class				_RegisterPolicy_,
    class				_VectorType_>
simd_stl_nodiscard simd_stl_always_inline _VectorType_ __simd_load_aligned(const void* __address) noexcept {
    __verify_register_policy(_SimdGeneration_, _RegisterPolicy_);
    return __simd_memory_access<_SimdGeneration_, _RegisterPolicy_>::template __load_aligned<_VectorType_>(__address);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    class				_RegisterPolicy_,
    class				_VectorType_>
simd_stl_nodiscard simd_stl_always_inline _VectorType_ __simd_non_temporal_load(const void* __address) noexcept {
    __verify_register_policy(_SimdGeneration_, _RegisterPolicy_);
    return __simd_memory_access<_SimdGeneration_, _RegisterPolicy_>::template __non_temporal_load<_VectorType_>(__address);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    class				_RegisterPolicy_,
    class				_VectorType_>
simd_stl_nodiscard simd_stl_always_inline _VectorType_ __simd_load_upper_half(const void* __address) noexcept {
    __verify_register_policy(_SimdGeneration_, _RegisterPolicy_);
    return __simd_memory_access<_SimdGeneration_, _RegisterPolicy_>::template __load_upper_half<_VectorType_>(__address);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    class				_RegisterPolicy_,
    class				_VectorType_>
simd_stl_nodiscard simd_stl_always_inline _VectorType_ __simd_load_lower_half(const void* __address) noexcept {
    __verify_register_policy(_SimdGeneration_, _RegisterPolicy_);
    return __simd_memory_access<_SimdGeneration_, _RegisterPolicy_>::template __load_lower_half<_VectorType_>(__address);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    class				_RegisterPolicy_,
    typename            _VectorType_,
    typename            _DesiredType_>
simd_stl_always_inline _VectorType_ __simd_mask_load_unaligned(
    const void*                                             __address,
    const type_traits::__deduce_simd_mask_type<
        _SimdGeneration_, _DesiredType_, _RegisterPolicy_>  __mask) noexcept
{
    __verify_register_policy(_SimdGeneration_, _RegisterPolicy_);
    return __simd_memory_access<_SimdGeneration_, _RegisterPolicy_>::template __mask_load_unaligned<_VectorType_, _DesiredType_>(__address, __mask);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    class				_RegisterPolicy_,
    typename            _VectorType_,
    typename            _DesiredType_,
    typename            _VectorMaskType_,
    std::enable_if_t<__is_intrin_type_v<_VectorMaskType_>, int> = 0>
simd_stl_always_inline _VectorType_ __simd_mask_load_unaligned(
    const void*             __address,
    const _VectorMaskType_  __mask) noexcept
{
    __verify_register_policy(_SimdGeneration_, _RegisterPolicy_);
    return __simd_memory_access<_SimdGeneration_, _RegisterPolicy_>::template __mask_load_unaligned<_VectorType_, _DesiredType_, _VectorMaskType_>(__address, __mask);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    class				_RegisterPolicy_,
    typename            _VectorType_,
    typename            _DesiredType_,
    typename            _VectorMaskType_,
    std::enable_if_t<__is_intrin_type_v<_VectorMaskType_>, int> = 0>
simd_stl_always_inline _VectorType_ __simd_mask_load_aligned(
    const void*             __address,
    const _VectorMaskType_  __mask) noexcept
{
    __verify_register_policy(_SimdGeneration_, _RegisterPolicy_);
    return __simd_memory_access<_SimdGeneration_, _RegisterPolicy_>::template __mask_load_aligned<_VectorType_, _DesiredType_, _VectorMaskType_>(__address, __mask);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    class				_RegisterPolicy_,
    typename            _VectorType_,
    typename            _DesiredType_>
simd_stl_always_inline _VectorType_ __simd_mask_load_aligned(
    const void*                                             __address,
    const type_traits::__deduce_simd_mask_type<
        _SimdGeneration_, _DesiredType_, _RegisterPolicy_>  __mask) noexcept
{
    __verify_register_policy(_SimdGeneration_, _RegisterPolicy_);
    return __simd_memory_access<_SimdGeneration_, _RegisterPolicy_>::template __mask_load_aligned<_VectorType_, _DesiredType_>(__address, __mask);
}


template <
    arch::CpuFeature	_SimdGeneration_,
    class				_RegisterPolicy_,
    class				_VectorType_>
simd_stl_nodiscard simd_stl_always_inline void __simd_store_unaligned(
    void*               __address,
    const _VectorType_  __vector) noexcept
{
    __verify_register_policy(_SimdGeneration_, _RegisterPolicy_);
    __simd_memory_access<_SimdGeneration_, _RegisterPolicy_>::template __store_unaligned(__address, __vector);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    class				_RegisterPolicy_,
    class				_VectorType_>
simd_stl_nodiscard simd_stl_always_inline void __simd_store_aligned(
    void*               __address,
    const _VectorType_  __vector) noexcept
{
    __verify_register_policy(_SimdGeneration_, _RegisterPolicy_);
    __simd_memory_access<_SimdGeneration_, _RegisterPolicy_>::template __store_aligned(__address, __vector);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    class				_RegisterPolicy_,
    class				_VectorType_>
simd_stl_nodiscard simd_stl_always_inline void __simd_store_upper_half(
    void*               __address,
    const _VectorType_  __vector) noexcept
{
    __verify_register_policy(_SimdGeneration_, _RegisterPolicy_);
    __simd_memory_access<_SimdGeneration_, _RegisterPolicy_>::template __store_upper_half(__address, __vector);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    class				_RegisterPolicy_,
    class				_VectorType_>
simd_stl_nodiscard simd_stl_always_inline void __simd_store_lower_half(
    void*               __address,
    const _VectorType_  __vector) noexcept
{
    __verify_register_policy(_SimdGeneration_, _RegisterPolicy_);
    __simd_memory_access<_SimdGeneration_, _RegisterPolicy_>::template __store_lower_half(__address, __vector);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    class				_RegisterPolicy_,
    class				_VectorType_>
simd_stl_nodiscard simd_stl_always_inline void __simd_non_temporal_store(
    void*               __address,
    const _VectorType_  __vector) noexcept
{
    __verify_register_policy(_SimdGeneration_, _RegisterPolicy_);
    __simd_memory_access<_SimdGeneration_, _RegisterPolicy_>::template __non_temporal_store(__address, __vector);
}

template <arch::CpuFeature _SimdGeneration_>
simd_stl_nodiscard simd_stl_always_inline void __simd_streaming_fence() noexcept {
    __simd_memory_access<_SimdGeneration_, __default_register_policy<_SimdGeneration_>>::__streaming_fence();
}

template <
    arch::CpuFeature	_SimdGeneration_,
    class				_RegisterPolicy_,
    class               _DesiredType_,
    class               _VectorType_>
simd_stl_always_inline void __simd_mask_store_unaligned(
    void*                                                   __address,
    const type_traits::__deduce_simd_mask_type<
        _SimdGeneration_, _DesiredType_, _RegisterPolicy_>  __mask,
    const _VectorType_                                      __vector) noexcept
{
    __verify_register_policy(_SimdGeneration_, _RegisterPolicy_);
    __simd_memory_access<_SimdGeneration_, _RegisterPolicy_>::template __mask_store_unaligned<_DesiredType_>(__address, __mask, __vector);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    class				_RegisterPolicy_,
    class               _DesiredType_,
    class               _VectorType_>
simd_stl_always_inline void __simd_mask_store_aligned(
    void*                                                   __address,
    const type_traits::__deduce_simd_mask_type<
        _SimdGeneration_, _DesiredType_, _RegisterPolicy_>  __mask,
    const _VectorType_                                      __vector) noexcept
{
    __verify_register_policy(_SimdGeneration_, _RegisterPolicy_);
    __simd_memory_access<_SimdGeneration_, _RegisterPolicy_>::template __mask_store_aligned<_DesiredType_>(__address, __mask, __vector);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    class				_RegisterPolicy_,
    class               _DesiredType_,
    class               _MaskVectorType_,
    class               _VectorType_,
    std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int> = 0>
simd_stl_always_inline void __simd_mask_store_unaligned(
    void*                   __address,
    const _MaskVectorType_  __mask,
    const _VectorType_      __vector) noexcept
{
    __verify_register_policy(_SimdGeneration_, _RegisterPolicy_);
    __simd_memory_access<_SimdGeneration_, _RegisterPolicy_>::template __mask_store_unaligned<_DesiredType_>(__address, __mask, __vector);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    class				_RegisterPolicy_,
    class               _DesiredType_,
    class               _MaskVectorType_,
    class               _VectorType_,
    std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int> = 0>
simd_stl_always_inline void __simd_mask_store_aligned(
    void*                   __address,
    const _MaskVectorType_  __mask,
    const _VectorType_      __vector) noexcept
{
    __verify_register_policy(_SimdGeneration_, _RegisterPolicy_);
    __simd_memory_access<_SimdGeneration_, _RegisterPolicy_>::template __mask_store_aligned<_DesiredType_>(__address, __mask, __vector);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    class				_RegisterPolicy_,
    typename            _DesiredType_,
    typename            _VectorType_>
simd_stl_always_inline _DesiredType_* __simd_compress_store_unaligned(
    _DesiredType_*                          __address,
    type_traits::__deduce_simd_mask_type<_SimdGeneration_,
        _DesiredType_, _RegisterPolicy_>    __mask,
    const _VectorType_                      __vector) noexcept
{
    __verify_register_policy(_SimdGeneration_, _RegisterPolicy_);
    return __simd_memory_access<_SimdGeneration_, _RegisterPolicy_>::template __compress_store_unaligned(__address, __mask, __vector);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    class				_RegisterPolicy_,
    typename            _VectorType_,
    typename            _DesiredType_>
simd_stl_always_inline _VectorType_ __simd_mask_load_unaligned(
    const void*                                             __address,
    const type_traits::__deduce_simd_mask_type<
        _SimdGeneration_, _DesiredType_, _RegisterPolicy_>  __mask,
    _VectorType_                                            __additional_source) noexcept
{
    __verify_register_policy(_SimdGeneration_, _RegisterPolicy_);
    return __simd_memory_access<_SimdGeneration_, _RegisterPolicy_>
        ::template __mask_load_unaligned<_VectorType_, _DesiredType_>(__address, __mask, __additional_source);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    class				_RegisterPolicy_,
    typename            _VectorType_,
    typename            _DesiredType_,
    typename            _VectorMaskType_,
    std::enable_if_t<__is_intrin_type_v<_VectorMaskType_>, int> = 0>
simd_stl_always_inline _VectorType_ __simd_mask_load_unaligned(
    const void*             __address,
    const _VectorMaskType_  __mask,
    _VectorType_            __additional_source) noexcept
{
    __verify_register_policy(_SimdGeneration_, _RegisterPolicy_);
    return __simd_memory_access<_SimdGeneration_, _RegisterPolicy_>
        ::template __mask_load_unaligned<_VectorType_, _DesiredType_, _VectorMaskType_>(__address, __mask, __additional_source);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    class				_RegisterPolicy_,
    typename            _VectorType_,
    typename            _DesiredType_,
    typename            _VectorMaskType_,
    std::enable_if_t<__is_intrin_type_v<_VectorMaskType_>, int> = 0>
simd_stl_always_inline _VectorType_ __simd_mask_load_aligned(
    const void*             __address,
    const _VectorMaskType_  __mask,
    _VectorType_            __additional_source) noexcept
{
    __verify_register_policy(_SimdGeneration_, _RegisterPolicy_);
    return __simd_memory_access<_SimdGeneration_, _RegisterPolicy_>
        ::template __mask_load_aligned<_VectorType_, _DesiredType_, _VectorMaskType_>(__address, __mask, __additional_source);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    class				_RegisterPolicy_,
    typename            _VectorType_,
    typename            _DesiredType_>
simd_stl_always_inline _VectorType_ __simd_mask_load_aligned(
    const void*                                             __address,
    const type_traits::__deduce_simd_mask_type<
        _SimdGeneration_, _DesiredType_, _RegisterPolicy_>  __mask,
    _VectorType_                                            __additional_source) noexcept
{
    __verify_register_policy(_SimdGeneration_, _RegisterPolicy_);
    return __simd_memory_access<_SimdGeneration_, _RegisterPolicy_>
        ::template __mask_load_aligned<_VectorType_, _DesiredType_>(__address, __mask, __additional_source);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    class				_RegisterPolicy_,
    typename            _DesiredType_,
    typename            _VectorType_>
simd_stl_always_inline _DesiredType_* __simd_compress_store_aligned(
    _DesiredType_*                          __address,
    type_traits::__deduce_simd_mask_type<_SimdGeneration_,
        _DesiredType_, _RegisterPolicy_>    __mask,
    const _VectorType_                      __vector) noexcept
{
    __verify_register_policy(_SimdGeneration_, _RegisterPolicy_);
    return __simd_memory_access<_SimdGeneration_, _RegisterPolicy_>::template __compress_store_aligned(__address, __mask, __vector);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_,
    typename            _Type_>
constexpr inline bool __is_native_mask_load_supported_v = __simd_memory_access<_SimdGeneration_, _RegisterPolicy_>
    ::template __native_mask_load_supported<sizeof(_Type_)>;

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_,
    typename            _Type_>
constexpr inline bool __is_native_mask_store_supported_v = __simd_memory_access<_SimdGeneration_, _RegisterPolicy_>
    ::template __native_mask_store_supported<sizeof(_Type_)>;

template <
    arch::CpuFeature	_SimdGeneration_,
    class               _RegisterPolicy_,
    typename            _Type_>
simd_stl_always_inline auto __simd_make_tail_mask(uint32 __bytes) noexcept {
    return __simd_memory_access<_SimdGeneration_, _RegisterPolicy_>::template __make_tail_mask<_Type_>(__bytes);
}

template <
    class		_BasicSimd_,
    typename	_ReturnType_>
using __make_tail_mask_return_type_helper = std::conditional_t<__is_intrin_type_v<_ReturnType_>,
    simd<_BasicSimd_::__generation, typename _BasicSimd_::value_type, typename _BasicSimd_::policy_type>, _ReturnType_>;

template <class _BasicSimd_>
using __make_tail_mask_return_type = __make_tail_mask_return_type_helper<_BasicSimd_,
    type_traits::invoke_result_type<decltype(__simd_make_tail_mask<_BasicSimd_::__generation, typename _BasicSimd_::policy_type,
    typename _BasicSimd_::value_type>), uint32>>;

template <
    arch::CpuFeature	_SimdGeneration_,
    class               _RegisterPolicy_,
    typename            _MaskOrBasicSimdType_>
simd_stl_always_inline auto __simd_to_native_mask(_MaskOrBasicSimdType_ __mask) noexcept {
    if constexpr (__is_valid_basic_simd_v<_MaskOrBasicSimdType_>)
        return __mask.to_mask();
    else
        return __mask;
}

__SIMD_STL_NUMERIC_NAMESPACE_END

#include <src/simd_stl/numeric/SimdMemoryAccess.inl>