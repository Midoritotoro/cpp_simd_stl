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

struct __aligned_policy {
    static constexpr bool __alignment = true;
};

struct __unaligned_policy {
    static constexpr bool __alignment = false;
};

#pragma region Sse2-Sse4.2 memory access 

template <>
class __simd_memory_access<arch::CpuFeature::SSE2, xmm128> {
    static constexpr auto __generation   = arch::CpuFeature::SSE2;
    using __register_policy = xmm128;
public:
    template <sizetype _TypeSize_>
    static constexpr auto __native_mask_load_supported = false;

    template <sizetype _TypeSize_>
    static constexpr auto __native_mask_store_supported = false;

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ __non_temporal_load(const void* __address) noexcept;

    template <typename _VectorType_>
    static simd_stl_always_inline void __non_temporal_store(
        void*           __address,
        _VectorType_    __vector) noexcept;

    static simd_stl_always_inline void __streaming_fence() noexcept;

    template <
        typename                    _VectorType_,
        class _AlignmentPolicy_ =   __unaligned_policy>
    static simd_stl_always_inline _VectorType_ __load(
        const void*         __address,
        _AlignmentPolicy_&& __policy = _AlignmentPolicy_{}) noexcept;

    template <
        typename                    _VectorType_,
        class _AlignmentPolicy_ =   __unaligned_policy>
    static simd_stl_always_inline void __store(
        void*               __address,
        _VectorType_        __vector,
        _AlignmentPolicy_&& __policy = _AlignmentPolicy_{}) noexcept;

    template <
        typename    _DesiredType_,
        typename    _VectorType_,
        typename    _MaskType_,
        class       _AlignmentPolicy_ = __unaligned_policy>
    static simd_stl_always_inline void __mask_store(
        void*               __address,
        _MaskType_          __mask,
        _VectorType_        __vector,
        _AlignmentPolicy_&& __policy = _AlignmentPolicy_{}) noexcept;

    template <
        typename    _DesiredType_,
        typename    _VectorType_,
        class       _MaskType_,
        class       _AlignmentPolicy_ = __unaligned_policy>
    static simd_stl_always_inline _VectorType_ __mask_load(
        const void*         __address,
        _MaskType_          __mask,
        _AlignmentPolicy_&& __policy = _AlignmentPolicy_{}) noexcept;

    template <
        typename    _DesiredType_,
        typename    _VectorType_,
        class       _MaskType_,
        class       _AlignmentPolicy_ = __unaligned_policy>
    static simd_stl_always_inline _VectorType_ __mask_load(
        const void*         __address,
        _MaskType_           __mask,
        _VectorType_        __additional_source,
        _AlignmentPolicy_&& __policy = _AlignmentPolicy_{}) noexcept;

    template <
        typename    _DesiredType_,
        typename    _VectorType_,
        class       _MaskType_,
        class       _AlignmentPolicy_ = __unaligned_policy>
    static simd_stl_always_inline _DesiredType_* __compress_store(
        _DesiredType_*      __address,
        _MaskType_          __mask,
        _VectorType_        __vector,
        _AlignmentPolicy_&& __policy = _AlignmentPolicy_{}) noexcept;

    template <typename _Type_>
    static simd_stl_always_inline __m128i __make_tail_mask(uint32 __bytes) noexcept;
};

template <>
class __simd_memory_access<arch::CpuFeature::SSE3, xmm128>:
    public __simd_memory_access<arch::CpuFeature::SSE2, xmm128>
{
public:
    template <
        typename                    _VectorType_,
        class _AlignmentPolicy_ =   __unaligned_policy>
    static simd_stl_always_inline _VectorType_ __load(
        const void*         __address,
        _AlignmentPolicy_&& __policy = _AlignmentPolicy_{}) noexcept;
};

template <>
class __simd_memory_access<arch::CpuFeature::SSSE3, xmm128> :
    public __simd_memory_access<arch::CpuFeature::SSE3, xmm128>
{
    static constexpr auto __generation  = arch::CpuFeature::SSSE3;
    using __register_policy             = xmm128;
public:
    template <
        typename    _DesiredType_,
        typename    _VectorType_,
        class       _MaskType_,
        class       _AlignmentPolicy_ = __unaligned_policy>
    static simd_stl_always_inline _DesiredType_* __compress_store(
        _DesiredType_*      __address,
        _MaskType_          __mask,
        _VectorType_        __vector,
        _AlignmentPolicy_&& __policy = _AlignmentPolicy_{}) noexcept;
};

template <>
class __simd_memory_access<arch::CpuFeature::SSE41, xmm128> :
    public __simd_memory_access<arch::CpuFeature::SSSE3, xmm128>
{    
    static constexpr auto __generation  = arch::CpuFeature::SSE41;
    using __register_policy             = xmm128;
public:
    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ __non_temporal_load(const void* __address) noexcept;

    template <
        typename    _DesiredType_,
        typename    _VectorType_,
        class       _MaskType_,
        class       _AlignmentPolicy_ = __unaligned_policy>
    static simd_stl_always_inline void __mask_store(
        void*               __address,
        _MaskType_          __mask,
        _VectorType_        __vector,
        _AlignmentPolicy_&& __policy = _AlignmentPolicy_{}) noexcept;

    template <
        typename    _DesiredType_,
        typename    _VectorType_,
        class       _MaskType_,
        class       _AlignmentPolicy_ = __unaligned_policy>
    static simd_stl_always_inline _VectorType_ __mask_load(
        const void*         __address,
        _MaskType_          __mask,
        _AlignmentPolicy_&& __policy = _AlignmentPolicy_{}) noexcept;

    template <
        typename    _DesiredType_,
        typename    _VectorType_,
        class       _MaskType_,
        class       _AlignmentPolicy_ = __unaligned_policy>
    static simd_stl_always_inline _VectorType_ __mask_load(
        const void*         __address,
        _MaskType_          __mask,
        _VectorType_        __additional_source,
        _AlignmentPolicy_&& __policy = _AlignmentPolicy_{}) noexcept;
};

template <>
class __simd_memory_access<arch::CpuFeature::SSE42, xmm128> :
    public __simd_memory_access<arch::CpuFeature::SSE41, xmm128>
{};

#pragma endregion

#pragma region Avx-Avx2 memory access

template <>
class __simd_memory_access<arch::CpuFeature::AVX2, xmm128> :
    public __simd_memory_access<arch::CpuFeature::SSE42, xmm128>
{
    static constexpr auto __generation  = arch::CpuFeature::AVX2;
    using __register_policy             = xmm128;
public:
    template <
        typename    _DesiredType_,
        typename    _VectorType_,
        class       _MaskType_,
        class       _AlignmentPolicy_ = __unaligned_policy>
    static simd_stl_always_inline void __mask_store(
        void*               __address,
        _MaskType_          __mask,
        _VectorType_        __vector,
        _AlignmentPolicy_&& __policy = _AlignmentPolicy_{}) noexcept;

    template <
        typename    _DesiredType_,
        typename    _VectorType_,
        class       _MaskType_,
        class       _AlignmentPolicy_ = __unaligned_policy>
    static simd_stl_always_inline _VectorType_ __mask_load(
        const void*         __address,
        _MaskType_          __mask,
        _AlignmentPolicy_&& __policy = _AlignmentPolicy_{}) noexcept;

    template <
        typename    _DesiredType_,
        typename    _VectorType_,
        class       _MaskType_,
        class       _AlignmentPolicy_ = __unaligned_policy>
    static simd_stl_always_inline _VectorType_ __mask_load(
        const void*         __address,
        _MaskType_          __mask,
        _VectorType_        __additional_source,
        _AlignmentPolicy_&& __policy = _AlignmentPolicy_{}) noexcept;
};


template <>
class __simd_memory_access<arch::CpuFeature::AVX2, ymm256>
{
    static constexpr auto __generation  = arch::CpuFeature::AVX2;
    using __register_policy             = ymm256;
public:
    template <sizetype _TypeSize_>
    static constexpr auto __native_mask_load_supported = (_TypeSize_ == 4) || (_TypeSize_ == 8);

    template <sizetype _TypeSize_>
    static constexpr auto __native_mask_store_supported = (_TypeSize_ == 4) || (_TypeSize_ == 8);

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ __non_temporal_load(const void* __address) noexcept;

    template <typename _VectorType_>
    static simd_stl_always_inline void __non_temporal_store(
        void*           __address,
        _VectorType_    __vector) noexcept;

    static simd_stl_always_inline void __streaming_fence() noexcept;

    template <
        typename                    _VectorType_,
        class _AlignmentPolicy_ =   __unaligned_policy>
    static simd_stl_always_inline _VectorType_ __load(
        const void*         __address,
        _AlignmentPolicy_&& __policy = _AlignmentPolicy_{}) noexcept;

    template <
        typename                    _VectorType_,
        class _AlignmentPolicy_ =   __unaligned_policy>
    static simd_stl_always_inline void __store(
        void*               __address,
        _VectorType_        __vector,
        _AlignmentPolicy_&& __policy = _AlignmentPolicy_{}) noexcept;

    template <
        typename    _DesiredType_,
        typename    _VectorType_,
        class       _MaskType_,
        class       _AlignmentPolicy_ = __unaligned_policy>
    static simd_stl_always_inline void __mask_store(
        void*               __address,
        _MaskType_          __mask,
        _VectorType_        __vector,
        _AlignmentPolicy_&& __policy = _AlignmentPolicy_{}) noexcept;

    template <
        typename    _DesiredType_,
        typename    _VectorType_,
        class       _MaskType_,
        class       _AlignmentPolicy_ = __unaligned_policy>
    static simd_stl_always_inline _VectorType_ __mask_load(
        const void*         __address,
        _MaskType_          __mask,
        _AlignmentPolicy_&& __policy = _AlignmentPolicy_{}) noexcept;

    template <
        typename    _DesiredType_,
        typename    _VectorType_,
        class       _MaskType_,
        class       _AlignmentPolicy_ = __unaligned_policy>
    static simd_stl_always_inline _VectorType_ __mask_load(
        const void*         __address,
        _MaskType_          __mask,
        _VectorType_        __additional_source,
        _AlignmentPolicy_&& __policy = _AlignmentPolicy_{}) noexcept;
    
    template <
        typename    _DesiredType_,
        typename    _VectorType_,
        class       _MaskType_,
        class       _AlignmentPolicy_ = __unaligned_policy>
    static simd_stl_always_inline _DesiredType_* __compress_store(
        _DesiredType_*      __address,
        _MaskType_          __mask,
        _VectorType_        __vector,
        _AlignmentPolicy_&& __policy = _AlignmentPolicy_{}) noexcept;

    template <typename _Type_>
    static simd_stl_always_inline __m256i __make_tail_mask(uint32 __bytes) noexcept;
};

#pragma endregion

#pragma region Avx512 memory access

template <>
class __simd_memory_access<arch::CpuFeature::AVX512F, zmm512>
{
    static constexpr auto __generation   = arch::CpuFeature::AVX512F;
    using __register_policy = zmm512;
public:
    template <sizetype _TypeSize_>
    static constexpr auto __native_mask_load_supported = (_TypeSize_ == 4) || (_TypeSize_ == 8);

    template <sizetype _TypeSize_>
    static constexpr auto __native_mask_store_supported = (_TypeSize_ == 4) || (_TypeSize_ == 8);

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ __non_temporal_load(const void* __address) noexcept;

    template <typename _VectorType_>
    static simd_stl_always_inline void __non_temporal_store(
        void*           __address,
        _VectorType_    __vector) noexcept;

    static simd_stl_always_inline void __streaming_fence() noexcept;

    template <
        typename                    _VectorType_,
        class _AlignmentPolicy_ =   __unaligned_policy>
    static simd_stl_always_inline _VectorType_ __load(
        const void*         __address,
        _AlignmentPolicy_&& __policy = _AlignmentPolicy_{}) noexcept;

    template <
        typename                    _VectorType_,
        class _AlignmentPolicy_ =   __unaligned_policy>
    static simd_stl_always_inline void __store(
        void*               __address,
        _VectorType_        __vector,
        _AlignmentPolicy_&& __policy = _AlignmentPolicy_{}) noexcept;

    template <
        typename    _DesiredType_,
        typename    _VectorType_,
        class       _MaskType_,
        class       _AlignmentPolicy_ = __unaligned_policy>
    static simd_stl_always_inline void __mask_store(
        void*               __address,
        _MaskType_          __mask,
        _VectorType_        __vector,
        _AlignmentPolicy_&& __policy = _AlignmentPolicy_{}) noexcept;

    template <
        typename    _DesiredType_,
        typename    _VectorType_,
        class       _MaskType_,
        class       _AlignmentPolicy_ = __unaligned_policy>
    static simd_stl_always_inline _VectorType_ __mask_load(
        const void*         __address,
        _MaskType_          __mask,
        _AlignmentPolicy_&& __policy = _AlignmentPolicy_{}) noexcept;

    template <
        typename    _DesiredType_,
        typename    _VectorType_,
        class       _MaskType_,
        class       _AlignmentPolicy_ = __unaligned_policy>
    static simd_stl_always_inline _VectorType_ __mask_load(
        const void*         __address,
        _MaskType_          __mask,
        _VectorType_        __additional_source,
        _AlignmentPolicy_&& __policy = _AlignmentPolicy_{}) noexcept;
    
    template <
        typename    _DesiredType_,
        typename    _VectorType_,
        class       _MaskType_,
        class       _AlignmentPolicy_ = __unaligned_policy>
    static simd_stl_always_inline _DesiredType_* __compress_store(
        _DesiredType_*      __address,
        _MaskType_          __mask,
        _VectorType_        __vector,
        _AlignmentPolicy_&& __policy = _AlignmentPolicy_{}) noexcept;

    template <typename _Type_>
    static simd_stl_always_inline auto __make_tail_mask(uint32 __bytes) noexcept;
};

template <>
class __simd_memory_access<arch::CpuFeature::AVX512BW, zmm512>:
    public __simd_memory_access<arch::CpuFeature::AVX512F, zmm512>
{
    static constexpr auto __generation = arch::CpuFeature::AVX512BW;
    using __register_policy = zmm512;
public:
    template <sizetype _TypeSize_>
    static constexpr auto __native_mask_load_supported = true;

    template <sizetype _TypeSize_>
    static constexpr auto __native_mask_store_supported = true;

    template <
        typename    _DesiredType_,
        typename    _VectorType_,
        class       _MaskType_,
        class       _AlignmentPolicy_ = __unaligned_policy>
    static simd_stl_always_inline void __mask_store(
        void*               __address,
        _MaskType_          __mask,
        _VectorType_        __vector,
        _AlignmentPolicy_&& __policy = _AlignmentPolicy_{}) noexcept;

    template <
        typename    _DesiredType_,
        typename    _VectorType_,
        class       _MaskType_,
        class       _AlignmentPolicy_ = __unaligned_policy>
    static simd_stl_always_inline _VectorType_ __mask_load(
        const void*             __address,
        _MaskType_              __mask,
        _AlignmentPolicy_&&     __policy = _AlignmentPolicy_{}) noexcept;
        
    template <
        typename    _DesiredType_,
        typename    _VectorType_,
        class       _MaskType_,
        class       _AlignmentPolicy_ = __unaligned_policy>
    static simd_stl_always_inline _VectorType_ __mask_load(
        const void*         __address,
        _MaskType_          __mask,
        _VectorType_        __additional_policy,
        _AlignmentPolicy_&& __policy = _AlignmentPolicy_{}) noexcept;
};

template <>
class __simd_memory_access<arch::CpuFeature::AVX512DQ, zmm512> :
    public __simd_memory_access<arch::CpuFeature::AVX512F, zmm512>
{};

template <>
class __simd_memory_access<arch::CpuFeature::AVX512BWDQ, zmm512> :
    public __simd_memory_access<arch::CpuFeature::AVX512BW, zmm512>
{};


template <>
class __simd_memory_access<arch::CpuFeature::AVX512VLF, ymm256> :
    public __simd_memory_access<arch::CpuFeature::AVX2, ymm256>
{
    static constexpr auto __generation   = arch::CpuFeature::AVX512VLF;
    using __register_policy = ymm256;
public:
    template <sizetype _TypeSize_>
    static constexpr auto __native_mask_load_supported = (_TypeSize_ == 4) || (_TypeSize_ == 8);

    template <sizetype _TypeSize_>
    static constexpr auto __native_mask_store_supported = (_TypeSize_ == 4) || (_TypeSize_ == 8);

    template <
        typename    _DesiredType_,
        typename    _VectorType_,
        class       _MaskType_,
        class       _AlignmentPolicy_ = __unaligned_policy>
    static simd_stl_always_inline void __mask_store(
        void*               __address,
        _MaskType_          __mask,
        _VectorType_        __vector,
        _AlignmentPolicy_&& __policy = _AlignmentPolicy_{}) noexcept;

    template <
        typename    _DesiredType_,
        typename    _VectorType_,
        class       _MaskType_,
        class       _AlignmentPolicy_ = __unaligned_policy>
    static simd_stl_always_inline _VectorType_ __mask_load(
        const void*             __address,
        _MaskType_              __mask,
        _AlignmentPolicy_&&     __policy = _AlignmentPolicy_{}) noexcept;
        
    template <
        typename    _DesiredType_,
        typename    _VectorType_,
        class       _MaskType_,
        class       _AlignmentPolicy_ = __unaligned_policy>
    static simd_stl_always_inline _VectorType_ __mask_load(
        const void*         __address,
        _MaskType_          __mask,
        _VectorType_        __additional_policy,
        _AlignmentPolicy_&& __policy = _AlignmentPolicy_{}) noexcept;
    
    template <
        typename    _DesiredType_,
        typename    _VectorType_,
        class       _MaskType_,
        class       _AlignmentPolicy_ = __unaligned_policy>
    static simd_stl_always_inline _DesiredType_* __compress_store(
        _DesiredType_*      __address,
        _MaskType_          __mask,
        _VectorType_        __vector,
        _AlignmentPolicy_&& __policy = _AlignmentPolicy_{}) noexcept;

    template <typename _Type_>
    static simd_stl_always_inline auto __make_tail_mask(uint32 __bytes) noexcept;
};

template <>
class __simd_memory_access<arch::CpuFeature::AVX512VLBW, ymm256> :
    public __simd_memory_access<arch::CpuFeature::AVX512VLF, ymm256>
{
    static constexpr auto __generation   = arch::CpuFeature::AVX512VLBW;
    using __register_policy = ymm256;
public:
    template <sizetype _TypeSize_>
    static constexpr auto __native_mask_load_supported = true;

    template <sizetype _TypeSize_>
    static constexpr auto __native_mask_store_supported = true;

    template <
        typename    _DesiredType_,
        typename    _VectorType_,
        class       _MaskType_,
        class       _AlignmentPolicy_ = __unaligned_policy>
    static simd_stl_always_inline void __mask_store(
        void*               __address,
        _MaskType_          __mask,
        _VectorType_        __vector,
        _AlignmentPolicy_&& __policy = _AlignmentPolicy_{}) noexcept;

    template <
        typename    _DesiredType_,
        typename    _VectorType_,
        class       _MaskType_,
        class       _AlignmentPolicy_ = __unaligned_policy>
    static simd_stl_always_inline _VectorType_ __mask_load(
        const void*             __address,
        _MaskType_              __mask,
        _AlignmentPolicy_&&     __policy = _AlignmentPolicy_{}) noexcept;
        
    template <
        typename    _DesiredType_,
        typename    _VectorType_,
        class       _MaskType_,
        class       _AlignmentPolicy_ = __unaligned_policy>
    static simd_stl_always_inline _VectorType_ __mask_load(
        const void*         __address,
        _MaskType_          __mask,
        _VectorType_        __additional_policy,
        _AlignmentPolicy_&& __policy = _AlignmentPolicy_{}) noexcept;
};

template <>
class __simd_memory_access<arch::CpuFeature::AVX512VLDQ, ymm256> :
    public __simd_memory_access<arch::CpuFeature::AVX512VLF, ymm256>
{};

template <>
class __simd_memory_access<arch::CpuFeature::AVX512VLBWDQ, ymm256> :
    public __simd_memory_access<arch::CpuFeature::AVX512VLBW, ymm256>
{};

template <>
class __simd_memory_access<arch::CpuFeature::AVX512VLF, xmm128> :
    public __simd_memory_access<arch::CpuFeature::SSE42, xmm128>
{
    static constexpr auto __generation      = arch::CpuFeature::AVX512VLF;
    using __register_policy                 = xmm128;
public:
    template <sizetype _TypeSize_>
    static constexpr auto __native_mask_load_supported = (_TypeSize_ == 4) || (_TypeSize_ == 8);

    template <sizetype _TypeSize_>
    static constexpr auto __native_mask_store_supported = (_TypeSize_ == 4) || (_TypeSize_ == 8);

    template <typename _Type_>
    static simd_stl_always_inline auto __make_tail_mask(uint32 __bytes) noexcept;

    template <
        typename    _DesiredType_,
        typename    _VectorType_,
        class       _MaskType_,
        class       _AlignmentPolicy_ = __unaligned_policy>
    static simd_stl_always_inline void __mask_store(
        void*               __address,
        _MaskType_          __mask,
        _VectorType_        __vector,
        _AlignmentPolicy_&& __policy = _AlignmentPolicy_{}) noexcept;

    template <
        typename    _DesiredType_,
        typename    _VectorType_,
        class       _MaskType_,
        class       _AlignmentPolicy_ = __unaligned_policy>
    static simd_stl_always_inline _VectorType_ __mask_load(
        const void*             __address,
        _MaskType_              __mask,
        _AlignmentPolicy_&&     __policy = _AlignmentPolicy_{}) noexcept;
        
    template <
        typename    _DesiredType_,
        typename    _VectorType_,
        class       _MaskType_,
        class       _AlignmentPolicy_ = __unaligned_policy>
    static simd_stl_always_inline _VectorType_ __mask_load(
        const void*         __address,
        _MaskType_          __mask,
        _VectorType_        __additional_policy,
        _AlignmentPolicy_&& __policy = _AlignmentPolicy_{}) noexcept;
    
    template <
        typename    _DesiredType_,
        typename    _VectorType_,
        class       _MaskType_,
        class       _AlignmentPolicy_ = __unaligned_policy>
    static simd_stl_always_inline _DesiredType_* __compress_store(
        _DesiredType_*      __address,
        _MaskType_          __mask,
        _VectorType_        __vector,
        _AlignmentPolicy_&& __policy = _AlignmentPolicy_{}) noexcept;
};

template <>
class __simd_memory_access<arch::CpuFeature::AVX512VLBW, xmm128> :
    public __simd_memory_access<arch::CpuFeature::AVX512VLF, xmm128>
{
    static constexpr auto __generation   = arch::CpuFeature::AVX512VLBW;
    using __register_policy               = xmm128;
public:
    template <sizetype _TypeSize_>
    static constexpr auto __native_mask_load_supported = true;

    template <sizetype _TypeSize_>
    static constexpr auto __native_mask_store_supported = true;

    template <
        typename    _DesiredType_,
        typename    _VectorType_,
        class       _MaskType_,
        class       _AlignmentPolicy_ = __unaligned_policy>
    static simd_stl_always_inline void __mask_store(
        void*               __address,
        const _MaskType_    __mask,
        const _VectorType_  __vector,
        _AlignmentPolicy_&& __policy = _AlignmentPolicy_{}) noexcept;

    template <
        typename    _DesiredType_,
        typename    _VectorType_,
        class       _MaskType_,
        class       _AlignmentPolicy_ = __unaligned_policy>
    static simd_stl_always_inline _VectorType_ __mask_load(
        const void*             __address,
        _MaskType_              __mask,
        _AlignmentPolicy_&&     __policy = _AlignmentPolicy_{}) noexcept;
        
    template <
        typename    _DesiredType_,
        typename    _VectorType_,
        class       _MaskType_,
        class       _AlignmentPolicy_ = __unaligned_policy>
    static simd_stl_always_inline _VectorType_ __mask_load(
        const void*         __address,
        _MaskType_          __mask,
        _VectorType_        __additional_policy,
        _AlignmentPolicy_&& __policy = _AlignmentPolicy_{}) noexcept;
};

template <>
class __simd_memory_access<arch::CpuFeature::AVX512VLDQ, xmm128> :
    public __simd_memory_access<arch::CpuFeature::AVX512VLF, xmm128>
{};

template <>
class __simd_memory_access<arch::CpuFeature::AVX512VLBWDQ, xmm128> :
    public __simd_memory_access<arch::CpuFeature::AVX512VLBW, xmm128>
{};

#pragma endregion

template <
    arch::CpuFeature	_SimdGeneration_,
    class				_RegisterPolicy_,
    class				_VectorType_,
    class               _AlignmentPolicy_ = __unaligned_policy>
simd_stl_nodiscard simd_stl_always_inline _VectorType_ __simd_load(
    const void*         __address, 
    _AlignmentPolicy_&& __policy = _AlignmentPolicy_{}) noexcept
{
    __verify_register_policy(_SimdGeneration_, _RegisterPolicy_);
    return __simd_memory_access<_SimdGeneration_, _RegisterPolicy_>::template __load<_VectorType_>(__address, __policy);
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
simd_stl_nodiscard simd_stl_always_inline void __simd_non_temporal_store(
    void*           __address,
    _VectorType_    __vector) noexcept
{
    __verify_register_policy(_SimdGeneration_, _RegisterPolicy_);
    __simd_memory_access<_SimdGeneration_, _RegisterPolicy_>::template __non_temporal_store(__address, __vector);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    class				_RegisterPolicy_,
    typename            _DesiredType_,
    typename            _VectorType_,
    class               _MaskType_,
    class               _AlignmentPolicy_ = __unaligned_policy>
simd_stl_always_inline _VectorType_ __simd_mask_load(
    const void*         __address,
    _MaskType_          __mask,
    _AlignmentPolicy_&& __policy = _AlignmentPolicy_{}) noexcept
{
    __verify_register_policy(_SimdGeneration_, _RegisterPolicy_);
    return __simd_memory_access<_SimdGeneration_, _RegisterPolicy_>::template __mask_load<_DesiredType_, _VectorType_>(__address, __mask, __policy);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    class				_RegisterPolicy_,
    typename            _DesiredType_,
    typename            _VectorType_,
    class               _MaskType_,
    class               _AlignmentPolicy_ = __unaligned_policy>
simd_stl_always_inline _VectorType_ __simd_mask_load(
    const void*         __address,
    _MaskType_          __mask,
    _VectorType_        __additional_source,
    _AlignmentPolicy_&& __policy) noexcept
{
    __verify_register_policy(_SimdGeneration_, _RegisterPolicy_);
    return __simd_memory_access<_SimdGeneration_, _RegisterPolicy_>::template __mask_load<_DesiredType_>(__address, __mask, __additional_source, __policy);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    class				_RegisterPolicy_,
    class				_VectorType_,
    class               _AlignmentPolicy_ = __unaligned_policy>
simd_stl_nodiscard simd_stl_always_inline void __simd_store(
    void*               __address,
    _VectorType_        __vector,
    _AlignmentPolicy_&& __policy = _AlignmentPolicy_{}) noexcept
{
    __verify_register_policy(_SimdGeneration_, _RegisterPolicy_);
    __simd_memory_access<_SimdGeneration_, _RegisterPolicy_>::template __store(__address, __vector, __policy);
}

template <arch::CpuFeature _SimdGeneration_>
simd_stl_nodiscard simd_stl_always_inline void __simd_streaming_fence() noexcept {
    __simd_memory_access<_SimdGeneration_, __default_register_policy<_SimdGeneration_>>::__streaming_fence();
}

template <
    arch::CpuFeature	_SimdGeneration_,
    class				_RegisterPolicy_,
    class               _DesiredType_,
    class               _MaskVectorType_,
    class               _VectorType_,
    class               _AlignmentPolicy_ = __unaligned_policy>
simd_stl_always_inline void __simd_mask_store(
    void*                   __address,
    _MaskVectorType_        __mask,
    _VectorType_            __vector,
    _AlignmentPolicy_&&     __policy = _AlignmentPolicy_{}) noexcept
{
    __verify_register_policy(_SimdGeneration_, _RegisterPolicy_);
    __simd_memory_access<_SimdGeneration_, _RegisterPolicy_>::template __mask_store<_DesiredType_>(__address, __mask, __vector, __policy);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    class				_RegisterPolicy_,
    typename            _DesiredType_,
    class               _MaskType_,
    typename            _VectorType_,
    class               _AlignmentPolicy_ = __unaligned_policy>
simd_stl_always_inline _DesiredType_* __simd_compress_store(
    _DesiredType_*      __address,
    _MaskType_          __mask,
    _VectorType_        __vector,
    _AlignmentPolicy_&& __policy) noexcept
{
    __verify_register_policy(_SimdGeneration_, _RegisterPolicy_);
    return __simd_memory_access<_SimdGeneration_, _RegisterPolicy_>::template __compress_store(__address, __mask, __vector, __policy);
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
    typename _BasicSimd_::value_type>), simd_mask<_BasicSimd_::__generation, typename _BasicSimd_::value_type, typename _BasicSimd_::policy_type>>;

__SIMD_STL_NUMERIC_NAMESPACE_END

#include <src/simd_stl/numeric/SimdMemoryAccess.inl>