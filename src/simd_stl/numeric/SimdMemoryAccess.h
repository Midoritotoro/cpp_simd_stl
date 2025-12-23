#pragma once 

#include <src/simd_stl/numeric/SimdElementWise.h>
#include <simd_stl/numeric/BasicSimdMask.h>

#include <src/simd_stl/algorithm/AdvanceBytes.h>
#include <src/simd_stl/numeric/SimdBroadcast.h>

__SIMD_STL_NUMERIC_NAMESPACE_BEGIN

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_>
class _SimdMemoryAccess;

template <
    arch::CpuFeature	_SimdGeneration_,
    class				_RegisterPolicy_,
    typename            _DesiredType_,
    typename            _VectorType_>
simd_stl_always_inline _DesiredType_* _SimdCompressStoreUnaligned(
    _DesiredType_*                          _Where,
    type_traits::__deduce_simd_mask_type<_SimdGeneration_,
        _DesiredType_, _RegisterPolicy_>    _Mask,
    _VectorType_                            _Vector) noexcept;

template <
    arch::CpuFeature	_SimdGeneration_,
    class				_RegisterPolicy_,
    typename            _DesiredType_,
    typename            _VectorType_>
simd_stl_always_inline _DesiredType_* _SimdCompressStoreAligned(
    _DesiredType_*                          _Where,
    type_traits::__deduce_simd_mask_type<_SimdGeneration_,
        _DesiredType_, _RegisterPolicy_>    _Mask,
    const _VectorType_                      _Vector) noexcept;

#pragma region Sse2-Sse4.2 memory access 

template <>
class _SimdMemoryAccess<arch::CpuFeature::SSE2, numeric::xmm128> {
    static constexpr auto _Generation   = arch::CpuFeature::SSE2;
    using _RegisterPolicy               = numeric::xmm128;

    template <class _DesiredType_>
    using _Simd_mask_type = type_traits::__deduce_simd_mask_type<_Generation, _DesiredType_, _RegisterPolicy>;
public:
    template <sizetype _TypeSize_>
    static constexpr auto _Native_mask_load_supported = false;

    template <sizetype _TypeSize_>
    static constexpr auto _Native_mask_store_supported = false;

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _LoadUpperHalf(const void* _Where) noexcept;

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _LoadLowerHalf(const void* _Where) noexcept;

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _NonTemporalLoad(const void* _Where) noexcept;

    template <typename _VectorType_>
    static simd_stl_always_inline void _NonTemporalStore(
        void*           _Where,
        _VectorType_    _Vector) noexcept;

    static simd_stl_always_inline void _StreamingFence() noexcept;

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _LoadUnaligned(const void* _Where) noexcept;

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _LoadAligned(const void* _Where) noexcept;

    template <typename _VectorType_>
    static simd_stl_always_inline void _StoreUpperHalf(
        void*           _Where,
        _VectorType_    _Vector) noexcept;

    template <typename _VectorType_>
    static simd_stl_always_inline void _StoreLowerHalf(
        void* _Where,
        _VectorType_    _Vector) noexcept;

    template <typename _VectorType_>
    static simd_stl_always_inline void _StoreUnaligned(
        void*           _Where,
        _VectorType_    _Vector) noexcept;

    template <typename _VectorType_>
    static simd_stl_always_inline void _StoreAligned(
        void*           _Where,
        _VectorType_    _Vector) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline void _MaskStoreUnaligned(
        void*                                   _Where,
        const _Simd_mask_type<_DesiredType_>    _Mask,
        const _VectorType_                      _Vector) noexcept;
    
    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline void _MaskStoreAligned(
        void*                                   _Where,
        const _Simd_mask_type<_DesiredType_>    _Mask,
        const _VectorType_                      _Vector) noexcept;

    template <
        typename _DesiredType_,
        typename _MaskVectorType_,
        typename _VectorType_,
        std::enable_if_t<_Is_intrin_type_v<_MaskVectorType_>, int> = 0>
    static simd_stl_always_inline void _MaskStoreUnaligned(
        void*                   _Where,
        const _MaskVectorType_  _Mask,
        const _VectorType_      _Vector) noexcept;

    template <
        typename _DesiredType_,
        typename _MaskVectorType_,
        typename _VectorType_,
        std::enable_if_t<_Is_intrin_type_v<_MaskVectorType_>, int> = 0>
    static simd_stl_always_inline void _MaskStoreAligned(
        void*                   _Where,
        const _MaskVectorType_  _Mask,
        const _VectorType_      _Vector) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline void _MaskBlendStoreUnaligned(
        void*                                   _Where,
        const _Simd_mask_type<_DesiredType_>    _Mask,
        const _VectorType_                      _Vector,
        const _VectorType_                      _AdditionalSource) noexcept;
    
    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline void _MaskBlendStoreAligned(
        void*                                   _Where,
        const _Simd_mask_type<_DesiredType_>    _Mask,
        const _VectorType_                      _Vector,
        const _VectorType_                      _AdditionalSource) noexcept;

    template <
        typename _DesiredType_,
        typename _MaskVectorType_,
        typename _VectorType_,
        std::enable_if_t<_Is_intrin_type_v<_MaskVectorType_>, int> = 0>
    static simd_stl_always_inline void _MaskBlendStoreUnaligned(
        void*                   _Where,
        const _MaskVectorType_  _Mask,
        const _VectorType_      _Vector,
        const _VectorType_      _AdditionalSource) noexcept;
    
    template <
        typename _DesiredType_,
        typename _MaskVectorType_,
        typename _VectorType_,
        std::enable_if_t<_Is_intrin_type_v<_MaskVectorType_>, int> = 0>
    static simd_stl_always_inline void _MaskBlendStoreAligned(
        void*                   _Where,
        const _MaskVectorType_  _Mask,
        const _VectorType_      _Vector,
        const _VectorType_      _AdditionalSource) noexcept;

    template <
        typename _VectorType_,
        typename _DesiredType_>
    static simd_stl_always_inline _VectorType_ _MaskLoadUnaligned(
        const void*                             _Where,
        const _Simd_mask_type<_DesiredType_>    _Mask) noexcept;

    template <
        typename _VectorType_,
        typename _DesiredType_>
    static simd_stl_always_inline _VectorType_ _MaskLoadAligned(
        const void*                             _Where,
        const _Simd_mask_type<_DesiredType_>    _Mask) noexcept;

    template <
        typename _VectorType_,
        typename _DesiredType_,
        typename _MaskVectorType_,
        std::enable_if_t<_Is_intrin_type_v<_MaskVectorType_>, int> = 0>
    static simd_stl_always_inline _VectorType_ _MaskLoadUnaligned(
        const void*             _Where,
        const _MaskVectorType_  _Mask) noexcept;

    template <
        typename _VectorType_,
        typename _DesiredType_,
        typename _MaskVectorType_,
        std::enable_if_t<_Is_intrin_type_v<_MaskVectorType_>, int> = 0>
    static simd_stl_always_inline _VectorType_ _MaskLoadAligned(
        const void*             _Where,
        const _MaskVectorType_  _Mask) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _DesiredType_* _CompressStoreLowerHalf(
        _DesiredType_*                  _Where,
        _Simd_mask_type<_DesiredType_>  _Mask,
        _VectorType_                    _Vector) noexcept;
    
    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _DesiredType_* _CompressStoreUpperHalf(
        _DesiredType_*                  _Where,
        _Simd_mask_type<_DesiredType_>  _Mask,
        _VectorType_                    _Vector) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _DesiredType_* _CompressStoreUnaligned(
        _DesiredType_*                      _Where,
        _Simd_mask_type<_DesiredType_>      _Mask,
        _VectorType_                        _Vector) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _DesiredType_* _CompressStoreAligned(
        _DesiredType_*                      _Where,
        _Simd_mask_type<_DesiredType_>      _Mask,
        _VectorType_                        _Vector) noexcept;

    template <typename _Type_>
    static simd_stl_always_inline auto _MakeTailMask(uint32 bytes) noexcept;
};

template <>
class _SimdMemoryAccess<arch::CpuFeature::SSE3, numeric::xmm128> :
    public _SimdMemoryAccess<arch::CpuFeature::SSE2, numeric::xmm128>
{
public:
    template <sizetype _TypeSize_>
    static constexpr auto _Native_mask_load_supported = false;

    template <sizetype _TypeSize_>
    static constexpr auto _Native_mask_store_supported = false;
};

template <>
class _SimdMemoryAccess<arch::CpuFeature::SSSE3, numeric::xmm128> :
    public _SimdMemoryAccess<arch::CpuFeature::SSE3, numeric::xmm128>
{
    static constexpr auto _Generation   = arch::CpuFeature::SSE2;
    using _RegisterPolicy               = numeric::xmm128;

    template <class _DesiredType_>
    using _Simd_mask_type = type_traits::__deduce_simd_mask_type<_Generation, _DesiredType_, _RegisterPolicy>;
public:
    template <sizetype _TypeSize_>
    static constexpr auto _Native_mask_load_supported = false;

    template <sizetype _TypeSize_>
    static constexpr auto _Native_mask_store_supported = false;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _DesiredType_* _CompressStoreLowerHalf(
        _DesiredType_*                          _Where,
        const _Simd_mask_type<_DesiredType_>    _Mask,
        const _VectorType_                      _Vector) noexcept;
    
    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _DesiredType_* _CompressStoreUpperHalf(
        _DesiredType_*                          _Where,
        const _Simd_mask_type<_DesiredType_>    _Mask,
        const _VectorType_                      _Vector) noexcept;
    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _DesiredType_* _CompressStoreUnaligned(
        _DesiredType_*                          _Where,
        const _Simd_mask_type<_DesiredType_>    _Mask,
        const _VectorType_                      _Vector) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _DesiredType_* _CompressStoreAligned(
        _DesiredType_*                          _Where,
        const _Simd_mask_type<_DesiredType_>    _Mask,
        const _VectorType_                      _Vector) noexcept;
};

template <>
class _SimdMemoryAccess<arch::CpuFeature::SSE41, numeric::xmm128> :
    public _SimdMemoryAccess<arch::CpuFeature::SSSE3, numeric::xmm128>
{    
    static constexpr auto _Generation   = arch::CpuFeature::SSE41;
    using _RegisterPolicy               = numeric::xmm128;

    template <class _DesiredType_>
    using _Simd_mask_type = type_traits::__deduce_simd_mask_type<_Generation, _DesiredType_, _RegisterPolicy>;
public:
    template <sizetype _TypeSize_>
    static constexpr auto _Native_mask_load_supported = false;

    template <sizetype _TypeSize_>
    static constexpr auto _Native_mask_store_supported = false;

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _NonTemporalLoad(const void* where) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline void _MaskStoreUnaligned(
        void*                                   _Where,
        const _Simd_mask_type<_DesiredType_>    _Mask,
        const _VectorType_                      _Vector) noexcept;
    
    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline void _MaskStoreAligned(
        void*                                   _Where,
        const _Simd_mask_type<_DesiredType_>    _Mask,
        const _VectorType_                      _Vector) noexcept;

    template <
        typename _DesiredType_,
        typename _MaskVectorType_,
        typename _VectorType_,
        std::enable_if_t<_Is_intrin_type_v<_MaskVectorType_>, int> = 0>
    static simd_stl_always_inline void _MaskStoreUnaligned(
        void*                   _Where,
        const _MaskVectorType_  _Mask,
        const _VectorType_      _Vector) noexcept;
    
    template <
        typename _DesiredType_,
        typename _MaskVectorType_,
        typename _VectorType_,
        std::enable_if_t<_Is_intrin_type_v<_MaskVectorType_>, int> = 0>
    static simd_stl_always_inline void _MaskStoreAligned(
        void*                   _Where,
        const _MaskVectorType_  _Mask,
        const _VectorType_      _Vector) noexcept;

    template <
        typename _VectorType_,
        typename _DesiredType_>
    static simd_stl_always_inline _VectorType_ _MaskLoadUnaligned(
        const void*                             _Where,
        const _Simd_mask_type<_DesiredType_>    _Mask) noexcept;

    template <
        typename _VectorType_,
        typename _DesiredType_>
    static simd_stl_always_inline _VectorType_ _MaskLoadAligned(
        const void*                             _Where,
        const _Simd_mask_type<_DesiredType_>    _Mask) noexcept;

    template <
        typename _VectorType_,
        typename _DesiredType_,
        typename _MaskVectorType_,
        std::enable_if_t<_Is_intrin_type_v<_MaskVectorType_>, int> = 0>
    static simd_stl_always_inline _VectorType_ _MaskLoadUnaligned(
        const void*             _Where,
        const _MaskVectorType_  _Mask) noexcept;

    template <
        typename _VectorType_,
        typename _DesiredType_,
        typename _MaskVectorType_,
        std::enable_if_t<_Is_intrin_type_v<_MaskVectorType_>, int> = 0>
    static simd_stl_always_inline _VectorType_ _MaskLoadAligned(
        const void*             _Where,
        const _MaskVectorType_  _Mask) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline void _MaskBlendStoreUnaligned(
        void*                                   _Where,
        const _Simd_mask_type<_DesiredType_>    _Mask,
        const _VectorType_                      _Vector,
        const _VectorType_                      _AdditionalSource) noexcept;
    
    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline void _MaskBlendStoreAligned(
        void*                                   _Where,
        const _Simd_mask_type<_DesiredType_>    _Mask,
        const _VectorType_                      _Vector,
        const _VectorType_                      _AdditionalSource) noexcept;

    template <
        typename _DesiredType_,
        typename _MaskVectorType_,
        typename _VectorType_,
        std::enable_if_t<_Is_intrin_type_v<_MaskVectorType_>, int> = 0>
    static simd_stl_always_inline void _MaskBlendStoreUnaligned(
        void*                   _Where,
        const _MaskVectorType_  _Mask,
        const _VectorType_      _Vector,
        const _VectorType_      _AdditionalSource) noexcept;
    
    template <
        typename _DesiredType_,
        typename _MaskVectorType_,
        typename _VectorType_,
        std::enable_if_t<_Is_intrin_type_v<_MaskVectorType_>, int> = 0>
    static simd_stl_always_inline void _MaskBlendStoreAligned(
        void*                   _Where,
        const _MaskVectorType_  _Mask,
        const _VectorType_      _Vector,
        const _VectorType_      _AdditionalSource) noexcept;
};

template <>
class _SimdMemoryAccess<arch::CpuFeature::SSE42, numeric::xmm128> :
    public _SimdMemoryAccess<arch::CpuFeature::SSE41, numeric::xmm128>
{
public:
    template <sizetype _TypeSize_>
    static constexpr auto _Native_mask_load_supported = false;

    template <sizetype _TypeSize_>
    static constexpr auto _Native_mask_store_supported = false;
};

#pragma endregion

#pragma region Avx-Avx2 memory access

template <>
class _SimdMemoryAccess<arch::CpuFeature::AVX, numeric::ymm256>
{
    static constexpr auto _Generation   = arch::CpuFeature::AVX;
    using _RegisterPolicy               = numeric::ymm256;

    template <class _DesiredType_>
    using _Simd_mask_type = type_traits::__deduce_simd_mask_type<_Generation, _DesiredType_, _RegisterPolicy>;

    template <
        int32 _First_,
        int32 _Second_>
    static constexpr int32 _Max() noexcept {
        return (_First_ > _Second_) ? _First_ : _Second_;
    }

    template <sizetype _TypeSize_>
    struct _Native_mask_load_support:
        std::bool_constant<false> 
    {};

    template <>
    struct _Native_mask_load_support<4>:
        std::bool_constant<true> 
    {}; 

    template <>
    struct _Native_mask_load_support<8>:
        std::bool_constant<true>
    {}; 
public:
    template <sizetype _TypeSize_>
    static constexpr auto _Native_mask_load_supported = _Native_mask_load_support<_TypeSize_>::value;

    template <sizetype _TypeSize_>
    static constexpr auto _Native_mask_store_supported = _Native_mask_load_support<_TypeSize_>::value;

    template <typename _Type_>
    static simd_stl_always_inline auto _MakeTailMask(uint32 _Bytes) noexcept;

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _LoadUpperHalf(const void* _Where) noexcept;

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _LoadLowerHalf(const void* _Where) noexcept;

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _NonTemporalLoad(const void* _Where) noexcept;

    template <typename _VectorType_>
    static simd_stl_always_inline void _NonTemporalStore(
        void*           _Where,
        _VectorType_    _Vector) noexcept;

    static simd_stl_always_inline void _StreamingFence() noexcept;

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _LoadUnaligned(const void* _Where) noexcept;

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _LoadAligned(const void* _Where) noexcept;

    template <typename _VectorType_>
    static simd_stl_always_inline void _StoreUpperHalf(
        void*           _Where,
        _VectorType_    _Vector) noexcept;

    template <typename _VectorType_>
    static simd_stl_always_inline void _StoreLowerHalf(
        void*           _Where,
        _VectorType_    _Vector) noexcept;

    template <typename _VectorType_>
    static simd_stl_always_inline void _StoreUnaligned(
        void*           _Where,
        _VectorType_    _Vector) noexcept;

    template <typename _VectorType_>
    static simd_stl_always_inline void _StoreAligned(
        void*           _Where,
        _VectorType_    _Vector) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline void _MaskStoreUnaligned(
        void*                                   _Where,
        const _Simd_mask_type<_DesiredType_>    _Mask,
        const _VectorType_                      _Vector) noexcept;
    
    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline void _MaskStoreAligned(
        void*                                   _Where,
        const _Simd_mask_type<_DesiredType_>    _Mask,
        const _VectorType_                      _Vector) noexcept;

     template <
        typename _DesiredType_,
        typename _MaskVectorType_,
        typename _VectorType_,
        std::enable_if_t<_Is_intrin_type_v<_MaskVectorType_>, int> = 0>
    static simd_stl_always_inline void _MaskStoreUnaligned(
        void*                   _Where,
        const _MaskVectorType_  _Mask,
        const _VectorType_      _Vector) noexcept;
    
    template <
        typename _DesiredType_,
        typename _MaskVectorType_,
        typename _VectorType_,
        std::enable_if_t<_Is_intrin_type_v<_MaskVectorType_>, int> = 0>
    static simd_stl_always_inline void _MaskStoreAligned(
        void*                   _Where,
        const _MaskVectorType_  _Mask,
        const _VectorType_      _Vector) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline void _MaskBlendStoreUnaligned(
        void*                                   _Where,
        const _Simd_mask_type<_DesiredType_>    _Mask,
        const _VectorType_                      _Vector,
        const _VectorType_                      _AdditionalSource) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline void _MaskBlendStoreAligned(
        void*                                   _Where,
        const _Simd_mask_type<_DesiredType_>    _Mask,
        const _VectorType_                      _Vector,
        const _VectorType_                      _AdditionalSource) noexcept;

    template <
        typename _DesiredType_,
        typename _MaskVectorType_,
        typename _VectorType_,
        std::enable_if_t<_Is_intrin_type_v<_MaskVectorType_>, int> = 0>
    static simd_stl_always_inline void _MaskBlendStoreUnaligned(
        void*                   _Where,
        const _MaskVectorType_  _Mask,
        const _VectorType_      _Vector,
        const _VectorType_      _AdditionalSource) noexcept;
    
    template <
        typename _DesiredType_,
        typename _MaskVectorType_,
        typename _VectorType_,
        std::enable_if_t<_Is_intrin_type_v<_MaskVectorType_>, int> = 0>
    static simd_stl_always_inline void _MaskBlendStoreAligned(
        void*                   _Where,
        const _MaskVectorType_  _Mask,
        const _VectorType_      _Vector,
        const _VectorType_      _AdditionalSource) noexcept;

    template <
        typename _VectorType_,
        typename _DesiredType_>
    static simd_stl_always_inline _VectorType_ _MaskLoadUnaligned(
        const void*                             _Where,
        const _Simd_mask_type<_DesiredType_>    _Mask) noexcept;

    template <
        typename _VectorType_,
        typename _DesiredType_>
    static simd_stl_always_inline _VectorType_ _MaskLoadAligned(
        const void*                             _Where,
        const _Simd_mask_type<_DesiredType_>    _Mask) noexcept;

    template <
        typename _VectorType_,
        typename _DesiredType_,
        typename _MaskVectorType_,
        std::enable_if_t<_Is_intrin_type_v<_MaskVectorType_>, int> = 0>
    static simd_stl_always_inline _VectorType_ _MaskLoadUnaligned(
        const void*             _Where,
        const _MaskVectorType_  _Mask) noexcept;

    template <
        typename _VectorType_,
        typename _DesiredType_,
        typename _MaskVectorType_,
        std::enable_if_t<_Is_intrin_type_v<_MaskVectorType_>, int> = 0>
    static simd_stl_always_inline _VectorType_ _MaskLoadAligned(
        const void*             _Where,
        const _MaskVectorType_  _Mask) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _DesiredType_* _CompressStoreLowerHalf(
        _DesiredType_*                  _Where,
        _Simd_mask_type<_DesiredType_>  _Mask,
        _VectorType_                    _Vector) noexcept;
    
    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _DesiredType_* _CompressStoreUpperHalf(
        _DesiredType_*                  _Where,
        _Simd_mask_type<_DesiredType_>  _Mask,
        _VectorType_                    _Vector) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _DesiredType_* _CompressStoreUnaligned(
        _DesiredType_*                      _Where,
        _Simd_mask_type<_DesiredType_>      _Mask,
        _VectorType_                        _Vector) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _DesiredType_* _CompressStoreAligned(
        _DesiredType_*                      _Where,
        _Simd_mask_type<_DesiredType_>      _Mask,
        _VectorType_                        _Vector) noexcept;
};

template <>
class _SimdMemoryAccess<arch::CpuFeature::AVX2, numeric::ymm256>:
    public _SimdMemoryAccess<arch::CpuFeature::AVX, numeric::ymm256>
{
    static constexpr auto _Generation   = arch::CpuFeature::AVX2;
    using _RegisterPolicy               = numeric::ymm256;

    template <class _DesiredType_>
    using _Simd_mask_type = type_traits::__deduce_simd_mask_type<_Generation, _DesiredType_, _RegisterPolicy>;

    template <
        int32 _First_,
        int32 _Second_>
    static constexpr int32 _Max() noexcept {
        return (_First_ > _Second_) ? _First_ : _Second_;
    }
    
    template <sizetype _TypeSize_>
    struct _Native_mask_load_support :
        std::bool_constant<false>
    {};

    template <>
    struct _Native_mask_load_support<4> :
        std::bool_constant<true>
    {};

    template <>
    struct _Native_mask_load_support<8> :
        std::bool_constant<true>
    {};
public:
    template <sizetype _TypeSize_>
    static constexpr auto _Native_mask_load_supported = _Native_mask_load_support<_TypeSize_>::value;

    template <sizetype _TypeSize_>
    static constexpr auto _Native_mask_store_supported = _Native_mask_load_support<_TypeSize_>::value;

    template <
        typename _VectorType_,
        typename _DesiredType_>
    static simd_stl_always_inline _VectorType_ _MaskLoadUnaligned(
        const void*                             _Where,
        const _Simd_mask_type<_DesiredType_>    _Mask) noexcept;

    template <
        typename _VectorType_,
        typename _DesiredType_>
    static simd_stl_always_inline _VectorType_ _MaskLoadAligned(
        const void*                             _Where,
        const _Simd_mask_type<_DesiredType_>    _Mask) noexcept;

    template <
        typename _VectorType_,
        typename _DesiredType_,
        typename _MaskVectorType_,
        std::enable_if_t<_Is_intrin_type_v<_MaskVectorType_>, int> = 0>
    static simd_stl_always_inline _VectorType_ _MaskLoadUnaligned(
        const void*             _Where,
        const _MaskVectorType_  _Mask) noexcept;

    template <
        typename _VectorType_,
        typename _DesiredType_,
        typename _MaskVectorType_,
        std::enable_if_t<_Is_intrin_type_v<_MaskVectorType_>, int> = 0>
    static simd_stl_always_inline _VectorType_ _MaskLoadAligned(
        const void*             _Where,
        const _MaskVectorType_  _Mask) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline void _MaskBlendStoreUnaligned(
        void*                                   _Where,
        const _Simd_mask_type<_DesiredType_>    _Mask,
        const _VectorType_                      _Vector,
        const _VectorType_                      _AdditionalSource) noexcept;
    
    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline void _MaskBlendStoreAligned(
        void*                                   _Where,
        const _Simd_mask_type<_DesiredType_>    _Mask,
        const _VectorType_                      _Vector,
        const _VectorType_                      _AdditionalSource) noexcept;

    template <
        typename _DesiredType_,
        typename _MaskVectorType_,
        typename _VectorType_,
        std::enable_if_t<_Is_intrin_type_v<_MaskVectorType_>, int> = 0>
    static simd_stl_always_inline void _MaskBlendStoreUnaligned(
        void*                   _Where,
        const _MaskVectorType_  _Mask,
        const _VectorType_      _Vector,
        const _VectorType_      _AdditionalSource) noexcept;
    
    template <
        typename _DesiredType_,
        typename _MaskVectorType_,
        typename _VectorType_,
        std::enable_if_t<_Is_intrin_type_v<_MaskVectorType_>, int> = 0>
    static simd_stl_always_inline void _MaskBlendStoreAligned(
        void*                   _Where,
        const _MaskVectorType_  _Mask,
        const _VectorType_      _Vector,
        const _VectorType_      _AdditionalSource) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline void _MaskStoreUnaligned(
        void*                                   _Where,
        const _Simd_mask_type<_DesiredType_>    _Mask,
        const _VectorType_                      _Vector) noexcept;
    
    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline void _MaskStoreAligned(
        void*                                   _Where,
        const _Simd_mask_type<_DesiredType_>    _Mask,
        const _VectorType_                      _Vector) noexcept;

    template <
        typename _DesiredType_,
        typename _MaskVectorType_,
        typename _VectorType_,
        std::enable_if_t<_Is_intrin_type_v<_MaskVectorType_>, int> = 0>
    static simd_stl_always_inline void _MaskStoreUnaligned(
        void*                   _Where,
        const _MaskVectorType_  _Mask,
        const _VectorType_      _Vector) noexcept;
    
    template <
        typename _DesiredType_,
        typename _MaskVectorType_,
        typename _VectorType_,
        std::enable_if_t<_Is_intrin_type_v<_MaskVectorType_>, int> = 0>
    static simd_stl_always_inline void _MaskStoreAligned(
        void*                   _Where,
        const _MaskVectorType_  _Mask,
        const _VectorType_      _Vector) noexcept;

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _NonTemporalLoad(const void* _Where) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _DesiredType_* _CompressStoreUnaligned(
        _DesiredType_*                      _Where,
        _Simd_mask_type<_DesiredType_>      _Mask,
        _VectorType_                        _Vector) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _DesiredType_* _CompressStoreAligned(
        _DesiredType_*                      _Where,
        _Simd_mask_type<_DesiredType_>      _Mask,
        _VectorType_                        _Vector) noexcept;
};

#pragma endregion

#pragma region Avx512 memory access

template <>
class _SimdMemoryAccess<arch::CpuFeature::AVX512F, zmm512>
{
    static constexpr auto _Generation   = arch::CpuFeature::AVX512F;
    using _RegisterPolicy               = zmm512;

    template <class _DesiredType_>
    using _Simd_mask_type = type_traits::__deduce_simd_mask_type<_Generation, _DesiredType_, _RegisterPolicy>;

    template <sizetype _TypeSize_>
    struct _Native_mask_load_support :
        std::bool_constant<false>
    {};

    template <>
    struct _Native_mask_load_support<4> :
        std::bool_constant<true>
    {};

    template <>
    struct _Native_mask_load_support<8> :
        std::bool_constant<true>
    {};
public:
    template <sizetype _TypeSize_>
    static constexpr auto _Native_mask_load_supported = _Native_mask_load_support<_TypeSize_>::value;

    template <sizetype _TypeSize_>
    static constexpr auto _Native_mask_store_supported = _Native_mask_load_support<_TypeSize_>::value;

    template <typename _Type_>
    static simd_stl_always_inline auto _MakeTailMask(uint32 _Bytes) noexcept;

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _LoadUpperHalf(const void* _Where) noexcept;

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _LoadLowerHalf(const void* _Where) noexcept;

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _NonTemporalLoad(const void* _Where) noexcept;

    template <typename _VectorType_>
    static simd_stl_always_inline void _NonTemporalStore(
        void*           _Where,
        _VectorType_    _Vector) noexcept;

    static simd_stl_always_inline void _StreamingFence() noexcept;

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _LoadUnaligned(const void* _Where) noexcept;

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _LoadAligned(const void* _Where) noexcept;

    template <typename _VectorType_>
    static simd_stl_always_inline void _StoreUpperHalf(
        void*           _Where,
        _VectorType_    _Vector) noexcept;

    template <typename _VectorType_>
    static simd_stl_always_inline void _StoreLowerHalf(
        void*           _Where,
        _VectorType_    _Vector) noexcept;

    template <typename _VectorType_>
    static simd_stl_always_inline void _StoreUnaligned(
        void*           _Where,
        _VectorType_    _Vector) noexcept;

    template <typename _VectorType_>
    static simd_stl_always_inline void _StoreAligned(
        void*           _Where,
        _VectorType_    _Vector) noexcept;

     template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline void _MaskStoreUnaligned(
        void*                                   _Where,
        const _Simd_mask_type<_DesiredType_>    _Mask,
        const _VectorType_                      _Vector) noexcept;
    
    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline void _MaskStoreAligned(
        void*                                   _Where,
        const _Simd_mask_type<_DesiredType_>    _Mask,
        const _VectorType_                      _Vector) noexcept;

    template <
        typename _DesiredType_,
        typename _MaskVectorType_,
        typename _VectorType_,
        std::enable_if_t<_Is_intrin_type_v<_MaskVectorType_>, int> = 0>
    static simd_stl_always_inline void _MaskStoreUnaligned(
        void*                   _Where,
        const _MaskVectorType_  _Mask,
        const _VectorType_      _Vector) noexcept;
    
    template <
        typename _DesiredType_,
        typename _MaskVectorType_,
        typename _VectorType_,
        std::enable_if_t<_Is_intrin_type_v<_MaskVectorType_>, int> = 0>
    static simd_stl_always_inline void _MaskStoreAligned(
        void*                   _Where,
        const _MaskVectorType_  _Mask,
        const _VectorType_      _Vector) noexcept;

    template <
        typename _VectorType_,
        typename _DesiredType_>
    static simd_stl_always_inline _VectorType_ _MaskLoadUnaligned(
        const void*                             _Where,
        const _Simd_mask_type<_DesiredType_>    _Mask) noexcept;

    template <
        typename _VectorType_,
        typename _DesiredType_>
    static simd_stl_always_inline _VectorType_ _MaskLoadAligned(
        const void*                             _Where,
        const _Simd_mask_type<_DesiredType_>    _Mask) noexcept;

    template <
        typename _VectorType_,
        typename _DesiredType_,
        typename _MaskVectorType_,
        std::enable_if_t<_Is_intrin_type_v<_MaskVectorType_>, int> = 0>
    static simd_stl_always_inline _VectorType_ _MaskLoadUnaligned(
        const void*             _Where,
        const _MaskVectorType_  _Mask) noexcept;

    template <
        typename _VectorType_,
        typename _DesiredType_,
        typename _MaskVectorType_,
        std::enable_if_t<_Is_intrin_type_v<_MaskVectorType_>, int> = 0>
    static simd_stl_always_inline _VectorType_ _MaskLoadAligned(
        const void*             _Where,
        const _MaskVectorType_  _Mask) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline void _MaskBlendStoreUnaligned(
        void*                                   _Where,
        const _Simd_mask_type<_DesiredType_>    _Mask,
        const _VectorType_                      _Vector,
        const _VectorType_                      _AdditionalSource) noexcept;
    
    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline void _MaskBlendStoreAligned(
        void*                                   _Where,
        const _Simd_mask_type<_DesiredType_>    _Mask,
        const _VectorType_                      _Vector,
        const _VectorType_                      _AdditionalSource) noexcept;

    template <
        typename _DesiredType_,
        typename _MaskVectorType_,
        typename _VectorType_,
        std::enable_if_t<_Is_intrin_type_v<_MaskVectorType_>, int> = 0>
    static simd_stl_always_inline void _MaskBlendStoreUnaligned(
        void*                   _Where,
        const _MaskVectorType_  _Mask,
        const _VectorType_      _Vector,
        const _VectorType_      _AdditionalSource) noexcept;
    
    template <
        typename _DesiredType_,
        typename _MaskVectorType_,
        typename _VectorType_,
        std::enable_if_t<_Is_intrin_type_v<_MaskVectorType_>, int> = 0>
    static simd_stl_always_inline void _MaskBlendStoreAligned(
        void*                   _Where,
        const _MaskVectorType_  _Mask,
        const _VectorType_      _Vector,
        const _VectorType_      _AdditionalSource) noexcept;
    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _DesiredType_* _CompressStoreUnaligned(
        _DesiredType_*                      _Where,
        _Simd_mask_type<_DesiredType_>      _Mask,
        _VectorType_                        _Vector) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _DesiredType_* _CompressStoreAligned(
        _DesiredType_*                      _Where,
        _Simd_mask_type<_DesiredType_>      _Mask,
        _VectorType_                        _Vector) noexcept;
};

template <>
class _SimdMemoryAccess<arch::CpuFeature::AVX512BW, zmm512>:
    public _SimdMemoryAccess<arch::CpuFeature::AVX512F, zmm512>
{
    static constexpr auto _Generation = arch::CpuFeature::AVX512BW;
    using _RegisterPolicy = zmm512;

    template <class _DesiredType_>
    using _Simd_mask_type = type_traits::__deduce_simd_mask_type<_Generation, _DesiredType_, _RegisterPolicy>;
public:
    template <sizetype _TypeSize_>
    static constexpr auto _Native_mask_load_supported = true;

    template <sizetype _TypeSize_>
    static constexpr auto _Native_mask_store_supported = true;

    template <typename _Type_>
    static simd_stl_always_inline auto _MakeTailMask(uint32 _Bytes) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline void _MaskBlendStoreUnaligned(
        void*                                   _Where,
        const _Simd_mask_type<_DesiredType_>    _Mask,
        const _VectorType_                      _Vector,
        const _VectorType_                      _AdditionalSource) noexcept;
    
    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline void _MaskBlendStoreAligned(
        void*                                   _Where,
        const _Simd_mask_type<_DesiredType_>    _Mask,
        const _VectorType_                      _Vector,
        const _VectorType_                      _AdditionalSource) noexcept;

    template <
        typename _DesiredType_,
        typename _MaskVectorType_,
        typename _VectorType_,
        std::enable_if_t<_Is_intrin_type_v<_MaskVectorType_>, int> = 0>
    static simd_stl_always_inline void _MaskBlendStoreUnaligned(
        void*                   _Where,
        const _MaskVectorType_  _Mask,
        const _VectorType_      _Vector,
        const _VectorType_      _AdditionalSource) noexcept;
    
    template <
        typename _DesiredType_,
        typename _MaskVectorType_,
        typename _VectorType_,
        std::enable_if_t<_Is_intrin_type_v<_MaskVectorType_>, int> = 0>
    static simd_stl_always_inline void _MaskBlendStoreAligned(
        void*                   _Where,
        const _MaskVectorType_  _Mask,
        const _VectorType_      _Vector,
        const _VectorType_      _AdditionalSource) noexcept;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline void _MaskStoreUnaligned(
        void*                                   _Where,
        const _Simd_mask_type<_DesiredType_>    _Mask,
        const _VectorType_                      _Vector) noexcept;
    
    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline void _MaskStoreAligned(
        void*                                   _Where,
        const _Simd_mask_type<_DesiredType_>    _Mask,
        const _VectorType_                      _Vector) noexcept;

    template <
        typename _DesiredType_,
        typename _MaskVectorType_,
        typename _VectorType_,
        std::enable_if_t<_Is_intrin_type_v<_MaskVectorType_>, int> = 0>
    static simd_stl_always_inline void _MaskStoreUnaligned(
        void*                   _Where,
        const _MaskVectorType_  _Mask,
        const _VectorType_      _Vector) noexcept;
    
    template <
        typename _DesiredType_,
        typename _MaskVectorType_,
        typename _VectorType_,
        std::enable_if_t<_Is_intrin_type_v<_MaskVectorType_>, int> = 0>
    static simd_stl_always_inline void _MaskStoreAligned(
        void*                   _Where,
        const _MaskVectorType_  _Mask,
        const _VectorType_      _Vector) noexcept;

    template <
        typename _VectorType_,
        typename _DesiredType_>
    static simd_stl_always_inline _VectorType_ _MaskLoadUnaligned(
        const void*                             _Where,
        const _Simd_mask_type<_DesiredType_>    _Mask) noexcept;

    template <
        typename _VectorType_,
        typename _DesiredType_>
    static simd_stl_always_inline _VectorType_ _MaskLoadAligned(
        const void*                             _Where,
        const _Simd_mask_type<_DesiredType_>    _Mask) noexcept;

    template <
        typename _VectorType_,
        typename _DesiredType_,
        typename _MaskVectorType_,
        std::enable_if_t<_Is_intrin_type_v<_MaskVectorType_>, int> = 0>
    static simd_stl_always_inline _VectorType_ _MaskLoadUnaligned(
        const void*             _Where,
        const _MaskVectorType_  _Mask) noexcept;

    template <
        typename _VectorType_,
        typename _DesiredType_,
        typename _MaskVectorType_,
        std::enable_if_t<_Is_intrin_type_v<_MaskVectorType_>, int> = 0>
    static simd_stl_always_inline _VectorType_ _MaskLoadAligned(
        const void*             _Where,
        const _MaskVectorType_  _Mask) noexcept;
};

template <>
class _SimdMemoryAccess<arch::CpuFeature::AVX512DQ, zmm512> :
    public _SimdMemoryAccess<arch::CpuFeature::AVX512BW, zmm512>
{};

template <>
class _SimdMemoryAccess<arch::CpuFeature::AVX512VL, zmm512> :
    public _SimdMemoryAccess<arch::CpuFeature::AVX512DQ, zmm512>
{};

#pragma endregion

template <
    arch::CpuFeature	_SimdGeneration_,
    class				_RegisterPolicy_,
    class				_VectorType_>
simd_stl_nodiscard simd_stl_always_inline _VectorType_ _SimdLoadUnaligned(const void* _Where) noexcept {
    _VerifyRegisterPolicy(_SimdGeneration_, _RegisterPolicy_);
    return _SimdMemoryAccess<_SimdGeneration_, _RegisterPolicy_>::template _LoadUnaligned<_VectorType_>(_Where);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    class				_RegisterPolicy_,
    class				_VectorType_>
simd_stl_nodiscard simd_stl_always_inline _VectorType_ _SimdLoadAligned(const void* _Where) noexcept {
    _VerifyRegisterPolicy(_SimdGeneration_, _RegisterPolicy_);
    return _SimdMemoryAccess<_SimdGeneration_, _RegisterPolicy_>::template _LoadAligned<_VectorType_>(_Where);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    class				_RegisterPolicy_,
    class				_VectorType_>
simd_stl_nodiscard simd_stl_always_inline _VectorType_ _SimdNonTemporalLoad(const void* _Where) noexcept {
    _VerifyRegisterPolicy(_SimdGeneration_, _RegisterPolicy_);
    return _SimdMemoryAccess<_SimdGeneration_, _RegisterPolicy_>::template _NonTemporalLoad<_VectorType_>(_Where);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    class				_RegisterPolicy_,
    class				_VectorType_>
simd_stl_nodiscard simd_stl_always_inline _VectorType_ _SimdLoadUpperHalf(const void* _Where) noexcept {
    _VerifyRegisterPolicy(_SimdGeneration_, _RegisterPolicy_);
    return _SimdMemoryAccess<_SimdGeneration_, _RegisterPolicy_>::template _LoadUpperHalf<_VectorType_>(_Where);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    class				_RegisterPolicy_,
    class				_VectorType_>
simd_stl_nodiscard simd_stl_always_inline _VectorType_ _SimdLoadLowerHalf(const void* _Where) noexcept {
    _VerifyRegisterPolicy(_SimdGeneration_, _RegisterPolicy_);
    return _SimdMemoryAccess<_SimdGeneration_, _RegisterPolicy_>::template _LoadLowerHalf<_VectorType_>(_Where);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    class				_RegisterPolicy_,
    typename            _VectorType_,
    typename            _DesiredType_>
simd_stl_always_inline _VectorType_ _SimdMaskLoadUnaligned(
    const void*                                             _Where,
    const type_traits::__deduce_simd_mask_type<
        _SimdGeneration_, _DesiredType_, _RegisterPolicy_>  _Mask) noexcept
{
    _VerifyRegisterPolicy(_SimdGeneration_, _RegisterPolicy_);
    return _SimdMemoryAccess<_SimdGeneration_, _RegisterPolicy_>::template _MaskLoadUnaligned<_VectorType_, _DesiredType_>(_Where, _Mask);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    class				_RegisterPolicy_,
    typename            _VectorType_,
    typename            _DesiredType_,
    typename            _VectorMaskType_,
    std::enable_if_t<_Is_intrin_type_v<_VectorMaskType_>, int> = 0>
simd_stl_always_inline _VectorType_ _SimdMaskLoadUnaligned(
    const void*             _Where,
    const _VectorMaskType_  _Mask) noexcept
{
    _VerifyRegisterPolicy(_SimdGeneration_, _RegisterPolicy_);
    return _SimdMemoryAccess<_SimdGeneration_, _RegisterPolicy_>::template _MaskLoadUnaligned<_VectorType_, _DesiredType_, _VectorMaskType_>(_Where, _Mask);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    class				_RegisterPolicy_,
    typename            _VectorType_,
    typename            _DesiredType_,
    typename            _VectorMaskType_,
    std::enable_if_t<_Is_intrin_type_v<_VectorMaskType_>, int> = 0>
simd_stl_always_inline _VectorType_ _SimdMaskLoadAligned(
    const void*             _Where,
    const _VectorMaskType_  _Mask) noexcept
{
    _VerifyRegisterPolicy(_SimdGeneration_, _RegisterPolicy_);
    return _SimdMemoryAccess<_SimdGeneration_, _RegisterPolicy_>::template _MaskLoadAligned<_VectorType_, _DesiredType_, _VectorMaskType_>(_Where, _Mask);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    class				_RegisterPolicy_,
    typename            _VectorType_,
    typename            _DesiredType_>
simd_stl_always_inline _VectorType_ _SimdMaskLoadAligned(
    const void*                                             _Where,
    const type_traits::__deduce_simd_mask_type<
        _SimdGeneration_, _DesiredType_, _RegisterPolicy_>  _Mask) noexcept
{
    _VerifyRegisterPolicy(_SimdGeneration_, _RegisterPolicy_);
    return _SimdMemoryAccess<_SimdGeneration_, _RegisterPolicy_>::template _MaskLoadAligned<_VectorType_, _DesiredType_>(_Where, _Mask);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    class				_RegisterPolicy_,
    class				_VectorType_>
simd_stl_nodiscard simd_stl_always_inline void _SimdStoreUnaligned(
    void*               _Where,
    const _VectorType_  _Vector) noexcept
{
    _VerifyRegisterPolicy(_SimdGeneration_, _RegisterPolicy_);
    _SimdMemoryAccess<_SimdGeneration_, _RegisterPolicy_>::template _StoreUnaligned(_Where, _Vector);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    class				_RegisterPolicy_,
    class				_VectorType_>
simd_stl_nodiscard simd_stl_always_inline void _SimdStoreAligned(
    void*               _Where,
    const _VectorType_  _Vector) noexcept
{
    _VerifyRegisterPolicy(_SimdGeneration_, _RegisterPolicy_);
    _SimdMemoryAccess<_SimdGeneration_, _RegisterPolicy_>::template _StoreAligned(_Where, _Vector);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    class				_RegisterPolicy_,
    class				_VectorType_>
simd_stl_nodiscard simd_stl_always_inline void _SimdStoreUpperHalf(
    void*               _Where,
    const _VectorType_  _Vector) noexcept
{
    _VerifyRegisterPolicy(_SimdGeneration_, _RegisterPolicy_);
    _SimdMemoryAccess<_SimdGeneration_, _RegisterPolicy_>::template _StoreUpperHalf(_Where, _Vector);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    class				_RegisterPolicy_,
    class				_VectorType_>
simd_stl_nodiscard simd_stl_always_inline void _SimdStoreLowerHalf(
    void*               _Where,
    const _VectorType_  _Vector) noexcept
{
    _VerifyRegisterPolicy(_SimdGeneration_, _RegisterPolicy_);
    _SimdMemoryAccess<_SimdGeneration_, _RegisterPolicy_>::template _StoreLowerHalf(_Where, _Vector);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    class				_RegisterPolicy_,
    class				_VectorType_>
simd_stl_nodiscard simd_stl_always_inline void _SimdNonTemporalStore(
    void*               _Where,
    const _VectorType_  _Vector) noexcept
{
    _VerifyRegisterPolicy(_SimdGeneration_, _RegisterPolicy_);
    _SimdMemoryAccess<_SimdGeneration_, _RegisterPolicy_>::template _NonTemporalStore(_Where, _Vector);
}

template <arch::CpuFeature _SimdGeneration_>
simd_stl_nodiscard simd_stl_always_inline void _SimdStreamingFence() noexcept {
    _SimdMemoryAccess<_SimdGeneration_, _DefaultRegisterPolicy<_SimdGeneration_>>::_StreamingFence();
}

template <
    arch::CpuFeature	_SimdGeneration_,
    class				_RegisterPolicy_,
    class               _DesiredType_,
    class               _VectorType_>
simd_stl_always_inline void _SimdMaskStoreUnaligned(
    void*                                                   _Where,
    const type_traits::__deduce_simd_mask_type<
        _SimdGeneration_, _DesiredType_, _RegisterPolicy_>  _Mask,
    const _VectorType_                                      _Vector) noexcept
{
    _VerifyRegisterPolicy(_SimdGeneration_, _RegisterPolicy_);
    _SimdMemoryAccess<_SimdGeneration_, _RegisterPolicy_>::template _MaskStoreUnaligned<_DesiredType_>(_Where, _Mask, _Vector);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    class				_RegisterPolicy_,
    class               _DesiredType_,
    class               _VectorType_>
simd_stl_always_inline void _SimdMaskStoreAligned(
    void*                                                   _Where,
    const type_traits::__deduce_simd_mask_type<
        _SimdGeneration_, _DesiredType_, _RegisterPolicy_>  _Mask,
    const _VectorType_                                      _Vector) noexcept
{
    _VerifyRegisterPolicy(_SimdGeneration_, _RegisterPolicy_);
    _SimdMemoryAccess<_SimdGeneration_, _RegisterPolicy_>::template _MaskStoreAligned<_DesiredType_>(_Where, _Mask, _Vector);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    class				_RegisterPolicy_,
    class               _DesiredType_,
    class               _MaskVectorType_,
    class               _VectorType_,
    std::enable_if_t<_Is_intrin_type_v<_MaskVectorType_>, int> = 0>
simd_stl_always_inline void _SimdMaskStoreUnaligned(
    void*                   _Where,
    const _MaskVectorType_  _Mask,
    const _VectorType_      _Vector) noexcept
{
    _VerifyRegisterPolicy(_SimdGeneration_, _RegisterPolicy_);
    _SimdMemoryAccess<_SimdGeneration_, _RegisterPolicy_>::template _MaskStoreUnaligned<_DesiredType_>(_Where, _Mask, _Vector);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    class				_RegisterPolicy_,
    class               _DesiredType_,
    class               _MaskVectorType_,
    class               _VectorType_,
    std::enable_if_t<_Is_intrin_type_v<_MaskVectorType_>, int> = 0>
simd_stl_always_inline void _SimdMaskStoreAligned(
    void*                   _Where,
    const _MaskVectorType_  _Mask,
    const _VectorType_      _Vector) noexcept
{
    _VerifyRegisterPolicy(_SimdGeneration_, _RegisterPolicy_);
    _SimdMemoryAccess<_SimdGeneration_, _RegisterPolicy_>::template _MaskStoreAligned<_DesiredType_>(_Where, _Mask, _Vector);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    class				_RegisterPolicy_,
    class               _DesiredType_,
    class               _VectorType_>
simd_stl_always_inline void _SimdMaskBlendStoreUnaligned(
    void*                                                   _Where,
    const type_traits::__deduce_simd_mask_type<
        _SimdGeneration_, _DesiredType_, _RegisterPolicy_>  _Mask,
    const _VectorType_                                      _Vector,
    const _VectorType_                                      _AdditionalSource) noexcept
{
    _VerifyRegisterPolicy(_SimdGeneration_, _RegisterPolicy_);
    _SimdMemoryAccess<_SimdGeneration_, _RegisterPolicy_>::template _MaskBlendStoreUnaligned<_DesiredType_>(
        _Where, _Mask, _Vector, _AdditionalSource);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    class				_RegisterPolicy_,
    class               _DesiredType_,
    class               _VectorType_>
simd_stl_always_inline void _SimdMaskBlendStoreAligned(
    void*                                                   _Where,
    const type_traits::__deduce_simd_mask_type<
        _SimdGeneration_, _DesiredType_, _RegisterPolicy_>  _Mask,
    const _VectorType_                                      _Vector,
    const _VectorType_                                      _AdditionalSource) noexcept
{
    _VerifyRegisterPolicy(_SimdGeneration_, _RegisterPolicy_);
    _SimdMemoryAccess<_SimdGeneration_, _RegisterPolicy_>::template _MaskBlendStoreAligned<_DesiredType_>(
        _Where, _Mask, _Vector, _AdditionalSource);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    class				_RegisterPolicy_,
    class               _DesiredType_,
    class               _MaskVectorType_,
    class               _VectorType_,
    std::enable_if_t<_Is_intrin_type_v<_MaskVectorType_>, int> = 0>
simd_stl_always_inline void _SimdMaskBlendStoreUnaligned(
    void*                   _Where,
    const _MaskVectorType_  _Mask,
    const _VectorType_      _Vector,
    const _VectorType_      _AdditionalSource) noexcept
{
    _VerifyRegisterPolicy(_SimdGeneration_, _RegisterPolicy_);
    _SimdMemoryAccess<_SimdGeneration_, _RegisterPolicy_>::template _MaskBlendStoreUnaligned<_DesiredType_>(
        _Where, _Mask, _Vector, _AdditionalSource);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    class				_RegisterPolicy_,
    class               _DesiredType_,
    class               _MaskVectorType_,
    class               _VectorType_,
    std::enable_if_t<_Is_intrin_type_v<_MaskVectorType_>, int> = 0>
simd_stl_always_inline void _SimdMaskBlendStoreAligned(
    void*                   _Where,
    const _MaskVectorType_  _Mask,
    const _VectorType_      _Vector,
    const _VectorType_      _AdditionalSource) noexcept
{
    _VerifyRegisterPolicy(_SimdGeneration_, _RegisterPolicy_);
    _SimdMemoryAccess<_SimdGeneration_, _RegisterPolicy_>::template _MaskBlendStoreAligned<_DesiredType_>(
        _Where, _Mask, _Vector, _AdditionalSource);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    class				_RegisterPolicy_,
    typename            _DesiredType_,
    typename            _VectorType_>
simd_stl_always_inline _DesiredType_* _SimdCompressStoreUnaligned(
    _DesiredType_*                          _Where,
    type_traits::__deduce_simd_mask_type<_SimdGeneration_,
        _DesiredType_, _RegisterPolicy_>    _Mask,
    const _VectorType_                      _Vector) noexcept
{
    _VerifyRegisterPolicy(_SimdGeneration_, _RegisterPolicy_);
    return _SimdMemoryAccess<_SimdGeneration_, _RegisterPolicy_>::template _CompressStoreUnaligned(_Where, _Mask, _Vector);
}

template <
    arch::CpuFeature	_SimdGeneration_,
    class				_RegisterPolicy_,
    typename            _DesiredType_,
    typename            _VectorType_>
simd_stl_always_inline _DesiredType_* _SimdCompressStoreAligned(
    _DesiredType_*                          _Where,
    type_traits::__deduce_simd_mask_type<_SimdGeneration_,
        _DesiredType_, _RegisterPolicy_>    _Mask,
    const _VectorType_                      _Vector) noexcept
{
    _VerifyRegisterPolicy(_SimdGeneration_, _RegisterPolicy_);
    return _SimdMemoryAccess<_SimdGeneration_, _RegisterPolicy_>::template _CompressStoreAligned(_Where, _Mask, _Vector);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_,
    typename            _Type_>
constexpr inline bool _Is_native_mask_load_supported_v = _SimdMemoryAccess<_SimdGeneration_, _RegisterPolicy_>
    ::template _Native_mask_load_supported<sizeof(_Type_)>;

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_,
    typename            _Type_>
constexpr inline bool _Is_native_mask_store_supported_v = _SimdMemoryAccess<_SimdGeneration_, _RegisterPolicy_>
    ::template _Native_mask_store_supported<sizeof(_Type_)>;

template <
    arch::CpuFeature	_SimdGeneration_,
    class               _RegisterPolicy_,
    typename            _Type_>
simd_stl_always_inline auto _SimdMakeTailMask(uint32 _Bytes) noexcept {
    return _SimdMemoryAccess<_SimdGeneration_, _RegisterPolicy_>::template _MakeTailMask<_Type_>(_Bytes);
}

template <
    class		_BasicSimd_,
    typename	_ReturnType_>
using _Make_tail_mask_return_type_helper = std::conditional_t<_Is_intrin_type_v<_ReturnType_>,
    simd<_BasicSimd_::_Generation, typename _BasicSimd_::value_type, typename _BasicSimd_::policy_type>, _ReturnType_>;

template <class _BasicSimd_>
using _Make_tail_mask_return_type = _Make_tail_mask_return_type_helper<_BasicSimd_,
    type_traits::invoke_result_type<decltype(_SimdMakeTailMask<_BasicSimd_::_Generation, typename _BasicSimd_::policy_type,
    typename _BasicSimd_::value_type>), uint32>>;

template <
    arch::CpuFeature	_SimdGeneration_,
    class               _RegisterPolicy_,
    typename            _MaskOrBasicSimdType_>
simd_stl_always_inline auto _SimdToNativeMask(_MaskOrBasicSimdType_ _Mask) noexcept {
    if constexpr (_Is_valid_basic_simd_v<_MaskOrBasicSimdType_>)
        return _Mask.toMask();
    else
        return _Mask;
}

__SIMD_STL_NUMERIC_NAMESPACE_END

#include <src/simd_stl/numeric/SimdMemoryAccess.inl>