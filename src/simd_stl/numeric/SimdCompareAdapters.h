#pragma once 

#include <src/simd_stl/numeric/SimdConvert.h>
#include <simd_stl/numeric/SimdIndexMask.h>

__SIMD_STL_NUMERIC_NAMESPACE_BEGIN

class __as_mask_t {
    template <class _CompareResult_>
    using __simd_mask_type_helper = simd_mask<_CompareResult_::__generation, typename _CompareResult_::element_type, typename _CompareResult_::register_policy>;

    template <class _CompareResult_>
    using __simd_mask_type = __simd_mask_type_helper<std::remove_cvref_t<_CompareResult_>>;
public:
    template <class _CompareResult_>
    simd_stl_always_inline __simd_mask_type<_CompareResult_> operator()(_CompareResult_&& __compare_result) const noexcept;
};

class __as_index_mask_t {
    template <class _CompareResult_>
    using __simd_index_mask_type_helper = simd_index_mask<_CompareResult_::__generation, typename _CompareResult_::element_type, typename _CompareResult_::register_policy>;

    template <class _CompareResult_>
    using __simd_index_mask_type = __simd_index_mask_type_helper<std::remove_cvref_t<_CompareResult_>>;
public:
    template <class _CompareResult_>
    simd_stl_always_inline __simd_index_mask_type<_CompareResult_> operator()(_CompareResult_&& __compare_result) const noexcept;
};

class __as_simd_t {
    template <class _CompareResult_>
    using __simd_type_helper = simd<_CompareResult_::__generation, typename _CompareResult_::element_type, typename _CompareResult_::register_policy>;

    template <class _CompareResult_>
    using __simd_type = __simd_type_helper<std::remove_cvref_t<_CompareResult_>>;
public:
    template <class _CompareResult_>
    simd_stl_always_inline __simd_type<_CompareResult_> operator()(_CompareResult_&& __compare_result) const noexcept;
};

class __as_native_t {
    template <class _CompareResult_>
    using __native_type = typename std::remove_cvref_t<_CompareResult_>::native_type;


public:
    template <class _CompareResult_>
    simd_stl_always_inline __native_type<_CompareResult_> operator()(_CompareResult_&& __compare_result) const noexcept;
};

constexpr inline __as_mask_t        as_mask       {};
constexpr inline __as_index_mask_t  as_index_mask {};
constexpr inline __as_simd_t        as_simd       {};
constexpr inline __as_native_t      as_native     {};

template <
    class _CompareResult_, 
    class _Adapter_>
requires requires(
    _Adapter_       __adapter, 
    _CompareResult_&& __compare_result) 
{
    __adapter(__compare_result);
}
auto operator|(
    _CompareResult_&&   __compare_result, 
    const _Adapter_&    __adapter) noexcept
{
    return __adapter(std::forward<_CompareResult_>(__compare_result));
}

template <
    class _Adapter_,
    class _CompareResult_>
requires requires(
    _Adapter_       __adapter, 
    _CompareResult_&& __compare_result) 
{
    __adapter(__compare_result); 
}
auto operator|(
    const _Adapter_&    __adapter,
    _CompareResult_&&   __compare_result) noexcept 
{
    return __adapter(std::forward<_CompareResult_>(__compare_result));
}

__SIMD_STL_NUMERIC_NAMESPACE_END

#include <src/simd_stl/numeric/SimdCompareAdapters.inl>