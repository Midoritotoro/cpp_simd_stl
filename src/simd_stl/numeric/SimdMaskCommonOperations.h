#pragma once 

#include <simd_stl/compatibility/Compatibility.h>
#include <src/simd_stl/type_traits/Invoke.h>

#include <src/simd_stl/numeric/SimdMaskTypeCheck.h>


__SIMD_STL_NUMERIC_NAMESPACE_BEGIN

template <class _Derived_>
struct __simd_mask_common_operations {
    using __self			= std::remove_cvref_t<_Derived_>;

    static constexpr simd_stl_always_inline arch::CpuFeature __generation() noexcept;
    static constexpr simd_stl_always_inline uint8 __bit_width() noexcept;
    constexpr simd_stl_always_inline auto __unwrap() const noexcept;

    constexpr simd_stl_always_inline bool all_of() const noexcept;
    constexpr simd_stl_always_inline bool any_of() const noexcept;
    constexpr simd_stl_always_inline bool none_of() const noexcept;

    constexpr simd_stl_always_inline __self operator&(const __self& __other) const noexcept;
    constexpr simd_stl_always_inline __self operator|(const __self& __other) const noexcept;
    constexpr simd_stl_always_inline __self operator^(const __self& __other) const noexcept;

    constexpr simd_stl_always_inline __self operator>>(const uint8 __shift) const noexcept;
    constexpr simd_stl_always_inline __self operator<<(const uint8 __shift) const noexcept;
    
    template <uint8 _Shift_>
    constexpr simd_stl_always_inline __self operator>>(const std::integral_constant<uint8, _Shift_> __shift) const noexcept;

    template <uint8 _Shift_>
    constexpr simd_stl_always_inline __self operator<<(const std::integral_constant<uint8, _Shift_> __shift) const noexcept;

    constexpr simd_stl_always_inline __self operator~() const noexcept;

    constexpr simd_stl_always_inline auto __to_kmask() const noexcept;
    constexpr simd_stl_always_inline auto __to_int() const noexcept;

    constexpr simd_stl_always_inline explicit operator bool() const noexcept;

    constexpr simd_stl_always_inline bool operator==(const __self& __other) const noexcept;
    constexpr simd_stl_always_inline bool operator!=(const __self& __other) const noexcept;
};

__SIMD_STL_NUMERIC_NAMESPACE_END

#include <src/simd_stl/numeric/SimdMaskCommonOperations.inl>
