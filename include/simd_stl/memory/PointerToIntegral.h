#pragma once 

#include <simd_stl/Types.h>

__SIMD_STL_MEMORY_NAMESPACE_BEGIN

template <class _Type_>
constexpr auto pointerToIntegral(_Type_ _Pointer) noexcept {
    if      constexpr (std::is_same_v<std::decay_t<_Type_>, std::nullptr_t>)
        return 0;
    else if constexpr (std::is_pointer_v<std::decay_t<_Type_>>)
        return reinterpret_cast<uintptr>(_Pointer);
    else
        return _Pointer;
}

__SIMD_STL_MEMORY_NAMESPACE_END
