#pragma once 

#include <simd_stl/Types.h>

__SIMD_STL_MEMORY_NAMESPACE_BEGIN

template <typename _Type_>
using __deduce_pointer_address_type = std::conditional_t<
    std::is_pointer_v<_Type_> || std::is_same_v<std::decay_t<_Type_>, std::nullptr_t>,
    uintptr, _Type_>;

template <typename _Type_>
constexpr __deduce_pointer_address_type<_Type_> pointerToIntegral(_Type_ pointer) noexcept {
    if constexpr (std::is_same_v<std::decay_t<_Type_>, std::nullptr_t>)
        return 0;
    else if constexpr (std::is_pointer_v<std::decay_t<_Type_>>)
        return reinterpret_cast<uintptr>(pointer);
    else
        return pointer;
}

__SIMD_STL_MEMORY_NAMESPACE_END
