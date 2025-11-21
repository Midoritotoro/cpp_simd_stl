#pragma once 

#include <type_traits>

#include <simd_stl/SimdStlNamespace.h>

#include <simd_stl/compatibility/CxxVersionDetection.h>
#include <simd_stl/compatibility/FunctionAttributes.h>

#include <simd_stl/compatibility/Nodiscard.h>
#include <src/simd_stl/type_traits/Invoke.h>


__SIMD_STL_TYPE_TRAITS_NAMESPACE_BEGIN

template <class _Function_> 
constexpr inline bool is_lightweight_callable_v = std::conjunction_v<
    std::bool_constant<sizeof(_Function_) <= sizeof(void*)>,
    std::is_trivially_copy_constructible<_Function_>,
    std::is_trivially_destructible<_Function_>>

template <class _Function_>
struct FunctionReference {
    _Function_& _function;

    template <class ... _Args_>
    constexpr decltype(auto) operator()(_Args_&& ... values) 
        noexcept(type_traits::is_nothrow_invocable_v<_Function_&, _Args_...>)
    {
        return type_traits::invoke(_function, std::forward<_Args_>(values)...);
    }
};

template <class _Function_>
simd_stl_nodiscard constexpr auto passFunction(_Function_& function) noexcept {
    if constexpr (is_lightweight_callable_v<_Function_>)
        return function;
    else
        return FunctionReference(function);
}

__SIMD_STL_TYPE_TRAITS_NAMESPACE_END

