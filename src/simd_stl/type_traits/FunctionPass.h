#pragma once 

#include <type_traits>

#include <simd_stl/SimdStlNamespace.h>

#include <simd_stl/compatibility/CxxVersionDetection.h>
#include <simd_stl/compatibility/FunctionAttributes.h>

#include <simd_stl/compatibility/Nodiscard.h>
#include <src/simd_stl/type_traits/Invoke.h>


__SIMD_STL_TYPE_TRAITS_NAMESPACE_BEGIN

template <class _Function_> 
constexpr inline bool __is_lightweight_callable_v = std::conjunction_v<
    std::bool_constant<sizeof(_Function_) <= sizeof(void*)>,
    std::is_trivially_copy_constructible<_Function_>,
    std::is_trivially_destructible<_Function_>>

template <class _Function_>
struct __function_reference {
    _Function_& _function;

    template <class ... _Args_>
    constexpr decltype(auto) operator()(_Args_&& ... __values) 
        noexcept(type_traits::is_nothrow_invocable_v<_Function_&, _Args_...>)
    {
        return type_traits::invoke(_function, std::forward<_Args_>(__values)...);
    }
};

template <class _Function_>
simd_stl_nodiscard constexpr auto __pass_function(_Function_& __function) noexcept {
    if constexpr (__is_lightweight_callable_v<_Function_>)
        return __function;
    else
        return __function_reference(__function);
}

__SIMD_STL_TYPE_TRAITS_NAMESPACE_END

