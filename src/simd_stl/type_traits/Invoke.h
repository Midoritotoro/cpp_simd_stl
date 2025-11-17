#pragma once 

#include <simd_stl/Types.h>

#include <src/simd_stl/type_traits/FunctionPass.h>
#include <src/simd_stl/type_traits/FunctionInformation.h>


__SIMD_STL_TYPE_TRAITS_NAMESPACE_BEGIN

enum class _InvokerStrategy : uint8 {
    _FunctorCallable,                               // invoke(_FunctorObject, _Args...);

    _PointerToMemberFunctionWithObject,             // invoke(&_ObjectType::_MemberFunction, _Object, _Args...);
    _PointerToMemberFunctionWithReferenceWrapper,   // invoke(&_ObjectType::_MemberFunction, std::ref(_Object), _Args...);
    _PointerToMemberFunctionWithPointer,            // invoke(&_ObjectType::_MemberFunction, _ObjectPointer, _Args...);

    _PointerToMemberDataWithObject,                 // invoke(&_ObjectType::_MemberData, _Object, _Args...);
    _PointerToMemberDataWithReferenceWrapper,       // invoke(&_ObjectType::_MemberData, std::ref(_Object), _Args...);
    _PointerToMemberDataWithPointer                 // invoke(&_ObjectType::_MemberData, _ObjectPointer, _Args...);
};

template <_InvokerStrategy _Strategy_>
struct _Invoker;

template <>
struct _Invoker<_InvokerStrategy::_FunctorCallable> {
    static constexpr auto _Strategy = _InvokerStrategy::_FunctorCallable;

    template <
        class       _Callable_,
        class...    _Args_>
    inline static constexpr auto call(
        _Callable_&&        object, 
        _Args_&&...         args)
            noexcept(noexcept(static_cast<_Callable_&&>(object)(static_cast<_Args_&&>(args)...)))
                -> decltype(static_cast<_Callable_&&>(object)(static_cast<_Args_&&>(args)...))
    {
        return static_cast<_Callable_&&>(object)(static_cast<_Args_&&>(args)...);
    }
};

template <>
struct _Invoker<_InvokerStrategy::_PointerToMemberFunctionWithObject> {
    static constexpr auto _Strategy = _InvokerStrategy::_PointerToMemberFunctionWithObject;

    template <
        class       _MemberFunction_, 
        class       _Object_, 
        class...    _Args_>
    inline static constexpr auto call(
        _MemberFunction_    memberFunction, 
        _Object_&&          object, 
        _Args_&&...         args)
            noexcept(noexcept((static_cast<_Object_&&>(object).*memberFunction)(static_cast<_Args_&&>(args)...)))
                -> decltype((static_cast<_Object_&&>(object).*memberFunction)(static_cast<_Args_&&>(args)...))
    {
        return (static_cast<_Object_&&>(object).*memberFunction)(static_cast<_Args_&&>(args)...);
    }
};

template <>
struct _Invoker<_InvokerStrategy::_PointerToMemberFunctionWithReferenceWrapper> {
    static constexpr auto _Strategy = _InvokerStrategy::_PointerToMemberFunctionWithReferenceWrapper;

    template <
        class       _MemberFunction_,
        class       _ReferenceWrapper_,
        class...    _Args_>
    inline static constexpr auto call(
        _MemberFunction_    memberFunction,
        _ReferenceWrapper_  referenceWrapper,
        _Args_&&...         args)
            noexcept(noexcept((referenceWrapper.get().*memberFunction)(static_cast<_Args_&&>(args)...)))
                -> decltype((referenceWrapper.get().*memberFunction)(static_cast<_Args_&&>(args)...))
    {
        return (referenceWrapper.get().*memberFunction)(static_cast<_Args_&&>(args)...);
    }
};

template <>
struct _Invoker<_InvokerStrategy::_PointerToMemberFunctionWithPointer> {
    static constexpr auto _Strategy = _InvokerStrategy::_PointerToMemberFunctionWithPointer;

    template <
        class       _MemberFunction_,
        class       _Pointer_,
        class...    _Args_>
    inline static constexpr auto call(
        _MemberFunction_    memberFunction,
        _Pointer_&&         pointerToObject,
        _Args_&&...         args)
            noexcept(noexcept(((*static_cast<_Pointer_&&>(pointerToObject)).*memberFunction)(static_cast<_Args_&&>(args)...)))
                -> decltype(((*static_cast<_Pointer_&&>(pointerToObject)).*memberFunction)(static_cast<_Args_&&>(args)...))
    {
        return ((*static_cast<_Pointer_&&>(pointerToObject)).*memberFunction)(static_cast<_Args_&&>(args)...);
    }
};


template <>
struct _Invoker<_InvokerStrategy::_PointerToMemberDataWithObject> {
    static constexpr auto _Strategy = _InvokerStrategy::_PointerToMemberDataWithObject;

    template <
        class _MemberData_, 
        class _Object_>
    inline static constexpr auto call(
        _MemberData_    memberData,
        _Object_&&      object) noexcept -> decltype(static_cast<_Object_&&>(object).*memberData)
    {
        return static_cast<_Object_&&>(object).*memberData;
    }
};

template <>
struct _Invoker<_InvokerStrategy::_PointerToMemberDataWithReferenceWrapper> {
    static constexpr auto _Strategy = _InvokerStrategy::_PointerToMemberDataWithReferenceWrapper;

    template <
        class _MemberData_,
        class _ReferenceWrapper_>
    inline static constexpr auto call(
        _MemberData_        memberData,
        _ReferenceWrapper_  referenceWrapper) noexcept -> decltype(referenceWrapper.get().*memberData)
    {
        return referenceWrapper.get().*memberData;
    }
};

template <>
struct _Invoker<_InvokerStrategy::_PointerToMemberDataWithPointer> {
    static constexpr auto _Strategy = _InvokerStrategy::_PointerToMemberDataWithPointer;

    template <
        class _MemberData_,
        class _Pointer_>
    inline static constexpr auto call(
        _MemberData_    memberData,
        _Pointer_&&     pointerToObject) noexcept(noexcept((*static_cast<_Pointer_&&>(pointerToObject)).*memberData))
            -> decltype((*static_cast<_Pointer_&&>(pointerToObject)).*memberData)
    {
        return (*static_cast<_Pointer_&&>(pointerToObject)).*memberData;
    }
};

template <
    class _Callable_, 
    class _Object_,
    class _RemovedQualifiers_       = std::remove_cvref_t<_Callable_>,
    bool _IsMemberFunctionPointer_  = std::is_member_function_pointer_v<_RemovedQualifiers_>,
    bool _IsMemberFunctionData_     = std::is_member_object_pointer_v<_RemovedQualifiers_>>
struct _SelectInvoker;

template <
    class _Callable_,
    class _Object_, 
    class _RemovedQualifiers_>
struct _SelectInvoker<_Callable_, _Object_, _RemovedQualifiers_, true, false> {
    using type = std::conditional_t<
        std::is_same_v<typename function_class_type<_RemovedQualifiers_>, std::remove_cvref_t<_Object_>> ||
        std::is_base_of_v<typename function_class_type<_RemovedQualifiers_>, std::remove_cvref_t<_Object_>>,
            _Invoker<_InvokerStrategy::_PointerToMemberFunctionWithObject>,
            std::conditional_t<
                is_specialization_v<std::remove_cvref_t<_Object_>, std::reference_wrapper>,
                    _Invoker<_InvokerStrategy::_PointerToMemberFunctionWithReferenceWrapper>,
                    _Invoker<_InvokerStrategy::_PointerToMemberFunctionWithPointer>>>;
};

template <
    class _Callable_, 
    class _Object_, 
    class _RemovedQualifiers_>
struct _SelectInvoker<_Callable_, _Object_, _RemovedQualifiers_, false, true> {
    using type = std::conditional_t<
        std::is_same_v<typename _Member_object_pointer_class_type<_RemovedQualifiers_>::type, std::remove_cvref_t<_Object_>> ||
        std::is_base_of_v<typename _Member_object_pointer_class_type<_RemovedQualifiers_>::type, std::remove_cvref_t<_Object_>>,
            _Invoker<_InvokerStrategy::_PointerToMemberDataWithObject>,
            std::conditional_t<
                is_specialization_v<std::remove_cvref_t<_Object_>, std::reference_wrapper>, 
                    _Invoker<_InvokerStrategy::_PointerToMemberDataWithReferenceWrapper>,
                    _Invoker<_InvokerStrategy::_PointerToMemberDataWithPointer>>>;
};

template <
    class _Callable_,
    class _Object_,
    class _RemovedQualifiers_>
struct _SelectInvoker<_Callable_, _Object_, _RemovedQualifiers_, false, false> {
    using type = _Invoker<_InvokerStrategy::_FunctorCallable>;
};

template <
    class _Callable_,
    class _Object_>
using invoker_type = typename _SelectInvoker<_Callable_, _Object_>::type;

template <
    class _From_, 
    class _To_, 
    class = void>
struct _Invoke_convertible:
    std::false_type 
{};

template <
    class _From_,
    class _To_>
struct _Invoke_convertible<_From_, _To_, std::void_t<decltype(_FakeCopyInit<_To_>(_ReturnsExactly<_From_>()))>>: 
    std::true_type
{};

template <
    class _From_,
    class _To_>
struct _Invoke_nothrow_convertible: 
    std::bool_constant<noexcept(_FakeCopyInit<_To_>(_ReturnsExactly<_From_>()))>
{};

template <
    class   _Result_,
    bool    _NoThrow_>
struct _Invoke_common_traits {
    using type                  = _Result_;
    using _Is_invocable         = std::true_type;
    using _Is_nothrow_invocable = std::bool_constant<_NoThrow_>;

    template <class _Rx>
    using _Is_invocable_r = std::bool_constant<std::disjunction_v<std::is_void<_Rx>, _Invoke_convertible<type, _Rx>>>;

    template <class _Rx>
    using _Is_nothrow_invocable_r = std::bool_constant<std::conjunction_v<_Is_nothrow_invocable,
        std::disjunction<std::is_void<_Rx>,
            std::conjunction<_Invoke_convertible<type, _Rx>, _Invoke_nothrow_convertible<type, _Rx>>>>>;
};

template <
    class _Void_, 
    class _Callable_>
struct _Invoke_traits_zero {
    using _Is_invocable         = std::false_type;
    using _Is_nothrow_invocable = std::false_type;

    template <class _Rx>
    using _Is_invocable_r = std::false_type;

    template <class _Rx>
    using _Is_nothrow_invocable_r = std::false_type;
};

template <class _Callable_>
using _Decltype_invoke_zero = decltype(std::declval<_Callable_>()());

template <class _Callable_>
struct _Invoke_traits_zero<std::void_t<_Decltype_invoke_zero<_Callable_>>, _Callable_>: 
    _Invoke_common_traits<_Decltype_invoke_zero<_Callable_>, noexcept(std::declval<_Callable_>()())>
{};

template <
    class       _Void_, 
    class...    _Args_>
struct _Invoke_traits_nonzero {
    using _Is_invocable         = std::false_type;
    using _Is_nothrow_invocable = std::false_type;

    template <class _Rx>
    using _Is_invocable_r = std::false_type;

    template <class _Rx>
    using _Is_nothrow_invocable_r = std::false_type;
};

template <
    class       _Callable_, 
    class       _FirstType_, 
    class...    _Args_>
using _Decltype_invoke_nonzero = decltype(invoker_type<_Callable_, _FirstType_>::call(
    std::declval<_Callable_>(), std::declval<_FirstType_>(), std::declval<_Args_>()...));

template <
    class       _Callable_, 
    class       _FirstType_, 
    class...    _Args_>
struct _Invoke_traits_nonzero<std::void_t<_Decltype_invoke_nonzero<_Callable_, _FirstType_, _Args_...>>, _Callable_, _FirstType_, _Args_...>:
    _Invoke_common_traits<_Decltype_invoke_nonzero<_Callable_, _FirstType_, _Args_...>, noexcept(invoker_type<_Callable_, _FirstType_>::call(
            std::declval<_Callable_>(), std::declval<_FirstType_>(), std::declval<_Args_>()...))> {};

template <
    class       _Callable_,
    class...    _Args_>
using _Select_invoke_traits = std::conditional_t<sizeof...(_Args_) == 0, _Invoke_traits_zero<void, _Callable_>,
    _Invoke_traits_nonzero<void, _Callable_, _Args_...>>;

template <
    class       _Callable_,
    class ...   _Args_>
constexpr inline bool is_invocable_v = _Select_invoke_traits<_Callable_, _Args_...>::_Is_invocable::value;

template <
    class       _Callable_,
    class...    _Args_>
inline constexpr bool is_nothrow_invocable_v = _Select_invoke_traits<_Callable_, _Args_...>::_Is_nothrow_invocable::value;

template <
    class       _Callable_,
    class...    _Args_>
inline constexpr bool is_invocable_r = _Select_invoke_traits<_Callable_, _Args_...>::_Is_invocable_r::value;

template <
    class       _Callable_,
    class...    _Args_>
inline constexpr bool is_nothrow_invocable_r = _Select_invoke_traits<_Callable_, _Args_...>::_Is_nothrow_invocable_r::value;

template <class _Callable_>
constexpr auto invoke(_Callable_&& callable) noexcept(is_nothrow_invocable_v<_Callable_>)
    -> decltype(static_cast<_Callable_&&>(callable)())
{
    static_assert(is_invocable_v<_Callable_>, "invoke argument is not callable");
    return static_cast<_Callable_&&>(callable)();
}

#define __INVOKER_CALL invoker_type<_Callable_, _FirstArgument_>::call( \
    static_cast<_Callable_&&>(callable), \
    static_cast<_FirstArgument_&&>(firstArgument), \
    static_cast<_Args_&&>(args)...) 

template <
    class       _Callable_,
    class       _FirstArgument_, 
    class...    _Args_>
constexpr auto invoke(
    _Callable_&&        callable,
    _FirstArgument_&&   firstArgument,
    _Args_&&...         args)
        noexcept(noexcept(__INVOKER_CALL)) -> decltype(__INVOKER_CALL)
{
    static_assert(is_invocable_v<_Callable_, _FirstArgument_, _Args_...>, "invoke argument is not callable");
    return __INVOKER_CALL;
}

#undef __INVOKER_CALL

__SIMD_STL_TYPE_TRAITS_NAMESPACE_END
