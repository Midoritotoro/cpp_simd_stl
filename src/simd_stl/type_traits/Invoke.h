#pragma once 

#include <simd_stl/Types.h>
#include <src/simd_stl/type_traits/FunctionPass.h>


__SIMD_STL_TYPE_TRAITS_NAMESPACE_BEGIN

template <
    class                       _Type_,
    template <class...> class   _Template_>
constexpr bool is_specialization_v = false;

template <
    template <class...> class   _Template_,
    class...                    _Types_>
constexpr bool is_specialization_v<_Template_<_Types_...>, _Template_> = true;

template <class _Type_>
struct _Class_type {
    using type = void;
};

template <
    class       _ReturnValue_,
    class       _Class_, 
    class...    _Types_>                                   
struct _Class_type {
    using type = _Class_;
};

template <class... _Types_>
using class_type = typename _Class_type<_Types_...>::type;

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
struct _SelectInvoker<
    _Callable_, _Object_, _RemovedQualifiers_, true, false
>:
    std::conditional_t<
        std::is_same_v<class_type<_RemovedQualifiers_>, std::remove_cvref_t<_Object_>> ||
        std::is_base_of_v<class_type<_RemovedQualifiers_>, std::remove_cvref_t<_Object_>>,
        _Invoker<_InvokerStrategy::_PointerToMemberFunctionWithObject>,
            std::conditional_t<
                is_specialization_v<std::remove_cvref_t<_Object_>, std::reference_wrapper>,
                    _Invoker<_InvokerStrategy::_PointerToMemberFunctionWithReferenceWrapper>,
                    _Invoker<_InvokerStrategy::_PointerToMemberFunctionWithPointer>>>
{};

template <
    class _Callable_, 
    class _Object_, 
    class _RemovedQualifiers_>
struct _SelectInvoker<
    _Callable_, _Object_, _RemovedQualifiers_, false, true
>: 
    std::conditional_t<
        std::is_same_v<class_type<_RemovedQualifiers_>, std::remove_cvref_t<_Object_>> || 
        is_base_of_v<class_type<_RemovedQualifiers_>, std::remove_cvref_t<_Object_>>,
            _Invoker<_InvokerStrategy::_PointerToMemberDataWithObject>,
            std::conditional_t<
                is_specialization_v<std::remove_cvref_t<_Object_>, std::reference_wrapper>, 
                    _Invoker<_InvokerStrategy::_PointerToMemberDataWithReferenceWrapper>,
                    _Invoker<_InvokerStrategy::_PointerToMemberDataWithPointer>>> 
{};

template <
    class _Callable_,
    class _Object_,
    class _RemovedQualifiers_>
struct _SelectInvoker<_Callable_, _Object_, _RemovedQualifiers_, false, false>: 
    _Invoker<_InvokerStrategy::_FunctorCallable> 
{};

template <
    class _Callable_,
    class _Object_,
    class _RemovedQualifiers_ = std::remove_cvref_t<_Callable_>>
using invoker_type = typename _SelectInvoker<_Callable_, _Object_, _RemovedQualifiers_>;

template <class _Callable_>
constexpr auto invoke(_Callable_&& callable) noexcept(noexcept(static_cast<_Callable_&&>(callable)())) 
    -> decltype(static_cast<_Callable_&&>(callable)())
{
    return static_cast<_Callable_&&>(callable)();
}

template <
    class       _Callable_,
    class       _FirstArg_,
    class...    _Args_>
constexpr auto invoke(
    _Callable_&&    object,
    _FirstArg_&&    firstArg,
    _Args_&&...     args) noexcept(noexcept(invoker_type<_Callable_, _FirstArg_>::call(
        static_cast<_Callable_&&>(object), static_cast<_FirstArg_&&>(firstArg), static_cast<_Args_&&>(args)...)))
            -> decltype(invoker_type<_Callable_, _FirstArg_>::call(
                static_cast<_Callable_&&>(object), static_cast<_FirstArg_&&>(firstArg), static_cast<_Args_&&>(args)...))
{
    return invoker_type<_Callable_, _FirstArg_>::call(
        static_cast<_Callable_&&>(object),
        static_cast<_FirstArg_&&>(firstArg),
        static_cast<_Args_&&>(args)...);
}

__SIMD_STL_TYPE_TRAITS_NAMESPACE_END
