#pragma once 

#include <simd_stl/SimdStlNamespace.h>

#include <simd_stl/compatibility/CompilerDetection.h>
#include <simd_stl/compatibility/CxxVersionDetection.h>

#include <type_traits>
#include <xmemory>

__SIMD_STL_TYPE_TRAITS_NAMESPACE_BEGIN


template <
	typename _Struct_,
	typename _Member_>
using _Has_member = std::void_t<decltype(std::declval<_Struct_>().*std::declval<_Member_>())>;

template <
	typename _Struct_,
	typename _Member_>
inline constexpr bool has_member_v = std::is_convertible_v<_Has_member<_Struct_, _Member_>, void>;

template <typename _Type_>
inline constexpr bool is_relocatable_v = std::is_trivially_copyable_v<_Type_> && std::is_trivially_destructible_v<_Type_>;

template <typename _Type_>
inline constexpr bool is_value_initialization_bitwise_zero_v = std::is_scalar_v<_Type_> && !std::is_member_object_pointer_v<_Type_>;


#ifdef __cpp_lib_remove_cvref
  using std::remove_cvref;
  using std::remove_cvref_t;
#else
  template <typename _Type_>
  using remove_cvref = std::remove_cv<std::remove_reference_t<_Type_>>;

  template <typename _Type_>
  using remove_cvref_t = std::remove_cv_t<std::remove_reference_t<_Type_>>;
#endif // __cpp_lib_remove_cvref

template <
	typename,
	typename = void>
constexpr inline bool is_type_complete_v = false;

template <typename _Type_>
constexpr inline bool is_type_complete_v<_Type_, std::void_t<decltype(sizeof(_Type_))>> = true


template <
    typename    _Type_,
    class       _Allocator_>
constexpr inline bool is_nothrow_move_constructible_v =
#if defined(simd_stl_cpp_msvc)
	std::conjunction_v<
		std::is_nothrow_move_constructible<_Type_>,
		std::_Uses_default_construct<_Allocator_, _Type_*, _Type_>>;
#else
	std::is_nothrow_move_constructible_v<_Allocator_>;
#endif // defined(simd_stl_cpp_msvc)

template <
    typename    _Type_,
    class       _Allocator_,
    class ...   _Args_>
constexpr inline bool is_nothrow_move_constructible_args_v =
#if defined(simd_stl_cpp_msvc)
	std::conjunction_v<
		std::is_nothrow_move_constructible<_Type_>,
		std::_Uses_default_construct<_Allocator_, _Type_*, _Args_...>>;
#else
	std::is_nothrow_move_constructible_v<_Allocator_>;
#endif // defined(simd_stl_cpp_msvc)

template <
    class _Source_, 
    class _Destination_>
// checks the convertibility of _Source_ to _Destination_
constexpr inline bool is_pointer_address_convertible_v = std::is_void_v<_Source_>
    || std::is_void_v<_Destination_>
    || std::is_same_v<
        std::remove_cv_t<_Source_>, 
        std::remove_cv_t<_Destination_>>
#ifdef __cpp_lib_is_pointer_interconvertible
    || std::is_pointer_interconvertible_base_of_v<_Destination_, _Source_>
#endif
    ;

template <
    class _FirstType_, 
    class _SecondType_,
    class = void>
constexpr bool can_compare_with_operator_equal_v = false;

template <
    class _FirstType_,
    class _SecondType_>
constexpr bool can_compare_with_operator_equal_v<
    _FirstType_, _SecondType_,
    std::void_t<decltype(std::declval<_FirstType_&>() == std::declval<_SecondType_&>())>> = true;

template <
    class _FirstType_,
    class _SecondType_>
constexpr bool is_pointer_address_comparable_v = can_compare_with_operator_equal_v<_FirstType_*, _SecondType_*> && 
        (is_pointer_address_convertible_v<_FirstType_, _SecondType_> 
    ||  is_pointer_address_convertible_v<_SecondType_, _FirstType_>);

template <
	class		_Ty,
	class...	_Types>
constexpr inline bool is_any_of_v =
#if simd_stl_has_cxx17
    (std::is_same_v<_Ty, _Types> || ...);
#else
    std::disjunction_v<std::is_same<_Ty, _Types>...>;
#endif


template <
    typename _Type_, 
    typename = void>
struct _Invocable_type {
    using type = void;
};

template <typename _Type_>
struct _Invocable_type<_Type_, std::void_t<decltype(&_Type_::operator())>> {
    using type = _Type_;
};

template <
    typename    _ProbablyCallable_,
    typename... _Args_>
struct _Invocable_type<_ProbablyCallable_(_Args_...), void> {
    using type = _ProbablyCallable_(_Args_ ...);
};

template <
    typename    _ProbablyCallable_,
    typename... _Args_>
struct _Invocable_type<_ProbablyCallable_(*)(_Args_...), void> {
    using type = _ProbablyCallable_(_Args_...);
};

template <typename _Type_>
using invocable_type = typename callable_type<std::decay_t<_Type_>>::type;

__SIMD_STL_TYPE_TRAITS_NAMESPACE_END
