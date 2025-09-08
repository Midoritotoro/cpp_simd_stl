#pragma once 

#include <type_traits>

#include <simd_stl/SimdStlNamespace.h>
#include <simd_stl/compatibility/Warnings.h>


__SIMD_STL_TYPE_TRAITS_NAMESPACE_BEGIN

struct _Nonesuch {
	~_Nonesuch() = delete;
	_Nonesuch(const _Nonesuch&) = delete;
	void operator=(const _Nonesuch&) = delete;
};

template <
	typename						_Type_,
	typename						_Void_,
	template <typename...> class	_Op_,
	typename...						_Args_>
struct _Detector 
{
	using value_t = std::false_type;
	using type = _Type_;
};

template <
	typename						_Type_,
	template <typename...> class	_Op_,
	typename...						_Args_>
struct _Detector<
	_Type_,
	std::void_t<_Op_<_Args_...>>,
	_Op_,
	_Args_...> 
{
	using value_t = std::true_type;
	using type = _Op_<_Args_...>;
};


template <
	template <typename...> class	_Op_,
	typename...						_Args_>
using is_detected = typename _Detector<
	_Nonesuch, void,
	_Op_, _Args_...>::value_t;

template <
	template <typename...> class	_Op_,
	typename...						_Args_>
constexpr inline bool is_detected_v = is_detected<_Op_, _Args_...>::value;

namespace _detail {
	simd_stl_warning_push

	simd_stl_disable_warning_gcc("-Wold-style-cast");
	simd_stl_disable_warning_clang("-Wold-style-cast");

	template <
		typename From,
		typename To>
	using is_virtual_base_conversion_test = decltype((To*)std::declval<From*>());

	simd_stl_warning_push

	template <
		typename Base,
		typename Derived,
		typename = void>
	struct is_virtual_base_of:
		std::false_type
	{};

	template <
		typename Base,
		typename Derived>
	struct is_virtual_base_of<
		Base, Derived,
		std::enable_if_t<
			std::conjunction_v<
				std::is_base_of<Base, Derived>,
				is_detected<is_virtual_base_conversion_test, Derived, Base>,
			std::negation<
				is_detected<is_virtual_base_conversion_test, Base, Derived>
				>
			>
		>	
	>: 
		std::true_type
	{};
}

template <
	typename Base,
	typename Derived>
using is_virtual_base_of = _detail::is_virtual_base_of<
	std::remove_cv_t<Base>,
	std::remove_cv_t<Derived>>;

template <
	typename Base,
	typename Derived>
constexpr inline bool is_virtual_base_of_v = is_virtual_base_of<Base, Derived>::value;

__SIMD_STL_TYPE_TRAITS_NAMESPACE_END
