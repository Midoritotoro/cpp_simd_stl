#pragma once 

#include <type_traits>
#include <simd_stl/SimdStlNamespace.h>


__SIMD_STL_TYPE_TRAITS_NAMESPACE_BEGIN

template <
	class	_Element_,
	bool	_IsEnum_ = std::is_enum_v<_Element_>>
struct _Unwrap_enum {
	using type = std::underlying_type_t<_Element_>;
};

template <class _Element_>
struct _Unwrap_enum<_Element_, false> {
	using type = _Element_;
};

template <class _Element_>
using unwrap_enum_t = typename _Unwrap_enum<_Element_>::type;

__SIMD_STL_TYPE_TRAITS_NAMESPACE_END
