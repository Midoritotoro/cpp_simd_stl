#pragma once 

#include <simd_stl/SimdStlNamespace.h>
#include <type_traits>


__SIMD_STL_TYPE_TRAITS_NAMESPACE_BEGIN

#if defined(__cpp_lib_is_constant_evaluated)
	using std::is_constant_evaluated;
#else
	constexpr bool is_constant_evaluated() noexcept {
	#if __has_builtin(__builtin_is_constant_evaluated)
		return __builtin_is_constant_evaluated();
	#else
		return false;
	#endif
	}
#endif // defined(__cpp_lib_is_constant_evaluated)

__SIMD_STL_TYPE_TRAITS_NAMESPACE_END
