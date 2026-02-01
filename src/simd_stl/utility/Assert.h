#pragma once  

#include <simd_stl/SimdStlNamespace.h>

#include <simd_stl/compatibility/Warnings.h>
#include <simd_stl/compatibility/BranchPrediction.h>

#include <cstdlib>
#include <cassert>

#include <iostream>


__SIMD_STL_NAMESPACE_BEGIN


simd_stl_disable_warning_msvc(6011);

struct __static_locale {
	__static_locale() noexcept {
		setlocale(LC_ALL, "");
	}
};

static const __static_locale __lc;

inline void __fail(
	const char* __message,
	const char* __file,
	int			__line) noexcept
{
	printf("Error: %s in File \"%s\", Line: %d\n", __message, __file, __line);

	volatile auto __nullptr_value = (int*)nullptr;
	*__nullptr_value = 0;
	
	std::abort();
	std::terminate();
}

inline const char* __extract_basename(
	const char* __path,
	size_t		__size) noexcept
{
	while (__size != 0 && __path[__size - 1] != '/' && __path[__size - 1] != '\\')
		--__size;

	return __path + __size;
}

#define __return_on_failure(__message, __file, __line, __return_value) \
	do { \
		printf("Error: %s in File \"%s\", Line: %d\n", __message, __file, __line); \
		return __return_value; \
	} \
		while (0)
	

#define __assert_validation_condition(__condition, __message, __file, __line)\
	((simd_stl_unlikely(!((__condition))))\
		? simd_stl::__fail(__message, __file, __line)\
		: void(0))

#define __assert_validation_condition_with_ret(__condition, __message, __file, __line, __return_value)\
	if ((simd_stl_unlikely(!(__condition)))) \
		__return_on_failure(__message, __file, __line, __return_value)

#define __source_file_basename (simd_stl::__extract_basename(\
	__FILE__,\
	sizeof(__FILE__)))

#define simd_stl_assert_log(__condition, __message) (__assert_validation_condition(\
	__condition,\
	__message,\
	__source_file_basename,\
	__LINE__))

#define simd_stl_assert_return(__condition, __message, __return_value) __assert_validation_condition_with_ret(\
	__condition,\
	__message,\
	__source_file_basename,\
	__LINE__, \
	__return_value)

#define simd_stl_assert(__condition) simd_stl_assert_log((__condition), "\"" #__condition "\"")
#define simd_stl_assert_unreachable() simd_stl_assert(false)

#if !defined(NDEBUG)

#define simd_stl_debug_assert_return	simd_stl_assert_return
#define simd_stl_debug_assert			simd_stl_assert

#define simd_stl_debug_assert_log		simd_stl_assert_log

#else 

#define simd_stl_debug_assert_return(__condition, __message, __return_value)
#define simd_stl_debug_assert(__condition)

#define simd_stl_debug_assert_log(__condition, __message)

#endif // !defined(NDEBUG)

__SIMD_STL_NAMESPACE_END
