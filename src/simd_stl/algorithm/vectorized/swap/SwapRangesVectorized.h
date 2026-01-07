#pragma once

#include <src/simd_stl/numeric/SizedSimdDispatcher.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <class _Type_>
simd_stl_always_inline void __swap_ranges_scalar(
	_Type_*		__first,
	_Type_*		__second,
	sizetype	__count) noexcept
{
	for (auto __current = sizetype(0); __current < __count; ++__current) {
		_Type_ __temp = *__first;

		*__first++ = *__second;
		*__second++ = __temp;
	}
}

template <class _Simd_>
struct __swap_ranges_vectorized_internal {
	simd_stl_always_inline void operator()(
		sizetype							__aligned_size,
		sizetype							__tail_size,
		typename _Simd_::value_type*		__first,
		typename _Simd_::value_type*		__second) noexcept
	{
		numeric::zero_upper_at_exit_guard<_Simd_::__generation> _Guard;

		auto __stop_at = __first;
		__advance_bytes(__stop_at, __aligned_size);

		do {
			const auto __loaded_first	= _Simd_::load(__first);
			const auto __loaded_second	= _Simd_::load(__second);

			__loaded_first.store(__second);
			__loaded_second.store(__first);

			__advance_bytes(__first, sizeof(_Simd_));
			__advance_bytes(__second, sizeof(_Simd_));
		} while (__first != __stop_at);

		auto __remaining = __tail_size / sizeof(typename _Simd_::value_type);
		return __swap_ranges_scalar(__first, __second, __remaining);
	}
};


template <typename _Type_>
void __swap_ranges_vectorized(
	_Type_*		__first,
	_Type_*		__second,
	sizetype	__count) noexcept
{
	numeric::__simd_sized_dispatcher<__swap_ranges_vectorized_internal>::__apply<_Type_>(
		__count * sizeof(_Type_), &__swap_ranges_scalar<_Type_>,
		std::make_tuple(__first, __second), std::make_tuple(__first, __second, __count));
}

__SIMD_STL_ALGORITHM_NAMESPACE_END
