#pragma once 

#include <src/simd_stl/algorithm/AlgorithmDebug.h>
#include <src/simd_stl/type_traits/SwapSafety.h>

#include <src/simd_stl/algorithm/vectorized/SwapRangesVectorized.h>
#include <src/simd_stl/algorithm/MsvcIteratorUnwrap.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <typename _Type_>
constexpr void swap(
	_Type_& first,
	_Type_& second) noexcept(
		std::is_nothrow_move_constructible<_Type_>::value &&
		std::is_nothrow_move_assignable<_Type_>::value
	)
{
	auto temp = first;

	first	= std::move(second);
	second	= std::move(temp);
}

#if defined(simd_stl_cpp_clang) || defined(__EDG__)
	void swap() = delete; // Block unqualified name lookup
#else
	void swap();
#endif

template <
	class,
	class = void>
struct _Has_adl_swap:
	std::false_type
{};

template <class _Type_>
struct _Has_adl_swap<_Type_, std::void_t<decltype(std::swap(
	std::declval<_Type_&>(),
	std::declval<_Type_&>()))>
>:
	std::true_type
{};

template <class _Type_>
constexpr bool is_trivially_swappable_v = std::conjunction_v<
	std::is_trivially_destructible<_Type_>,
	std::is_trivially_move_constructible<_Type_>,
	std::is_trivially_move_assignable<_Type_>,
	std::negation<_Has_adl_swap<_Type_>
>>;

#ifdef __cpp_lib_byte
template <>
inline constexpr bool is_trivially_swappable_v<std::byte> = true;
#endif // defined(__cpp_lib_byte)


template <
	typename _Type_,
	sizetype _Length_> 
constexpr void swap(
	_Type_ (&first)[_Length_],
	_Type_ (&second)[_Length_]) noexcept(noexcept(swap(*first, *second)))
{
	if constexpr (is_trivially_swappable_v<_Type_>)
		_SwapRangesVectorized<_Type_>(first, second, _Length_);
	else
		for (sizetype current = 0; current < _Length_; ++current)
			swap(first[current], second[current]);
}

__SIMD_STL_ALGORITHM_NAMESPACE_END
