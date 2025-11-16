#pragma once 

#include <src/simd_stl/algorithm/AlgorithmDebug.h>
#include <src/simd_stl/type_traits/SwapSafety.h>

#include <src/simd_stl/algorithm/vectorized/SwapRangesVectorized.h>
#include <src/simd_stl/algorithm/MsvcIteratorUnwrap.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <class _Type_>
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

template <
	typename _FirstForwardIterator_,
	typename _SecondForwardIterator_>
constexpr void iter_swap(
	_FirstForwardIterator_	first,
	_SecondForwardIterator_ second) noexcept(noexcept(swap(*first, *second)))
{
	swap(*first, *second);
}

#if defined(simd_stl_cpp_clang) || defined(__EDG__)
	void swap() = delete;
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
struct _Has_adl_swap<_Type_, std::void_t<decltype(swap(
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
	std::negation<std::_Has_ADL_swap_detail::_Has_ADL_swap<_Type_>
>>;

#ifdef __cpp_lib_byte
template <>
inline constexpr bool is_trivially_swappable_v<std::byte> = true;
#endif // defined(__cpp_lib_byte)


template <
	class _Type_,
	sizetype _Length_> 
constexpr void swap(
	_Type_ (&first)[_Length_],
	_Type_ (&second)[_Length_]) noexcept(noexcept(swap(*first, *second)))
{
	if (&first == &second)
		return;

	if constexpr (is_trivially_swappable_v<_Type_>) {
#if simd_stl_has_cxx20
		if (type_traits::is_constant_evaluated() == false)
#endif // simd_stl_has_cxx20
			return _SwapRangesVectorized<_Type_>(first, second, _Length_);
	}
	else {
		for (sizetype current = 0; current < _Length_; ++current)
			swap(first[current], second[current]);
	}
}

template <
	class _FirstForwardIterator_,
	class _SecondForwardIterator_>
constexpr _FirstForwardIterator_ swap_ranges(
	_FirstForwardIterator_	first1,
	_FirstForwardIterator_	last1,
	_SecondForwardIterator_ first2) noexcept
{
	using _FirstForwardIteratorUnwrapped_ = unwrapped_iterator_type<_FirstForwardIterator_>;

	__verifyRange(first1, last1);

	const auto first1Unwrapped	= _UnwrapIterator(first1);
	const auto last1Unwrapped	= _UnwrapIterator(last1);

	const auto first2Unwrapped	= _UnwrapIterator(first2);

	using _ValueType_ = type_traits::IteratorValueType<_FirstForwardIterator_>;

	if constexpr (is_trivially_swappable_v<_ValueType_>) {
		const auto difference = IteratorsDifference(first1Unwrapped, last1Unwrapped);

#if simd_stl_has_cxx20
		if (type_traits::is_constant_evaluated() == false)
#endif // simd_stl_has_cxx20
			_SwapRangesVectorized<_ValueType_>(
				std::to_address(first1Unwrapped), std::to_address(first2Unwrapped), difference);

		_SeekPossiblyWrappedIterator(first2, first2 + difference);
	}
	else {
		for (; first1Unwrapped != last1; ++first1Unwrapped, ++first2Unwrapped)
			swap(*first1Unwrapped, *first2Unwrapped);

		_SeekPossiblyWrappedIterator(first2, first2Unwrapped);
	}

	return first2;
}

template <
	class _ExecutionPolicy_,	
	class _FirstForwardIterator_,
	class _SecondForwardIterator_>
constexpr _FirstForwardIterator_ swap_ranges(
	_ExecutionPolicy_&&,
	_FirstForwardIterator_	first1,
	_FirstForwardIterator_	last1,
	_SecondForwardIterator_ first2) noexcept
{
	return simd_stl::algorithm::swap_ranges(first1, last1, first2);
}

__SIMD_STL_ALGORITHM_NAMESPACE_END
