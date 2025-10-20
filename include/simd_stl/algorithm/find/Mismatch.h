#pragma once 

#include <src/simd_stl/algorithm/AlgorithmDebug.h>
#include <src/simd_stl/type_traits/SimdAlgorithmSafety.h>

#include <simd_stl/compatibility/Nodiscard.h>
#include <simd_stl/compatibility/Inline.h>

#include <src/simd_stl/algorithm/vectorized/MismatchVectorized.h>
#include <src/simd_stl/algorithm/MsvcIteratorUnwrap.h>

#include <src/simd_stl/type_traits/CanMemcmpElements.h>
#include <src/simd_stl/type_traits/FunctionPass.h>

#include <cmath>

__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <
	class _FirstIterator_,
	class _SecondIterator_,
	class _Predicate_>
simd_stl_nodiscard simd_stl_always_inline simd_stl_constexpr_cxx20 std::pair<_FirstIterator_, _SecondIterator_> mismatch(
	_FirstIterator_			first1,
	const _FirstIterator_	last1,
	_SecondIterator_		first2,
	_Predicate_				predicate) noexcept(
		type_traits::is_nothrow_invocable_v<
			_Predicate_,
			type_traits::IteratorValueType<_FirstIterator_>,
			type_traits::IteratorValueType<_SecondIterator_>
		>
	)
{
	__verifyRange(first1, last1);

#if defined(simd_stl_cpp_msvc)
	using _FirstIteratorUnwrappedType_	= std::_Unwrapped_t<_FirstIterator_>;
	using _SecondIteratorUnwrappedType_	= std::_Unwrapped_t<_SecondIterator_>;
#else 
	using _FirstIteratorUnwrappedType_	= _FirstIterator_;
	using _SecondIteratorUnwrappedType_ = _SecondIterator_;
#endif // defined(simd_stl_cpp_msvc) 

	auto first1Unwrapped		= __unwrapIterator(first1);
	const auto last1Unwrapped	= __unwrapIterator(last1);

	const auto length			= IteratorsDifference(first1Unwrapped, last1Unwrapped);
	auto first2Unwrapped		= __unwrapSizedIterator(first2, length);

	if constexpr (type_traits::is_vectorized_search_algorithm_safe_v<
		_FirstIteratorUnwrappedType_, _SecondIteratorUnwrappedType_, _Predicate_>)
	{
#if simd_stl_has_cxx20
		if (type_traits::is_constant_evaluated() == false)
#endif
		{
			using _ValueType_ = type_traits::IteratorValueType<_FirstIterator_>;

			const auto position = MismatchVectorized<_ValueType_>(
				std::to_address(first1Unwrapped),
				std::to_address(first2Unwrapped), 
				length);

			first1Unwrapped += static_cast<type_traits::IteratorDifferenceType<_FirstIterator_>>(position);
			first2Unwrapped += static_cast<type_traits::IteratorDifferenceType<_SecondIterator_>>(position);

#if defined(simd_stl_cpp_msvc)
			std::_Seek_wrapped(first2, first2Unwrapped);
			std::_Seek_wrapped(first1, first1Unwrapped);
#else 
			first1 = first1Unwrapped;
			first2 = first2Unwrapped;
#endif

			return { first1, first2 };
		}
	}

	while (first1Unwrapped != last1Unwrapped && predicate(*first1Unwrapped, *first2Unwrapped)) {
		++first1Unwrapped;
		++first2Unwrapped;
	}

#if defined(simd_stl_cpp_msvc)
	std::_Seek_wrapped(first2, first2Unwrapped);
	std::_Seek_wrapped(first1, first1Unwrapped);
#else 
	first1 = first1Unwrapped;
	first2 = first2Unwrapped;
#endif

	return { first1, first2 };
}

template <
	class _FirstIterator_,
	class _SecondIterator_,
	class _Predicate_>
simd_stl_nodiscard simd_stl_always_inline simd_stl_constexpr_cxx20 std::pair<_FirstIterator_, _SecondIterator_> mismatch(
	_FirstIterator_			first1,
	const _FirstIterator_	last1,
	_SecondIterator_		first2,
	const _SecondIterator_	last2,
	_Predicate_				predicate) noexcept(
		type_traits::is_nothrow_invocable_v<
			_Predicate_,
			type_traits::IteratorValueType<_FirstIterator_>,
			type_traits::IteratorValueType<_SecondIterator_>
		>
	)
{
	__verifyRange(first1, last1);

#if defined(simd_stl_cpp_msvc)
	using _FirstIteratorUnwrappedType_	= std::_Unwrapped_t<_FirstIterator_>;
	using _SecondIteratorUnwrappedType_	= std::_Unwrapped_t<_SecondIterator_>;
#else 
	using _FirstIteratorUnwrappedType_	= _FirstIterator_;
	using _SecondIteratorUnwrappedType_ = _SecondIterator_;
#endif // defined(simd_stl_cpp_msvc) 

	auto first1Unwrapped		= __unwrapIterator(first1);
	const auto last1Unwrapped	= __unwrapIterator(last1);

	auto first2Unwrapped		= __unwrapIterator(first2);
	const auto last2Unwrapped	= __unwrapIterator(last2);


	if constexpr (type_traits::is_vectorized_search_algorithm_safe_v<
		_FirstIteratorUnwrappedType_, _SecondIteratorUnwrappedType_, _Predicate_>)
	{
#if simd_stl_has_cxx20
		if (type_traits::is_constant_evaluated() == false)
#endif
		{
			using _ValueType_ = type_traits::IteratorValueType<_FirstIterator_>;

			const auto firstRangeLength		= IteratorsDifference(first1Unwrapped, last1Unwrapped);
			const auto secondRangeLength	= IteratorsDifference(first2Unwrapped, last2Unwrapped);

			const auto position = MismatchVectorized<_ValueType_>(
				std::to_address(first1Unwrapped),
				std::to_address(first2Unwrapped),
				(std::min)(firstRangeLength, secondRangeLength));
				
			first1Unwrapped += static_cast<type_traits::IteratorDifferenceType<_FirstIterator_>>(position);
			first2Unwrapped += static_cast<type_traits::IteratorDifferenceType<_SecondIterator_>>(position);

#if defined(simd_stl_cpp_msvc)
			std::_Seek_wrapped(first1, first1Unwrapped);
			std::_Seek_wrapped(first2, first2Unwrapped);
#else 
			first1 = first1Unwrapped;
			first2 = first2Unwrapped;
#endif

			return { first1, first2 };
		}
	}

	while (first1Unwrapped != last1Unwrapped && predicate(*first1Unwrapped, *first2Unwrapped)) {
		++first1Unwrapped;
		++first2Unwrapped;
	}

#if defined(simd_stl_cpp_msvc)
	std::_Seek_wrapped(first1, first1Unwrapped);
	std::_Seek_wrapped(first2, first2Unwrapped);
#else 
	first1 = first1Unwrapped;
	first2 = first2Unwrapped;
#endif

	return { first1, first2 };
}

template <
	class _FirstIterator_,
	class _SecondIterator_>
simd_stl_nodiscard simd_stl_always_inline simd_stl_constexpr_cxx20 std::pair<_FirstIterator_, _SecondIterator_> mismatch(
	_FirstIterator_			first1,
	const _FirstIterator_	last1,
	_SecondIterator_		first2) noexcept(
		type_traits::is_nothrow_invocable_v<
			type_traits::equal_to<>,
			type_traits::IteratorValueType<_FirstIterator_>,
			type_traits::IteratorValueType<_SecondIterator_>
		>
	)
{
	return simd_stl::algorithm::mismatch(first1, last1, first2, type_traits::equal_to<>{});
}

template <
	class _FirstIterator_,
	class _SecondIterator_>
simd_stl_nodiscard simd_stl_always_inline simd_stl_constexpr_cxx20 std::pair<_FirstIterator_, _SecondIterator_> mismatch(
	_FirstIterator_			first1,
	const _FirstIterator_	last1,
	_SecondIterator_		first2,
	const _SecondIterator_	last2) noexcept(
		type_traits::is_nothrow_invocable_v<
			type_traits::equal_to<>,
			type_traits::IteratorValueType<_FirstIterator_>,
			type_traits::IteratorValueType<_SecondIterator_>
		>
	)
{
	return simd_stl::algorithm::mismatch(first1, last1, first2, last2, type_traits::equal_to<>{});
}

__SIMD_STL_ALGORITHM_NAMESPACE_END
