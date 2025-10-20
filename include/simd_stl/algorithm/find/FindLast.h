#pragma once 

#include <src/simd_stl/algorithm/AlgorithmDebug.h>
#include <src/simd_stl/type_traits/SimdAlgorithmSafety.h>

#include <src/simd_stl/algorithm/vectorized/FindLastVectorized.h>

#include <src/simd_stl/algorithm/MsvcIteratorUnwrap.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <
	class _Iterator_,
	class _Type_>
simd_stl_nodiscard simd_stl_always_inline simd_stl_constexpr_cxx20 _Iterator_ find_last(
	_Iterator_			first,
	const _Iterator_	last,
	const _Type_&		value) noexcept
{
	__verifyRange(first, last);

#if defined(simd_stl_cpp_msvc)
	using _IteratorUnwrappedType_ = std::_Unwrapped_t<_Iterator_>;
#else 
	using _IteratorUnwrappedType_ = _Iterator_;
#endif // defined(simd_stl_cpp_msvc) 

	if constexpr (type_traits::is_vectorized_find_algorithm_safe_v<_IteratorUnwrappedType_, _Type_>) {
		auto firstUnwrapped		= __unwrapIterator(first);
		auto lastUnwrapped		= __unwrapIterator(last);

#if simd_stl_has_cxx20
		if (type_traits::is_constant_evaluated() == false)
#endif
		{
			if (math::couldCompareEqualToValueType<_IteratorUnwrappedType_>(value) == false)
				return last;

			const auto firstAddress = std::to_address(firstUnwrapped);
			const auto position = FindLastVectorized(firstAddress, std::to_address(lastUnwrapped), value);

			if constexpr (std::is_pointer_v<_Iterator_>)
				__seekWrappedIterator(first, reinterpret_cast<const _Type_*>(position));
			else
				__seekWrappedIterator(first, firstUnwrapped + static_cast<type_traits::IteratorDifferenceType<_Iterator_>>(reinterpret_cast<const _Type_*>(position) - firstAddress));

			return first;
		}
	}

	for (; first != last; ++first)
		if (*first == value)
			break;

	return first;
}

template <
	class _InputIterator_,
	class _Predicate_>
simd_stl_nodiscard simd_stl_constexpr_cxx20 simd_stl_always_inline _InputIterator_ find_last_if_not(
	_InputIterator_	first, 
	_InputIterator_	last, 
	_Predicate_		predicate) noexcept(
		type_traits::is_nothrow_invocable_v<
			_Predicate_,
			type_traits::IteratorValueType<_InputIterator_>
		>
	)
{
	__verifyRange(first, last);

	auto firstUnwrapped	= __unwrapIterator(first);
	auto lastUnwrapped	= __unwrapIterator(last);

	while (lastUnwrapped != firstUnwrapped) {
		--lastUnwrapped;
		if (predicate(*lastUnwrapped) == false) {
			__seekWrappedIterator(first, lastUnwrapped);
			return first;
		}
	}

	return last;
}

template <
	class _InputIterator_,
	class _Predicate_>
simd_stl_nodiscard simd_stl_constexpr_cxx20 simd_stl_always_inline _InputIterator_ find_last_if(
	_InputIterator_	first, 
	_InputIterator_	last,
	_Predicate_		predicate) noexcept(
		type_traits::is_nothrow_invocable_v<
			_Predicate_,
			type_traits::IteratorValueType<_InputIterator_>
		>
	)
{
	__verifyRange(first, last);
	
	auto firstUnwrapped	= __unwrapIterator(first);
	auto lastUnwrapped	= __unwrapIterator(last);

	while (lastUnwrapped != firstUnwrapped) {
		--lastUnwrapped;
		if (predicate(*lastUnwrapped)) {
			__seekWrappedIterator(first, lastUnwrapped);
			return first;
		}
			
	}

	return last;
}

__SIMD_STL_ALGORITHM_NAMESPACE_END
