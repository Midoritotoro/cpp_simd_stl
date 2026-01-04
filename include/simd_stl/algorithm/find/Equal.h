#pragma once 

#include <src/simd_stl/algorithm/unchecked/find/EqualUnchecked.h>
#include <simd_stl/concurrency/Execution.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <
	class _FirstIterator_,
	class _SecondIterator_,
	class _Predicate_>
__simd_nodiscard_inline_constexpr bool equal(
	_FirstIterator_		_First1,
	_FirstIterator_		_Last1,
	_SecondIterator_	_First2,
	_Predicate_			_Predicate) noexcept(
		type_traits::is_nothrow_invocable_v<
			_Predicate_,
			type_traits::iterator_value_type<_FirstIterator_>,
			type_traits::iterator_value_type<_SecondIterator_>>)
{
	__verifyRange(_First1, _Last1);
	return _EqualUnchecked(_UnwrapIterator(_First1), _UnwrapIterator(_Last1),
		_UnwrapIterator(_First2), type_traits::passFunction(_Predicate));
}

template <
	class _FirstIterator_,
	class _SecondIterator_,
	class _Predicate_>
__simd_nodiscard_inline_constexpr bool equal(
	_FirstIterator_		_First1,
	_FirstIterator_		_Last1,
	_SecondIterator_	_First2,
	_SecondIterator_	_Last2,
	_Predicate_			_Predicate) noexcept(
		type_traits::is_nothrow_invocable_v<
			_Predicate_,
			type_traits::iterator_value_type<_FirstIterator_>,
			type_traits::iterator_value_type<_SecondIterator_>>)
{
	__verifyRange(_First1, _Last1);
	__verifyRange(_First2, _Last2);

	return _EqualUnchecked(_UnwrapIterator(_First1), _UnwrapIterator(_Last1),
		_UnwrapIterator(_First2), _UnwrapIterator(_Last2), type_traits::passFunction(_Predicate));
}

template <
	class _FirstIterator_,
	class _SecondIterator_>
__simd_nodiscard_inline_constexpr bool equal(
	_FirstIterator_		_First1,
	_FirstIterator_		_Last1,
	_SecondIterator_	_First2) noexcept(
		type_traits::is_nothrow_invocable_v<
			type_traits::equal_to<>,
			type_traits::iterator_value_type<_FirstIterator_>,
			type_traits::iterator_value_type<_SecondIterator_>>)
{
	return simd_stl::algorithm::equal(_First1, _Last1, _First2, type_traits::equal_to<>{});
}

template <
	class _FirstIterator_,
	class _SecondIterator_>
__simd_nodiscard_inline_constexpr bool equal(
	_FirstIterator_		_First1,
	_FirstIterator_		_Last1,
	_SecondIterator_	_First2,
	_SecondIterator_	_Last2) noexcept(
		type_traits::is_nothrow_invocable_v<
			type_traits::equal_to<>,
			type_traits::iterator_value_type<_FirstIterator_>,
			type_traits::iterator_value_type<_SecondIterator_>>)
{
	return simd_stl::algorithm::equal(_First1, _Last1, _First2, _Last2, type_traits::equal_to<>{});
}

template <
	class _ExecutionPolicy_,
	class _FirstIterator_,
	class _SecondIterator_,
	class _Predicate_,
	concurrency::enable_if_execution_policy<_ExecutionPolicy_> = 0>
__simd_nodiscard_inline_constexpr bool equal(
	_ExecutionPolicy_&&,
	_FirstIterator_		_First1,
	_FirstIterator_		_Last1,
	_SecondIterator_	_First2,
	_Predicate_			_Predicate) noexcept(
		type_traits::is_nothrow_invocable_v<
			_Predicate_,
			type_traits::iterator_value_type<_FirstIterator_>,
			type_traits::iterator_value_type<_SecondIterator_>>)
{
	return simd_stl::algorithm::equal(_First1, _Last1, _First2, type_traits::passFunction(_Predicate));
}

template <
	class _ExecutionPolicy_,
	class _FirstIterator_,
	class _SecondIterator_,
	class _Predicate_,
	concurrency::enable_if_execution_policy<_ExecutionPolicy_> = 0>
__simd_nodiscard_inline_constexpr bool equal(
	_ExecutionPolicy_&&,
	_FirstIterator_		_First1,
	_FirstIterator_		_Last1,
	_SecondIterator_	_First2,
	_SecondIterator_	_Last2,
	_Predicate_			_Predicate) noexcept(
		type_traits::is_nothrow_invocable_v<
			_Predicate_,
			type_traits::iterator_value_type<_FirstIterator_>,
			type_traits::iterator_value_type<_SecondIterator_>>)
{
	return simd_stl::algorithm::equal(_First1, _Last1, _First2, _Last2, type_traits::passFunction(_Predicate));
}

template <
	class _ExecutionPolicy_,
	class _FirstIterator_,
	class _SecondIterator_,
	concurrency::enable_if_execution_policy<_ExecutionPolicy_> = 0>
__simd_nodiscard_inline_constexpr bool equal(
	_ExecutionPolicy_&&,
	_FirstIterator_		_First1,
	_FirstIterator_		_Last1,
	_SecondIterator_	_First2) noexcept(
		type_traits::is_nothrow_invocable_v<
			type_traits::equal_to<>,
			type_traits::iterator_value_type<_FirstIterator_>,
			type_traits::iterator_value_type<_SecondIterator_>>)
{
	return simd_stl::algorithm::equal(_First1, _Last1, _First2);
}


template <
	class _ExecutionPolicy_,
	class _FirstIterator_,
	class _SecondIterator_,
	concurrency::enable_if_execution_policy<_ExecutionPolicy_> = 0>
__simd_nodiscard_inline_constexpr  bool equal(
	_ExecutionPolicy_&&,
	_FirstIterator_		_First1,
	_FirstIterator_		_Last1,
	_SecondIterator_	_First2,
	_SecondIterator_	_Last2) noexcept(
		type_traits::is_nothrow_invocable_v<
			type_traits::equal_to<>,
			type_traits::iterator_value_type<_FirstIterator_>,
			type_traits::iterator_value_type<_SecondIterator_>>)
{
	return simd_stl::algorithm::equal(_First1, _Last1, _First2, _Last2);
}

__SIMD_STL_ALGORITHM_NAMESPACE_END
