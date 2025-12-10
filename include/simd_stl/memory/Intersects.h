#pragma once 

#include <simd_stl/Types.h>

#include <src/simd_stl/algorithm/MsvcIteratorUnwrap.h>
#include <src/simd_stl/algorithm/AdvanceBytes.h>


__SIMD_STL_MEMORY_NAMESPACE_BEGIN

template <
	typename _FirstForwardIterator_,
	typename _SecondForwardIterator_>
simd_stl_nodiscard simd_stl_always_inline bool intersects(
	const _FirstForwardIterator_	firstBegin,
	const _FirstForwardIterator_	firstEnd,
	const _SecondForwardIterator_	secondBegin) noexcept
{
	const auto firstRangeBeginAddress	= reinterpret_cast<const volatile char*>(std::to_address(firstBegin));
	const auto secondRangeBeginAddress	= reinterpret_cast<const volatile char*>(std::to_address(secondBegin));

	const auto firstRangeEndAddress		= reinterpret_cast<const volatile char*>(std::to_address(firstEnd));
	const auto secondRangeEndAddress	= reinterpret_cast<const volatile char*>(secondRangeBeginAddress) + algorithm::ByteLength(firstRangeBeginAddress, firstRangeEndAddress);

	return ((firstRangeBeginAddress > secondRangeBeginAddress) && (firstRangeBeginAddress < secondRangeEndAddress)) ||
		((secondRangeBeginAddress > firstRangeBeginAddress) && (secondRangeBeginAddress < firstRangeEndAddress));
}

template <
	typename _FirstForwardIterator_,
	typename _SecondForwardIterator_>
void _CheckIntersection(
	const _FirstForwardIterator_	firstBegin,
	const _FirstForwardIterator_	firstEnd,
	const _SecondForwardIterator_	secondBegin) noexcept
{
#if !defined(NDEBUG)
	using _FirstForwardIteratorUnwrappedType_	= algorithm::unwrapped_iterator_type<_FirstForwardIterator_>;
	using _SecondForwardIteratorUnwrappedType_	= algorithm::unwrapped_iterator_type<_SecondForwardIterator_>;

	if constexpr (
		type_traits::is_iterator_contiguous_v<_FirstForwardIteratorUnwrappedType_> &&
		type_traits::is_iterator_contiguous_v<_SecondForwardIteratorUnwrappedType_>)
	{
		DebugAssert(memory::intersects(
			std::move(firstBegin), std::move(firstEnd), std::move(secondBegin)) == false);
	}
#endif
}

__SIMD_STL_MEMORY_NAMESPACE_END
