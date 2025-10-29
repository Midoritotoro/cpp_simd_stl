#pragma once 

#include <simd_stl/Types.h>
#include <src/simd_stl/algorithm/MsvcIteratorUnwrap.h>

__SIMD_STL_MEMORY_NAMESPACE_BEGIN

template <typename _ForwardIterator_>
simd_stl_nodiscard simd_stl_always_inline bool intersects(
	const _ForwardIterator_ firstBegin,
	const _ForwardIterator_ firstEnd,
	const _ForwardIterator_ secondBegin,
	const _ForwardIterator_ secondEnd) noexcept
{
	using _ForwardIteratorUnwrapped_ = algorithm::unwrapped_iterator_type<_ForwardIterator_>;

	const auto firstBeginUnwrapped	= algorithm::_UnwrapIterator(firstBegin);
	const auto secondBeginUnwrapped	= algorithm::_UnwrapIterator(secondBegin);

	const auto firstEndUnwrapped	= algorithm::_UnwrapIterator(secondBegin);
	const auto secondEndUnwrapped	= algorithm::_UnwrapIterator(secondEnd);

	const auto firstRangeBeginAddress	= std::to_address(firstBeginUnwrapped);
	const auto secondRangeBeginAddress	= std::to_address(secondBeginUnwrapped);

	const auto firstRangeEndAddress		= std::to_address(firstRangeBeginAddress);
	const auto secondRangeEndAddress	= std::to_address(secondRangeBeginAddress);

	return ((firstRangeBeginAddress > secondRangeBeginAddress) && (firstRangeBeginAddress < secondRangeEndAddress)) ||
		((secondRangeBeginAddress > firstRangeBeginAddress) && (secondRangeBeginAddress < firstRangeEndAddress));
}

template <typename _ForwardIterator_>
simd_stl_nodiscard simd_stl_always_inline bool intersects(
	const _ForwardIterator_	firstBegin,
	const _ForwardIterator_	secondBegin,
	const sizetype			bytes) noexcept
{
	using _ForwardIteratorUnwrapped_ = algorithm::unwrapped_iterator_type<_ForwardIterator_>;

	const auto firstBeginUnwrapped	= algorithm::_UnwrapIteratorBytesOffset(firstBegin, bytes);
	const auto secondBeginUnwrapped	= algorithm::_UnwrapIteratorBytesOffset(secondBegin, bytes);

	const auto firstRangeAddress	= std::to_address(firstBeginUnwrapped);
	const auto secondRangeAddress	= std::to_address(secondBeginUnwrapped);

	const auto firstRangeAddressChar	= const_cast<const char*>(reinterpret_cast<const volatile char* const>(firstRangeAddress));
	const auto secondRangeAddressChar	= const_cast<const char*>(reinterpret_cast<const volatile char* const>(secondRangeAddress));

	return ((firstRangeAddressChar > secondRangeAddressChar) && (firstRangeAddressChar < (secondRangeAddressChar + bytes))) ||
		((secondRangeAddressChar > firstRangeAddressChar) && (secondRangeAddressChar < (firstRangeAddressChar + bytes)));
}

__SIMD_STL_MEMORY_NAMESPACE_END
