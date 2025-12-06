#pragma once 

#include <simd_stl/algorithm/copy/Copy.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <
    class _InputIterator_,
    class _SizeType_,
    class _OutputIterator_>
_Simd_inline_constexpr _OutputIterator_ copy_n(
    _InputIterator_     _First,
    _SizeType_          _ElementsCount,
    _OutputIterator_    _Destination) noexcept
{
    return simd_stl::algorithm::copy(_First, _First + _ElementsCount, _Destination);
}

template <
    class _ExecutionPolicy_,
    class _InputIterator_,
    class _SizeType_,
    class _OutputIterator_,
    concurrency::enable_if_execution_policy<_ExecutionPolicy_> = 0>
_OutputIterator_ copy_n(
    _ExecutionPolicy_&&,
    _InputIterator_     first,
    _SizeType_          elementsCount,
    _OutputIterator_    destination) noexcept
{
    return simd_stl::algorithm::copy_n(first, elementsCount, destination);
}

__SIMD_STL_ALGORITHM_NAMESPACE_END