#pragma once 

#include <simd_stl/compatibility/CompilerDetection.h>
#include <simd_stl/SimdStlNamespace.h>

#include <simd_stl/math/IntegralTypesConversions.h>
#include <xutility>

#include <src/simd_stl/algorithm/AlgorithmDebug.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <class _Iterator_>
simd_stl_nodiscard constexpr decltype(auto) _UnwrapIterator(_Iterator_&& iterator) 
    noexcept(type_traits::is_iterator_unwrappable_v<_Iterator_> == false || type_traits::is_nothrow_unwrappable_v<_Iterator_>)
{
    if constexpr (std::is_pointer_v<std::decay_t<_Iterator_>>)
        return iterator + 0;
    else if constexpr (type_traits::is_iterator_unwrappable_v<_Iterator_>)
        return std::move(iterator)._Unwrapped();
    else
        return std::move(iterator);
}

template <class _Iterator_>
using unwrapped_iterator_type = std::remove_cvref_t<decltype(_UnwrapIterator(std::declval<_Iterator_>()))>;

template <class _Iterator_>
simd_stl_nodiscard constexpr decltype(auto) _UnwrapUnverifiedIterator(_Iterator_&& iterator) 
    noexcept(type_traits::is_possibly_unverified_iterator_unwrappable_v<_Iterator_> == false)
{
    if constexpr (std::is_pointer_v<std::decay_t<_Iterator_>>)
        return iterator + 0;
    else if constexpr (type_traits::is_possibly_unverified_iterator_unwrappable_v<_Iterator_>)
        return std::move(iterator)._Unwrapped();
    else
        return std::move(iterator);
}

template <class _Iterator_>
using unwrapped_unverified_iterator_type = std::remove_cvref_t<decltype(_UnwrappedUnverifiedIterator(std::declval<_Iterator_>()))>;

template <
    class _Iterator_, 
    class _DifferenceType_>
simd_stl_nodiscard constexpr decltype(auto) _UnwrapIteratorOffset(
    _Iterator_&&            iterator,
    const _DifferenceType_  offset) noexcept(
        type_traits::is_possibly_unverified_iterator_unwrappable_v<_Iterator_> == false ||
        (type_traits::is_iterator_unwrappable_for_offset_v<_Iterator_> == false ||
            type_traits::is_iterator_unwrappable_for_offset_v<_Iterator_>)
    )
{
    if constexpr (std::is_pointer_v<std::decay_t<_Iterator_>>) {
        return iterator + 0;
    } 
    else if constexpr (
        type_traits::is_iterator_unwrappable_for_offset_v<_Iterator_> &&
        type_traits::is_nonbool_integral_v<_DifferenceType_>) 
    {
        using _IteratorDifferenceType_      = type_traits::IteratorDifferenceType<std::remove_cvref_t<_Iterator_>>;
        using _CommonDifferenceType_        = std::common_type_t<_DifferenceType_, _IteratorDifferenceType_>;

        const auto commonOffset = static_cast<_CommonDifferenceType_>(offset);

        constexpr auto maximum = math::MaximumIntegralLimit<_IteratorDifferenceType_>();
        constexpr auto minimum = math::MinimumIntegralLimit<_IteratorDifferenceType_>();


        DebugAssert(commonOffset <= static_cast<_CommonDifferenceType_>(maximum)
            && (std::is_unsigned_v<_DifferenceType_> || static_cast<_CommonDifferenceType_>(minimum) <= commonOffset),
            "integer overflow");

        iterator._Verify_offset(static_cast<_IteratorDifferenceType_>(offset));
        return std::move(iterator)._Unwrapped();
    } 
    else if constexpr (type_traits::is_possibly_unverified_iterator_unwrappable_v<_Iterator_>) {
        return std::move(iterator)._Unwrapped();
    } 
    else {
        return std::move(iterator);
    }
}

template <
    class _Iterator_,
    class _UnwrappedIterator_>
constexpr void _SeekPossiblyWrappedIterator(
    _Iterator_&             iterator,
    _UnwrappedIterator_&&   unwrappedIterator) noexcept(
        type_traits::is_wrapped_iterator_seekable_v<_Iterator_, _UnwrappedIterator_> == false || 
        type_traits::is_wrapped_iterator_nothrow_seekable_v<_Iterator_, _UnwrappedIterator_>
    )
{
    if constexpr (type_traits::is_wrapped_iterator_seekable_v<_Iterator_, _UnwrappedIterator_>)
        iterator._Seek_to(std::forward<_UnwrappedIterator_>(unwrappedIterator));
    else
        iterator = std::forward<_UnwrappedIterator_>(unwrappedIterator);
}

__SIMD_STL_ALGORITHM_NAMESPACE_END
