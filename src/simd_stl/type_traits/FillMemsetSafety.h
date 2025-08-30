#pragma once 

#include <src/simd_stl/type_traits/TypeTraits.h>
#include <simd_stl/compatibility/FunctionAttributes.h>

#include <src/simd_stl/type_traits/IntegralProperties.h>

__SIMD_STL_TYPE_TRAITS_NAMESPACE_BEGIN

template <
    class _ForwardIterator_,
    class _Type_,
    bool = is_iterator_contiguous_v<_ForwardIterator_>>
constexpr bool is_fill_memset_safe_v = std::conjunction_v<
    std::is_scalar<_Type_>,
    is_character_or_byte_or_bool_v<
        unwrap_enum_t<
            std::remove_reference_t<
                IteratorReferenceType<_ForwardIterator_>>>>,
    std::negation<
        std::is_volatile<
            std::remove_reference_t<
                IteratorReferenceType<_ForwardIterator_>>>>,
    std::is_assignable<
        IteratorReferenceType<_ForwardIterator_>,
    const _Type_ &>>;

template <
    class _ForwardIterator_,
    class _Type_>
constexpr bool is_fill_memset_safe_v<_ForwardIterator_, _Type_, false> = false;

template <
    class _ForwardIterator_,
    class _Type_,
    bool = is_iterator_contiguous_v<_ForwardIterator_>>
constexpr bool is_fill_zero_memset_safe_v =
    std::conjunction_v<
        std::is_scalar<_Type_>, 
        std::is_scalar<
            IteratorValueType<_ForwardIterator_>>,
        std::negation<
            std::is_member_pointer<
                IteratorValueType<_ForwardIterator_>>>,
        std::negation<
            std::is_volatile<
                std::remove_reference_t<
                    IteratorReferenceType<_ForwardIterator_>>>>,
        std::is_assignable<IteratorReferenceType<_ForwardIterator_>,
    const _Type_&>>;

template <
    class _ForwardIterator_,
    class _Type_>
constexpr bool is_fill_zero_memset_safe_v<_ForwardIterator_, _Type_, false> = false;

__SIMD_STL_TYPE_TRAITS_NAMESPACE_END
