#pragma once 

#include <src/simd_stl/algorithm/vectorized/transform/TransformVectorized.h>
#include <src/simd_stl/algorithm/AlgorithmDebug.h>

#include <src/simd_stl/type_traits/SimdAlgorithmSafety.h>
#include <src/simd_stl/algorithm/MsvcIteratorUnwrap.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <
    class _UnwrappedInputIterator_,
    class _UnwrappedOutputIterator_,
    class _UnaryPredicate_>
_Simd_nodiscard_inline_constexpr _UnwrappedOutputIterator_ _TransformUnchecked(
    _UnwrappedInputIterator_    _FirstUnwrapped,
    _UnwrappedInputIterator_    _LastUnwrapped,
    _UnwrappedOutputIterator_   _DestinationUnwrapped,
    _UnaryPredicate_            _Predicate) noexcept
{
    using _IteratorValueType = type_traits::IteratorValueType<_UnwrappedInputIterator_>;

    constexpr auto _Is_vectorizable = type_traits::is_vectorized_find_algorithm_safe_v<_UnwrappedInputIterator_, _IteratorValueType>
        && type_traits::is_vectorized_find_algorithm_safe_v<_UnwrappedOutputIterator_, _IteratorValueType>;

    if constexpr (_Is_vectorizable) {
#if simd_stl_has_cxx20
        if (type_traits::is_constant_evaluated() == false)
#endif // simd_stl_has_cxx20
        {
            auto _DestinationAddress = std::to_address(_DestinationUnwrapped);

            const auto _DestinationLast = _TransformVectorized<_IteratorValueType>(std::to_address(_FirstUnwrapped),
                std::to_address(_LastUnwrapped), _DestinationAddress,
                type_traits::passFunction(_Predicate));
            
            if constexpr (std::is_pointer_v<_UnwrappedOutputIterator_>)
                return _DestinationLast;
            else
                return _DestinationUnwrapped + (_DestinationLast - _DestinationAddress);
        }
    }

    for (; _FirstUnwrapped != _LastUnwrapped; ++_FirstUnwrapped, ++_DestinationUnwrapped)
        *_DestinationUnwrapped = _Predicate(*_FirstUnwrapped);

    return _DestinationUnwrapped;
}

template <
    class _UnwrappedFirstInputIterator_,
    class _UnwrappedSecondInputIterator_,
    class _UnwrappedOutputIterator_,
    class _BinaryPredicate_>
_Simd_nodiscard_inline_constexpr _UnwrappedOutputIterator_ _TransformUnchecked(
    _UnwrappedFirstInputIterator_   _First1Unwrapped,
    _UnwrappedFirstInputIterator_   _Last1Unwrapped,
    _UnwrappedSecondInputIterator_  _First2Unwrapped,
    _UnwrappedOutputIterator_       _DestinationUnwrapped,
    _BinaryPredicate_               _Predicate) noexcept
{
    using _IteratorValueType = type_traits::IteratorValueType<_UnwrappedFirstInputIterator_>;

    constexpr auto _Is_vectorizable = type_traits::is_vectorized_find_algorithm_safe_v<_UnwrappedFirstInputIterator_, _IteratorValueType>
        && type_traits::is_vectorized_find_algorithm_safe_v<_UnwrappedOutputIterator_, _IteratorValueType>
        && type_traits::is_vectorized_find_algorithm_safe_v<_UnwrappedSecondInputIterator_, _IteratorValueType>;

    if constexpr (_Is_vectorizable) {
#if simd_stl_has_cxx20
        if (type_traits::is_constant_evaluated() == false)
#endif // simd_stl_has_cxx20
        {
            auto _DestinationAddress = std::to_address(_DestinationUnwrapped);

            const auto _DestinationLast = _TransformVectorized<_IteratorValueType>(std::to_address(_First1Unwrapped),
                std::to_address(_Last1Unwrapped), std::to_address(_First2Unwrapped), _DestinationAddress,
                type_traits::passFunction(_Predicate));
            
            if constexpr (std::is_pointer_v<_UnwrappedOutputIterator_>)
                return _DestinationLast;
            else
                return _DestinationUnwrapped + (_DestinationLast - _DestinationAddress);
        }
    }

    for (; _First1Unwrapped != _Last1Unwrapped; ++_First1Unwrapped, ++_First2Unwrapped, ++_DestinationUnwrapped)
        *_DestinationUnwrapped = _Predicate(*_First1Unwrapped, *_First2Unwrapped);

    return _DestinationUnwrapped;
}

__SIMD_STL_ALGORITHM_NAMESPACE_END
