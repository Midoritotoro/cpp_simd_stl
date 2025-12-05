#pragma once 

#include <src/simd_stl/algorithm/AlgorithmDebug.h>
#include <src/simd_stl/type_traits/SimdAlgorithmSafety.h>

#include <src/simd_stl/algorithm/vectorized/CopyVectorized.h>
#include <src/simd_stl/algorithm/MsvcIteratorUnwrap.h>

#include <simd_stl/concurrency/Execution.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <
    class _InputUnwrappedIterator_,
    class _OutputUnwrappedIterator_>
_Simd_inline_constexpr _OutputUnwrappedIterator_ _CopyUnchecked(
    _InputUnwrappedIterator_     _FirstUnwrapped,
    _InputUnwrappedIterator_     _LastUnwrapped,
    _OutputUnwrappedIterator_    _DestinationUnwrapped) noexcept
{
    const auto _Difference = IteratorsDifference(_FirstUnwrapped, _LastUnwrapped);

    if constexpr (type_traits::IteratorCopyCategory<_InputUnwrappedIterator_, _OutputUnwrappedIterator_>::BitcopyAssignable) {
#if simd_stl_has_cxx20
        if (type_traits::is_constant_evaluated() == false)
#endif // simd_stl_has_cxx20
        {
            auto _FirstAddress = std::to_address(_FirstUnwrapped);

            _MemcpyVectorized(std::to_address(_DestinationUnwrapped), _FirstAddress, 
                ByteLength(_FirstAddress, std::to_address(_LastUnwrapped)));

            return (_DestinationUnwrapped + _Difference);
        }
    }

    for (; _FirstUnwrapped != _LastUnwrapped; ++_DestinationUnwrapped, ++_FirstUnwrapped)
        *_DestinationUnwrapped = *_FirstUnwrapped;

    return _DestinationUnwrapped;
}

__SIMD_STL_ALGORITHM_NAMESPACE_END