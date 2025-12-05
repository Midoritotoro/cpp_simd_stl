#pragma once 

#include <src/simd_stl/algorithm/AlgorithmDebug.h>
#include <src/simd_stl/type_traits/SimdAlgorithmSafety.h>

#include <src/simd_stl/algorithm/vectorized/CopyVectorized.h>
#include <src/simd_stl/algorithm/MsvcIteratorUnwrap.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <
    class _UnwrappedBidirectionalFirstIterator_,
    class _UnwrappedBidirectionalSecondIterator_>
_Simd_inline_constexpr _UnwrappedBidirectionalSecondIterator_ _CopyBackwardUnchecked(
    _UnwrappedBidirectionalFirstIterator_    _FirstUnwrapped,
    _UnwrappedBidirectionalFirstIterator_    _LastUnwrapped,
    _UnwrappedBidirectionalSecondIterator_   _DestinationLastUnwrapped) noexcept
{
    const auto _Difference = IteratorsDifference(_FirstUnwrapped, _LastUnwrapped);

    if constexpr (type_traits::IteratorCopyCategory<_UnwrappedBidirectionalFirstIterator_, _UnwrappedBidirectionalSecondIterator_>::BitcopyAssignable) {
#if simd_stl_has_cxx20
        if (type_traits::is_constant_evaluated() == false)
#endif
        {
            auto _FirstAddress = std::to_address(_FirstUnwrapped);
            const auto _Size = ByteLength(_FirstAddress, std::to_address(_LastUnwrapped));

            _MemcpyVectorized(const_cast<char*>(reinterpret_cast<const volatile char*>(
                std::to_address(_DestinationLastUnwrapped))) - _Size, _FirstAddress, _Size);

            return (_DestinationLastUnwrapped - _Difference);
        }
    }

     while (_FirstUnwrapped != _LastUnwrapped)
        *(--_DestinationLastUnwrapped) = *(--_LastUnwrapped);

    return _DestinationLastUnwrapped;
}

__SIMD_STL_ALGORITHM_NAMESPACE_END
