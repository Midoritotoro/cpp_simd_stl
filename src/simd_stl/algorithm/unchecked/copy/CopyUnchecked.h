#pragma once 

#include <src/simd_stl/algorithm/AlgorithmDebug.h>
#include <src/simd_stl/type_traits/SimdAlgorithmSafety.h>

#include <src/simd_stl/algorithm/vectorized/copy/CopyVectorized.h>
#include <src/simd_stl/algorithm/MsvcIteratorUnwrap.h>

#include <simd_stl/concurrency/Execution.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <
    class _InputUnwrappedIterator_,
    class _OutputUnwrappedIterator_>
__simd_inline_constexpr _OutputUnwrappedIterator_ __copy_unchecked(
    _InputUnwrappedIterator_     __first_unwrapped,
    _InputUnwrappedIterator_     __last_unwrapped,
    _OutputUnwrappedIterator_    __destination_unwrapped) noexcept
{
    const auto __difference = __iterators_difference(__first_unwrapped, __last_unwrapped);

    if constexpr (type_traits::__copy_iterator_category<_InputUnwrappedIterator_, _OutputUnwrappedIterator_>::BitcopyAssignable) {
#if simd_stl_has_cxx20
        if (type_traits::is_constant_evaluated() == false)
#endif // simd_stl_has_cxx20
        {
            auto __first_address = std::to_address(__first_unwrapped);
 
            __memcpy_vectorized(std::to_address(__destination_unwrapped), __first_address,
                __byte_length(__first_address, std::to_address(__last_unwrapped)));

            return (__destination_unwrapped + __difference);
        }
    }

    for (; __first_unwrapped != __last_unwrapped; ++__destination_unwrapped, ++__first_unwrapped)
        *__destination_unwrapped = *__first_unwrapped;

    return __destination_unwrapped;
}

__SIMD_STL_ALGORITHM_NAMESPACE_END