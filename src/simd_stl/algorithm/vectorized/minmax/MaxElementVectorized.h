#pragma once

#include <simd_stl/SimdStlNamespace.h>

#include <simd_stl/compatibility/Inline.h>
#include <simd_stl/compatibility/FunctionAttributes.h>

#include <simd_stl/compatibility/SimdCompatibility.h>
#include <simd_stl/arch/ProcessorFeatures.h>

#include <src/simd_stl/algorithm/AdvanceBytes.h>

#include <simd_stl/math/BitMath.h>
#include <simd_stl/compatibility/Inline.h>

#include <simd_stl/datapar/BasicSimd.h>



__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <class _Type_>
simd_stl_always_inline const void* _MaxElementScalar(
    const void* _First,
    const void* _Last) noexcept
{
    if (_First == _Last)
        return _Last;

    const _Type_* _FirstCasted = static_cast<const _Type_*>(_First);
    auto _Max = _FirstCasted;

    for (; ++_FirstCasted != _FirstCasted; )
        if (*_FirstCasted > *_Max)
            _Max = _FirstCasted;

    return _Max;
}

template <class _Type_>
simd_stl_always_inline _Type_* _MaxElementVectorized(
    const void* __first,
    const void* __last) noexcept
{
    
}

__SIMD_STL_ALGORITHM_NAMESPACE_END