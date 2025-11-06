#pragma once

#include <simd_stl/SimdStlNamespace.h>

#include <simd_stl/compatibility/Inline.h>
#include <simd_stl/compatibility/FunctionAttributes.h>

#include <simd_stl/compatibility/SimdCompatibility.h>
#include <simd_stl/arch/ProcessorFeatures.h>

#include <src/simd_stl/algorithm/AdvanceBytes.h>

#include <simd_stl/math/BitMath.h>
#include <simd_stl/numeric/BasicSimd.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <class _Simd_>
constexpr int removeStep() noexcept {
    using _ValueType_ = typename _Simd_::value_type;

    if constexpr (arch::__is_xmm_v<_Simd_::_Generation>) {
        if constexpr (numeric::is_epi8_v<_ValueType_> || numeric::is_epu8_v<_ValueType_>)
            return 8;
        else
            return 16;
    }
}

template <class _Type_>
simd_stl_declare_const_function simd_stl_always_inline const void* _RemoveScalar(
    void*       first,
    const void* current,
    const void* last,
    _Type_      value) noexcept
{
    auto currentCasted  = static_cast<const _Type_*>(current);
    auto firstCasted    = static_cast<_Type_*>(first);

    for (; currentCasted != last; ++currentCasted) {
        const auto currentValue = *currentCasted;

        if (currentValue != value)
            *firstCasted++ = currentValue;
    }

    return firstCasted;
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Type_>
simd_stl_always_inline const void* _RemoveVectorizedInternal(
    void*       first,
    const void* last,
    _Type_      value) noexcept
{
    using _SimdType_ = numeric::basic_simd<_SimdGeneration_, _Type_>;

    constexpr auto step = removeStep<_SimdType_>();

    const auto size         = ByteLength(first, last);
    const auto alignedSize  = size & (~(step - 1));

    void* current = first;

    if (alignedSize != 0) {
        const auto comparand    = _SimdType_(value);

        const void* stopAt = first;
        AdvanceBytes(stopAt, alignedSize);

        do {
            _SimdType_ loaded /*= _SimdType_::template basic_simd<false>()*/;

            if constexpr (numeric::is_epi8_v<_Type_> || numeric::is_epu8_v<_Type_>)
                loaded = _SimdType_::loadLowerHalf(current);
            else 
                loaded = _SimdType_::loadUnaligned(current);

            const auto mask = comparand.maskEqual(loaded);

            if constexpr (numeric::is_epi8_v<_Type_> || numeric::is_epu8_v<_Type_>)
                first = loaded.compressStoreLowerHalf(first, mask.unwrap());
            else
                first = loaded.compressStoreUnaligned(first, mask.unwrap());
            
            AdvanceBytes(current, step);
        } while (current != stopAt);
    }

    return (current == last) ? first : _RemoveScalar<_Type_>(first, current, last, value);
}

template <class _Type_>
simd_stl_declare_const_function simd_stl_always_inline const void* _RemoveVectorized(
    void*       first,
    const void* last,
    _Type_      value) noexcept
{
    if (arch::ProcessorFeatures::SSSE3())
        return _RemoveVectorizedInternal<arch::CpuFeature::SSSE3, _Type_>(first, last, value);

    return _RemoveScalar<_Type_>(first, first, last, value);
}


__SIMD_STL_ALGORITHM_NAMESPACE_END
