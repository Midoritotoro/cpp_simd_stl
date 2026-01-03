#pragma once

#include <simd_stl/numeric/BasicSimd.h>

__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <class _Type_>
simd_stl_always_inline void _ReverseScalar(
    void*   _FirstPointer,
    void*   _LastPointer) noexcept
{
    auto _First  = static_cast<_Type_*>(_FirstPointer);
    auto _Last   = static_cast<_Type_*>(_LastPointer);

    for (; _First != _Last && _First != --_Last; ++_First) {
        _Type_ _Temp = *_Last;

        *_Last = *_First;
        *_First = _Temp;
    }
        
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Type_>
simd_stl_declare_const_function simd_stl_always_inline void _ReverseVectorizedInternal(
    void* _First,
    void* _Last) noexcept
{
    using _SimdType_ = numeric::simd<_SimdGeneration_, _Type_>;
    numeric::zero_upper_at_exit_guard<_SimdGeneration_> _Guard;

    const auto _AlignedSize  = __byte_length(_First, _Last) & (~((sizeof(_SimdType_) << 1) - 1));

    if (_AlignedSize != 0) {
        void* _StopAt = _First;
        __advance_bytes(_StopAt, _AlignedSize >> 1);

        do {
            __rewind_bytes(_Last, sizeof(_SimdType_));

            auto _LoadedBegin  = _SimdType_::loadUnaligned(_First);
            auto _LoadedEnd    = _SimdType_::loadUnaligned(_Last);

            _LoadedBegin.reverse();
            _LoadedEnd.reverse();

            _LoadedBegin.storeUnaligned(_Last);
            _LoadedEnd.storeUnaligned(_First);

            __advance_bytes(_First, sizeof(_SimdType_));
        } while (_First != _StopAt);
    }

    if (_First != _Last)
        _ReverseScalar<_Type_>(_First, _Last);
}

template <class _Type_>
void _ReverseVectorized(
    void* _FirstPointer,
    void* _LastPointer) noexcept
{
    if constexpr (sizeof(_Type_) == 2) {
        if (arch::ProcessorFeatures::AVX512BW())
            return _ReverseVectorizedInternal<arch::CpuFeature::AVX512BW, _Type_>(_FirstPointer, _LastPointer);
    }
    else if constexpr (sizeof(_Type_) >= 4) {
        if (arch::ProcessorFeatures::AVX512F())
            return _ReverseVectorizedInternal<arch::CpuFeature::AVX512F, _Type_>(_FirstPointer, _LastPointer);
    }

    if (arch::ProcessorFeatures::AVX2())
        return _ReverseVectorizedInternal<arch::CpuFeature::AVX2, _Type_>(_FirstPointer, _LastPointer);
    else if (arch::ProcessorFeatures::SSSE3())
        return _ReverseVectorizedInternal<arch::CpuFeature::SSSE3, _Type_>(_FirstPointer, _LastPointer);
    else if (arch::ProcessorFeatures::SSE2())
        return _ReverseVectorizedInternal<arch::CpuFeature::SSE2, _Type_>(_FirstPointer, _LastPointer);

    _ReverseScalar<_Type_>(_FirstPointer, _LastPointer);
}


__SIMD_STL_ALGORITHM_NAMESPACE_END
