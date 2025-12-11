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
    void* _FirstPointer,
    void* _LastPointer) noexcept
{
    using _SimdType_ = numeric::basic_simd<_SimdGeneration_, _Type_>;
    numeric::zero_upper_at_exit_guard<_SimdGeneration_> _Guard;

    const auto _AlignedSize  = ByteLength(_FirstPointer, _LastPointer) & (~((sizeof(_SimdType_) << 1) - 1));

    if (_AlignedSize != 0) {
        void* _StopAt = _FirstPointer;
        AdvanceBytes(_StopAt, _AlignedSize >> 1);

        do {
            auto _LoadedBegin  = _SimdType_::loadUnaligned(_FirstPointer);
            auto _LoadedEnd    = _SimdType_::loadUnaligned(static_cast<char*>(_LastPointer) - sizeof(_SimdType_));

            _LoadedBegin.reverse();
            _LoadedEnd.reverse();

            _LoadedBegin.storeUnaligned(static_cast<char*>(_LastPointer) - sizeof(_SimdType_));
            _LoadedEnd.storeUnaligned(_FirstPointer);

            AdvanceBytes(_FirstPointer, sizeof(_SimdType_));
            RewindBytes(_LastPointer, sizeof(_SimdType_));
        } while (_FirstPointer != _StopAt);
    }

    if (_FirstPointer != _LastPointer)
        _ReverseScalar<_Type_>(_FirstPointer, _LastPointer);
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
