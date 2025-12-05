#pragma once

#include <simd_stl/numeric/BasicSimd.h>

__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <class _Type_>
simd_stl_declare_const_function simd_stl_always_inline sizetype simd_stl_stdcall _CountScalar(
    const void*     _FirstPointer,
    const sizetype  _Bytes,
    sizetype&       _Count,
    _Type_          _Value) noexcept
{
    auto _Current = static_cast<const _Type_*>(_FirstPointer);
    const auto _Length = _Bytes / sizeof(_Type_);

    for (sizetype _Index = 0; _Index < _Length; ++_Index)
        _Count += (*_Current++ == _Value);

    return _Count;
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Type_>
simd_stl_declare_const_function simd_stl_always_inline sizetype simd_stl_stdcall _CountVectorizedInternal(
    const void*     _FirstPointer,
    const sizetype  _Bytes,
    _Type_          _Value) noexcept
{
    using _SimdType_    = numeric::basic_simd<_SimdGeneration_, _Type_>;
    auto _AlignedSize   = _Bytes & (~(sizeof(_SimdType_) - 1));

    sizetype _Count = 0;

    if (_AlignedSize != 0) {
        const auto _Comparand = _SimdType_(_Value);

        for (sizetype _Index = 0; _Index < _AlignedSize; _Index += sizeof(_SimdType_)) {
            const auto _Loaded   = _SimdType_::loadUnaligned(static_cast<const char*>(_FirstPointer) + _Index);
            const auto _Compared = _Comparand.maskEqual(_Loaded);

            _Count += _Compared.countSet();
        }
    }
   
    AdvanceBytes(_FirstPointer, _AlignedSize);
    return _CountScalar(_FirstPointer, (_Bytes - _AlignedSize), _Count, _Value);
}

template <class _Type_>
simd_stl_declare_const_function sizetype simd_stl_stdcall _CountVectorized(
    const void*     _FirstPointer,
    const sizetype  _Bytes,
    _Type_          _Value) noexcept
{
    if constexpr (sizeof(_Type_) <= 2) {
        if (arch::ProcessorFeatures::AVX512BW())
            return CountVectorizedInternal<arch::CpuFeature::AVX512BW, _Type_>(_FirstPointer, _Bytes, _Value);
    }
    else {
        if (arch::ProcessorFeatures::AVX512F())
            return CountVectorizedInternal<arch::CpuFeature::AVX512F, _Type_>(_FirstPointer, _Bytes, _Value);
    }

    if (arch::ProcessorFeatures::AVX2())
        return CountVectorizedInternal<arch::CpuFeature::AVX2, _Type_>(_FirstPointer, _Bytes, _Value);
    else if (arch::ProcessorFeatures::SSE2())
        return CountVectorizedInternal<arch::CpuFeature::SSE2, _Type_>(_FirstPointer, _Bytes, _Value);

    sizetype _Count = 0;
    return _CountScalar(_FirstPointer, _Bytes, _Count, _Value);
}


__SIMD_STL_ALGORITHM_NAMESPACE_END
