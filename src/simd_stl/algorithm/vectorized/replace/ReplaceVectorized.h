#pragma once


#include <simd_stl/numeric/BasicSimd.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <typename _Type_>
simd_stl_always_inline void simd_stl_stdcall _ReplaceScalar(
    void*           _First,
    void*           _Last,
    const _Type_&   _OldValue,
    const _Type_&   _NewValue) noexcept
{
    auto _Current = static_cast<_Type_*>(_First);

    for (; _Current != _Last; ++_Current)
        if (*_Current == _OldValue)
            *_Current = _NewValue;
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Type_>
simd_stl_always_inline void simd_stl_stdcall _ReplaceVectorizedInternal(
    void*           _First,
    void*           _Last,
    const _Type_&   _OldValue,
    const _Type_&   _NewValue) noexcept
{
    using _Simd_ = numeric::basic_simd<_SimdGeneration_, _Type_>;

    const auto _AlignedSize = ByteLength(_First, _Last) & (~(sizeof(_SimdType_) - 1));

    if (_AlignedSize != 0) {
        void* _StopAt = _First;
        AdvanceBytes(_StopAt, _AlignedSize);

        const auto _Comparand   = _SimdType_(_OldValue);
        const auto _Replacement = _SimdType_(_NewValue);

        do {
            const auto _Loaded = _SimdType_::loadUnaligned(_First);
            const auto _Compared = _SimdType_::equal
        }
    }
}

template <typename _Type_>
simd_stl_declare_const_function void simd_stl_stdcall ReplaceVectorized(
    void*           _First,
    void*           _Last,
    const _Type_&   _OldValue,
    const _Type_&   _NewValue) noexcept
{

}

__SIMD_STL_ALGORITHM_NAMESPACE_END

