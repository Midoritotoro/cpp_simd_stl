#pragma once 

#include <simd_stl/Types.h>

__SIMD_STL_NUMERIC_NAMESPACE_BEGIN

template <
    sizetype _VerticalSize_,
    sizetype _HorizontalSize_>
struct _ShuffleTables {
    uint8 _Shuffle[_VerticalSize_][_HorizontalSize_];
    uint8 _Size[_VerticalSize_];
};

template <
    sizetype _VerticalSize_,
    sizetype _HorizontalSize_>
constexpr auto _MakeShuffleTables(
    const uint32 _Multiplier,
    const uint32 _ElementGroupStride) noexcept
{
    _ShuffleTables<_VerticalSize_, _HorizontalSize_> _Result;

    for (uint32 _VerticalIndex = 0; _VerticalIndex != _VerticalSize_; ++_VerticalIndex) {
        uint32 _ActiveGroupCount = 0;

        for (uint32 _HorizontalIndex = 0; _HorizontalIndex != _HorizontalSize_ / _ElementGroupStride; ++_HorizontalIndex) {
            if ((_VerticalIndex & (1 << _HorizontalIndex)) == 0) {
                for (uint32 _ElementOffset = 0; _ElementOffset != _ElementGroupStride; ++_ElementOffset)
                    _Result._Shuffle[_VerticalIndex][_ActiveGroupCount * _ElementGroupStride + _ElementOffset] =
                    static_cast<uint8>(_HorizontalIndex * _ElementGroupStride + _ElementOffset);

                ++_ActiveGroupCount;
            }
        }

        _Result._Size[_VerticalIndex] = static_cast<uint8>(_ActiveGroupCount * _Multiplier);


        for (; _ActiveGroupCount != _HorizontalSize_ / _ElementGroupStride; ++_ActiveGroupCount)
            for (uint32 _ElementOffset = 0; _ElementOffset != _ElementGroupStride; ++_ElementOffset)
                _Result._Shuffle[_VerticalIndex][_ActiveGroupCount * _ElementGroupStride + _ElementOffset] =
                static_cast<uint8>(_ActiveGroupCount * _ElementGroupStride + _ElementOffset);
    }

    return _Result;
}


constexpr auto _Tables8BitSse   = _MakeShuffleTables<256, 8>(1, 1);
constexpr auto _Tables16BitSse  = _MakeShuffleTables<256, 16>(2, 2);
constexpr auto _Tables32BitSse  = _MakeShuffleTables<16, 16>(4, 4);
constexpr auto _Tables64BitSse  = _MakeShuffleTables<4, 16>(8, 8);

constexpr auto _Tables32BitAvx = _MakeShuffleTables<256, 8>(4, 1);
constexpr auto _Tables64BitAvx = _MakeShuffleTables<16, 8>(8, 2);

template <
    class _VectorType_,
    class _DesiredType_>
struct _SimdInsertMask {
    _DesiredType_   _Array[(sizeof(_VectorType_) / sizeof(_DesiredType_))];
    int32           _Offset = 0;
};

template <
    class _VectorType_,
    class _DesiredType_>
constexpr auto _MakeSimdInsertMask() noexcept { 
    constexpr auto _Length = sizeof(_VectorType_) / sizeof(_DesiredType_);
    _SimdInsertMask<_VectorType_, _DesiredType_> _InsertMask;

    for (auto _Index = 0; _Index < _Length; ++_Index)
        _InsertMask._Array[_Index] = 0;

    _InsertMask._Offset = _Length >> 1;
    _InsertMask._Array[(_InsertMask._Offset + 1)] = -1;

    return _InsertMask;
}

__SIMD_STL_NUMERIC_NAMESPACE_END
