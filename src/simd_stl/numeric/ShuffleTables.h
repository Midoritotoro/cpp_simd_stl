#pragma once 

#include <simd_stl/Types.h>

__SIMD_STL_NUMERIC_NAMESPACE_BEGIN

template <
    sizetype _VerticalSize_,
    sizetype _HorizontalSize_>
struct __shuffle_tables {
    uint8 __shuffle[_VerticalSize_][_HorizontalSize_];
    uint8 __size[_VerticalSize_];
    uint8 __unprocessed_tail[_VerticalSize_][_HorizontalSize_];
};

template <
    sizetype _VerticalSize_,
    sizetype _HorizontalSize_>
constexpr auto __make_shuffle_tables(
    const uint32 __multiplier,
    const uint32 __element_group_stride) noexcept
{
    auto __result = __shuffle_tables<_VerticalSize_, _HorizontalSize_>();

    for (auto __vertical_index = uint32(0); __vertical_index != _VerticalSize_; ++__vertical_index) {
        auto __active_group_count = uint32(0);

        for (auto __horizontal_index = uint32(0); __horizontal_index != _HorizontalSize_ / __element_group_stride; ++__horizontal_index) {
            if ((__vertical_index & (1 << __horizontal_index)) == 0) {
                for (auto __element_offset = uint32(0); __element_offset != __element_group_stride; ++__element_offset)
                    __result.__shuffle[__vertical_index][__active_group_count * __element_group_stride + __element_offset] =
                        static_cast<uint8>(__horizontal_index * __element_group_stride + __element_offset);

                ++__active_group_count;
            }
        }

        __result.__size[__vertical_index] = static_cast<uint8>(__active_group_count * __multiplier);

        for (; __active_group_count != _HorizontalSize_ / __element_group_stride; ++__active_group_count)
            for (auto __element_offset = uint32(0); __element_offset != __element_group_stride; ++__element_offset)
                __result.__shuffle[__vertical_index][__active_group_count * __element_group_stride + __element_offset] =
                    static_cast<uint8>(__active_group_count * __element_group_stride + __element_offset);

        for (auto __inactive_group_count = uint32(0); __inactive_group_count < _HorizontalSize_; ++__inactive_group_count)
            __result.__unprocessed_tail[__vertical_index][__inactive_group_count] = static_cast<uint8>(__inactive_group_count);
    }

    return __result;
}

template <sizetype _Size_> 
constexpr auto __tables_sse = [] { 
    static_assert(_Size_ == 1 || _Size_ == 2 || _Size_ == 4 || _Size_ == 8, "Unsupported element size for __tables_sse");
    return __shuffle_tables<1, 1>();
}();

template <>
constexpr auto __tables_sse<1>  = __make_shuffle_tables<256, 8>(1, 1);

template <>
constexpr auto __tables_sse<2>  = __make_shuffle_tables<256, 16>(2, 2);

template <>
constexpr auto __tables_sse<4>  = __make_shuffle_tables<16, 16>(4, 4);

template <>
constexpr auto __tables_sse<8>  = __make_shuffle_tables<4, 16>(8, 8);

template <sizetype _Size_>
constexpr auto __tables_avx = [] { 
    static_assert(_Size_ == 1 || _Size_ == 2 || _Size_ == 4 || _Size_ == 8, "Unsupported element size for __tables_avx");
    return __shuffle_tables<1, 1>();
}();

template <>
constexpr auto __tables_avx<4> = __make_shuffle_tables<256, 8>(4, 1);

template <>
constexpr auto __tables_avx<8> = __make_shuffle_tables<16, 8>(8, 2);

template <
    class _VectorType_,
    class _DesiredType_>
struct __insert_mask {
    _DesiredType_   __array[(sizeof(_VectorType_) / sizeof(_DesiredType_))];
    int32           __offset = 0;
};

template <
    class _VectorType_,
    class _DesiredType_>
constexpr auto __simd_make_insert_mask() noexcept { 
    constexpr auto __length = sizeof(_VectorType_) / sizeof(_DesiredType_);
    auto __mask = __insert_mask<_VectorType_, _DesiredType_>();

    for (auto __index = 0; __index < __length; ++__index)
        __mask.__array[__index] = 0;

    __mask.__offset                         = __length >> 1;
    __mask.__array[(__mask.__offset + 1)]   = -1;

    return __mask;
}

__SIMD_STL_NUMERIC_NAMESPACE_END
