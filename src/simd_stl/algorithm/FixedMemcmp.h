#pragma once 

#include <simd_stl/compatibility/FunctionAttributes.h>
#include <simd_stl/compatibility/Inline.h>

#include <simd_stl/Types.h>

__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN


simd_stl_declare_const_function simd_stl_always_inline bool alwaysTrue(
    const void*,
    const void*) noexcept
{
    return true;
}

simd_stl_declare_const_function simd_stl_always_inline bool memcmp1(
    const void* first,
    const void* second) noexcept
{
    return static_cast<const char*>(first)[0] == static_cast<const char*>(second)[0];
}

simd_stl_declare_const_function simd_stl_always_inline bool memcmp2(
    const void* first,
    const void* second) noexcept
{
    const uint16 __first    = *reinterpret_cast<const uint16*>(first);
    const uint16 __second   = *reinterpret_cast<const uint16*>(second);

    return __first == __second;
}

simd_stl_declare_const_function simd_stl_always_inline bool memcmp3(
    const void* first,
    const void* second) noexcept
{
    const uint32 __first    = *reinterpret_cast<const uint32*>(first);
    const uint32 __second   = *reinterpret_cast<const uint32*>(second);

    return (__first & 0x00ffffff) == (__second & 0x00ffffff);
}

simd_stl_declare_const_function simd_stl_always_inline bool memcmp4(
    const void* first,
    const void* second) noexcept
{
    const uint32 __first    = *reinterpret_cast<const uint32*>(first);
    const uint32 __second   = *reinterpret_cast<const uint32*>(second);

    return __first == __second;
}

simd_stl_declare_const_function simd_stl_always_inline bool memcmp5(
    const void* first,
    const void* second) noexcept
{
    const uint64 __first    = *reinterpret_cast<const uint64*>(first);
    const uint64 __second   = *reinterpret_cast<const uint64*>(second);

    return ((__first ^ __second) & 0x000000fffffffffflu) == 0;
}

simd_stl_declare_const_function simd_stl_always_inline bool memcmp6(
    const void* first,
    const void* second) noexcept
{
    const uint64 __first    = *reinterpret_cast<const uint64*>(first);
    const uint64 __second   = *reinterpret_cast<const uint64*>(second);

    return ((__first ^ __second) & 0x0000fffffffffffflu) == 0;
}

simd_stl_declare_const_function simd_stl_always_inline bool memcmp7(
    const void* first,
    const void* second) noexcept
{
    const uint64 __first    = *reinterpret_cast<const uint64*>(first);
    const uint64 __second   = *reinterpret_cast<const uint64*>(second);

    return ((__first ^ __second) & 0x00fffffffffffffflu) == 0;
}

simd_stl_declare_const_function simd_stl_always_inline bool memcmp8(
    const void* first,
    const void* second) noexcept
{
    const uint64 __first    = *reinterpret_cast<const uint64*>(first);
    const uint64 __second   = *reinterpret_cast<const uint64*>(second);

    return __first == __second;
}

simd_stl_declare_const_function simd_stl_always_inline bool memcmp9(
    const void* first,
    const void* second) noexcept
{
    const uint64 __first    = *reinterpret_cast<const uint64*>(first);
    const uint64 __second   = *reinterpret_cast<const uint64*>(second);

    return (__first == __second) & (static_cast<const char*>(first)[8] == static_cast<const char*>(second)[8]);
}

simd_stl_declare_const_function simd_stl_always_inline bool memcmp10(
    const void* first,
    const void* second) noexcept
{
    const uint64 __firstQuad    = *reinterpret_cast<const uint64*>(first);
    const uint64 __secondQuad   = *reinterpret_cast<const uint64*>(second);

    const uint16 __firstWord    = *reinterpret_cast<const uint16*>(static_cast<const char*>(first) + 8);
    const uint16 __secondWord   = *reinterpret_cast<const uint16*>(static_cast<const char*>(second) + 8);

    return (__firstQuad == __secondQuad) & (__firstWord == __secondWord);
}

simd_stl_declare_const_function simd_stl_always_inline bool memcmp11(
    const void* first,
    const void* second) noexcept
{
    const uint64 __firstQuad      = *reinterpret_cast<const uint64*>(first);
    const uint64 __secondQuad     = *reinterpret_cast<const uint64*>(second);

    const uint32 __firstDWord     = *reinterpret_cast<const uint32*>(static_cast<const char*>(first) + 8);
    const uint32 __secondDWord    = *reinterpret_cast<const uint32*>(static_cast<const char*>(second) + 8);

    return (__firstQuad == __secondQuad) & ((__firstDWord & 0x00ffffff) == (__secondDWord & 0x00ffffff));
}

simd_stl_declare_const_function simd_stl_always_inline bool memcmp12(
    const void* first,
    const void* second) noexcept
{
    const uint64 __firstQuad    = *reinterpret_cast<const uint64*>(first);
    const uint64 __secondQuad   = *reinterpret_cast<const uint64*>(second);

    const uint32 __firstDWord   = *reinterpret_cast<const uint32*>(static_cast<const char*>(first) + 8);
    const uint32 __secondDWord  = *reinterpret_cast<const uint32*>(static_cast<const char*>(second) + 8);

    return (__firstQuad == __secondQuad) & (__firstDWord == __secondDWord);
}

template <class _Type_>
bool IsAllBitsZero(const _Type_& value) {
    static_assert(std::is_scalar_v<_Type_> && !std::is_member_pointer_v<_Type_>);

    if constexpr (std::is_same_v<_Type_, std::nullptr_t>)
        return true;

    constexpr auto zero = _Type_{};
    
    if      constexpr (sizeof(_Type_) == 8) return memcmp8(&value, &zero);
    else if constexpr (sizeof(_Type_) == 4) return memcmp4(&value, &zero);
    else if constexpr (sizeof(_Type_) == 2) return memcmp2(&value, &zero);
    else if constexpr (sizeof(_Type_) == 1) return memcmp1(&value, &zero);

    return memcmp(&value, &zero, sizeof(_Type_)) == 0;
}

using _Memcmp_fn = bool(*)(const void*, const void*);

template <sizetype _NeedleSizeInBytes_>
struct _Memcmp_selector {
    using type                          = void;
    static constexpr _Memcmp_fn value   = nullptr;
};

template <>
struct _Memcmp_selector<0> {
    using type                          = _Memcmp_fn;
    static constexpr _Memcmp_fn value   = &alwaysTrue;
};

template <>
struct _Memcmp_selector<1> {
    using type                          = _Memcmp_fn;
    static constexpr _Memcmp_fn value   = &memcmp1;
};

template <>
struct _Memcmp_selector<2> {
    using type                          = _Memcmp_fn;
    static constexpr _Memcmp_fn value   = &memcmp2;
};

template <>
struct _Memcmp_selector<3> {
    using type                          = _Memcmp_fn;
    static constexpr _Memcmp_fn value   = &memcmp3;
};

template <>
struct _Memcmp_selector<4> {
    using type                          = _Memcmp_fn;
    static constexpr _Memcmp_fn value   = &memcmp4;
};

template <>
struct _Memcmp_selector<5> {
    using type                          = _Memcmp_fn;
    static constexpr _Memcmp_fn value   = &memcmp5;
};

template <>
struct _Memcmp_selector<6> {
    using type                          = _Memcmp_fn;
    static constexpr _Memcmp_fn value   = &memcmp6;
};

template <>
struct _Memcmp_selector<7> {
    using type                          = _Memcmp_fn;
    static constexpr _Memcmp_fn value   = &memcmp7;
};

template <>
struct _Memcmp_selector<8> {
    using type                          = _Memcmp_fn;
    static constexpr _Memcmp_fn value   = &memcmp8;
};

template <>
struct _Memcmp_selector<9> {
    using type                          = _Memcmp_fn;
    static constexpr _Memcmp_fn value   = &memcmp9;
};

template <>
struct _Memcmp_selector<10> {
    using type                          = _Memcmp_fn;
    static constexpr _Memcmp_fn value   = &memcmp10;
};

template <>
struct _Memcmp_selector<11> {
    using type                          = _Memcmp_fn;
    static constexpr _Memcmp_fn value   = &memcmp11;
};

template <>
struct _Memcmp_selector<12> {
    using type                          = _Memcmp_fn;
    static constexpr _Memcmp_fn value   = &memcmp12;
};

template <sizetype _NeedleSizeInBytes_>
using _Choose_fixed_memcmp_function = _Memcmp_selector<_NeedleSizeInBytes_>;

__SIMD_STL_ALGORITHM_NAMESPACE_END

