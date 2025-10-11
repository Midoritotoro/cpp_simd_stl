#pragma once 

#include <type_traits> 

#include <simd_stl/compatibility/Compatibility.h>
#include <simd_stl/SimdStlNamespace.h>

#include <vector> 

template <class _Type_>
simd_stl_nodiscard _Type_ _FakeCopyInit(_Type_) noexcept;

__SIMD_STL_TYPE_TRAITS_NAMESPACE_BEGIN

template <class _Type_ = void>
struct plus {
    simd_stl_nodiscard constexpr _Type_ operator()(
        const _Type_& left,
        const _Type_& right) const noexcept(noexcept(left + right))
    {
        return left + right;
    }
};

template <class _Type_ = void>
struct minus {
    simd_stl_nodiscard constexpr _Type_ operator()(
        const _Type_& left,
        const _Type_& right) const noexcept(noexcept(left - right))
    {
        return left - right;
    }
};

template <class _Type_ = void>
struct multiplies {
    simd_stl_nodiscard constexpr _Type_ operator()(
        const _Type_& left,
        const _Type_& right) const noexcept(noexcept(left * right))
    {
        return left * right;
    }
};

template <class _Type_ = void>
struct divides {
    simd_stl_nodiscard constexpr _Type_ operator()(
        const _Type_& left,
        const _Type_& right) const 
    {
        return left / right;
    }
};

template <class _Type_ = void>
struct modulus {
    simd_stl_nodiscard constexpr _Type_ operator()(
        const _Type_& left,
        const _Type_& right) const noexcept(noexcept(left % right))
    {
        return left % right;
    }
};

template <class _Type_ = void>
struct negate {
    simd_stl_nodiscard constexpr _Type_ operator()(const _Type_& left) const noexcept(noexcept(-left)) {
        return -left;
    }
};


template <class _Type_ = void>
struct logical_and {
    simd_stl_nodiscard constexpr bool operator()(
        const _Type_& left,
        const _Type_& right) const noexcept(noexcept(left && right))
    {
        return left && right;
    }
};

template <class _Type_ = void>
struct logical_or {
    simd_stl_nodiscard constexpr bool operator()(
        const _Type_& left, 
        const _Type_& right) const noexcept(noexcept(left || right))
    {
        return left || right;
    }
};

template <class _Type_ = void>
struct logical_not {
    simd_stl_nodiscard constexpr bool operator()(const _Type_& left) const noexcept(noexcept(!left)) {
        return !left;
    }
};

template <class _Type_ = void>
struct bit_and {
    simd_stl_nodiscard constexpr _Type_ operator()(
        const _Type_& left, 
        const _Type_& right) const noexcept(noexcept(left & right))
    {
        return left & right;
    }
};

template <class _Type_ = void>
struct bit_or {
    simd_stl_nodiscard constexpr _Type_ operator()(
        const _Type_& left, 
        const _Type_& right) const noexcept(noexcept(left | right))
    {
        return left | right;
    }
};

template <class _Type_ = void>
struct bit_xor {
    simd_stl_nodiscard constexpr _Type_ operator()(
        const _Type_& left,
        const _Type_& right) const noexcept(noexcept(left ^ right))
    {
        return left ^ right;
    }
};

template <class _Type_ = void>
struct bit_not {
    simd_stl_nodiscard constexpr _Type_ operator()(const _Type_& left) const noexcept(noexcept(~left)) {
        return ~left;
    }
};

template <class _Type_ = void>
struct equal_to {
    simd_stl_nodiscard constexpr bool operator()(
        const _Type_& left,
        const _Type_& right) const noexcept(noexcept(_FakeCopyInit<bool>(left == right)))
    {
        return left == right;
    }
};

template <class _Type_ = void>
struct not_equal_to {
    simd_stl_nodiscard constexpr bool operator()(
        const _Type_& left,
        const _Type_& right) const noexcept(noexcept(_FakeCopyInit<bool>(left != right)))
    {
        return left != right;
    }
};

template <class _Type_ = void>
struct greater {
    simd_stl_nodiscard constexpr bool operator()(
        const _Type_& left,
        const _Type_& right) const noexcept(noexcept(_FakeCopyInit<bool>(left > right)))
    {
        return left > right;
    }
};

template <class _Type_ = void>
struct greater_equal {
    simd_stl_nodiscard constexpr bool operator()(
        const _Type_& left,
        const _Type_& right) const noexcept(noexcept(_FakeCopyInit<bool>(left >= right)))
    {
        return left >= right;
    }
};

template <class _Type_ = void>
struct less_equal {
    simd_stl_nodiscard constexpr bool operator()(
        const _Type_& left,
        const _Type_& right) const noexcept(noexcept(_FakeCopyInit<bool>(left <= right)))
    {
        return left <= right;
    }
};

template <>
struct plus<void> {
    template <
        class _FirstType_, 
        class _SecondType_>
    simd_stl_nodiscard constexpr auto operator()(
        _FirstType_&&   left, 
        _SecondType_&&  right) const noexcept(noexcept(static_cast<_FirstType_&&>(left) + static_cast<_SecondType_&&>(right)))
            -> decltype(static_cast<_FirstType_&&>(left) + static_cast<_SecondType_&&>(right)) 
    {
        return static_cast<_FirstType_&&>(left) + static_cast<_SecondType_&&>(right);
    }

    using is_transparent = int;
};

template <>
struct minus<void> {
    template <
        class _FirstType_,
        class _SecondType_>
    simd_stl_nodiscard constexpr auto operator()(
        _FirstType_&&   left, 
        _SecondType_&&  right) const noexcept(noexcept(static_cast<_FirstType_&&>(left) - static_cast<_SecondType_&&>(right)))
            -> decltype(static_cast<_FirstType_&&>(left) - static_cast<_SecondType_&&>(right))
    {
        return static_cast<_FirstType_&&>(left) - static_cast<_SecondType_&&>(right);
    }

    using is_transparent = int;
};

template <>
struct multiplies<void> {
    template <
        class _FirstType_,
        class _SecondType_>
    simd_stl_nodiscard constexpr auto operator()(
        _FirstType_&&   left, 
        _SecondType_&&  right) const noexcept(noexcept(static_cast<_FirstType_&&>(left) * static_cast<_SecondType_&&>(right)))
            -> decltype(static_cast<_FirstType_&&>(left) * static_cast<_SecondType_&&>(right)) 
    {
        return static_cast<_FirstType_&&>(left) * static_cast<_SecondType_&&>(right);
    }

    using is_transparent = int;
};

template <>
struct equal_to<void> {
    template <
        class _FirstType_, 
        class _SecondType_>
    simd_stl_nodiscard constexpr auto operator()(
        _FirstType_&&   left, 
        _SecondType_&&  right) const noexcept(noexcept(static_cast<_FirstType_&&>(left) == static_cast<_SecondType_&&>(right)))
            -> decltype(static_cast<_FirstType_&&>(left) == static_cast<_SecondType_&&>(right)) 
    {
        return static_cast<_FirstType_&&>(left) == static_cast<_SecondType_&&>(right);
    }

    using is_transparent = int;
};

template <>
struct not_equal_to<void> {
    template <
        class _FirstType_, 
        class _SecondType_>
    simd_stl_nodiscard constexpr auto operator()(
        _FirstType_&&   left,
        _SecondType_&&  right) const noexcept(noexcept(static_cast<_FirstType_&&>(left) != static_cast<_SecondType_&&>(right)))
            -> decltype(static_cast<_FirstType_&&>(left) != static_cast<_SecondType_&&>(right))
    {
        return static_cast<_FirstType_&&>(left) != static_cast<_SecondType_&&>(right);
    }

    using is_transparent = int;
};

template <>
struct greater<void> {
    template <
        class _FirstType_,
        class _SecondType_>
    simd_stl_nodiscard constexpr auto operator()(
        _FirstType_&&   left, 
        _SecondType_&&  right) const noexcept(noexcept(static_cast<_FirstType_&&>(left) > static_cast<_SecondType_&&>(right)))
            -> decltype(static_cast<_FirstType_&&>(left) > static_cast<_SecondType_&&>(right)) 
    {
        return static_cast<_FirstType_&&>(left) > static_cast<_SecondType_&&>(right);
    }

    using is_transparent = int;
};

template <>
struct greater_equal<void> {
    template <
        class _FirstType_, 
        class _SecondType_>
    simd_stl_nodiscard constexpr auto operator()(
        _FirstType_&& left, 
        _SecondType_&& right) const noexcept(noexcept(static_cast<_FirstType_&&>(left) >= static_cast<_SecondType_&&>(right)))
            -> decltype(static_cast<_FirstType_&&>(left) >= static_cast<_SecondType_&&>(right))
    {
        return static_cast<_FirstType_&&>(left) >= static_cast<_SecondType_&&>(right);
    }

    using is_transparent = int;
};

template <>
struct less_equal<void> {
    template <
        class _FirstType_,
        class _SecondType_>
    simd_stl_nodiscard constexpr auto operator()(
        _FirstType_&& left, 
        _SecondType_&& right) const noexcept(noexcept(static_cast<_FirstType_&&>(left) <= static_cast<_SecondType_&&>(right)))
            -> decltype(static_cast<_FirstType_&&>(left) <= static_cast<_SecondType_&&>(right)) 
    {
        return static_cast<_FirstType_&&>(left) <= static_cast<_SecondType_&&>(right);
    }

    using is_transparent = int;
};

template <>
struct divides<void> {
    template <
        class _FirstType_, 
        class _SecondType_>
    simd_stl_nodiscard constexpr auto operator()(
        _FirstType_&& left, 
        _SecondType_&& right) const noexcept(noexcept(std::forward<_FirstType_>(left) / std::forward<_SecondType_>(right)))
            -> decltype(std::forward<_FirstType_>(left) / std::forward<_SecondType_>(right)) 
    {
        return std::forward<_FirstType_>(left) / std::forward<_SecondType_>(right);
    }

    using is_transparent = int;
};

template <>
struct modulus<void> {
    template <
        class _FirstType_,
        class _SecondType_>
    simd_stl_nodiscard constexpr auto operator()(
        _FirstType_&& left,
        _SecondType_&& right) const noexcept(noexcept(std::forward<_FirstType_>(left) % std::forward<_SecondType_>(right)))
            -> decltype(std::forward<_FirstType_>(left) % std::forward<_SecondType_>(right))
    {
        return std::forward<_FirstType_>(left) % std::forward<_SecondType_>(right);
    }

    using is_transparent = int;
};

template <>
struct negate<void> {
    template <class _Type_>
    simd_stl_nodiscard constexpr auto operator()(_Type_&& left) const noexcept(noexcept(-std::forward<_Type_>(left))) 
        -> decltype(-std::forward<_Type_>(left))
    {
        return -std::forward<_Type_>(left);
    }

    using is_transparent = int;
};

template <class _Type_ = void>
struct less {
    simd_stl_nodiscard constexpr bool operator()(
        const _Type_& left, 
        const _Type_& right) const noexcept(noexcept(_FakeCopyInit<bool>(left < right)))
    {
        return left < right;
    }
};

template <>
struct less<void> {
    template <
        class _FirstType_, 
        class _SecondType_>
    simd_stl_nodiscard constexpr auto operator()(
        _FirstType_&&   left, 
        _SecondType_&&  right) const noexcept(noexcept(static_cast<_FirstType_&&>(left) < static_cast<_SecondType_&&>(right)))
            -> decltype(static_cast<_FirstType_&&>(left) < static_cast<_SecondType_&&>(right))
    {
        return static_cast<_FirstType_&&>(left) < static_cast<_SecondType_&&>(right);
    }

    using is_transparent = int;
};

template <>
struct logical_and<void> {
    template <
        class _FirstType_, 
        class _SecondType_>
    simd_stl_nodiscard constexpr auto operator()(
        _FirstType_&& left,
        _SecondType_&& right) const noexcept(noexcept(std::forward<_FirstType_>(left) && std::forward<_SecondType_>(right)))
            -> decltype(std::forward<_FirstType_>(left) && std::forward<_SecondType_>(right)) {
        return std::forward<_FirstType_>(left) && std::forward<_SecondType_>(right);
    }

    using is_transparent = int;
};

template <>
struct logical_or<void> {
    template <
        class _FirstType_, 
        class _SecondType_>
    simd_stl_nodiscard constexpr auto operator()(
        _FirstType_&& left, 
        _SecondType_&& right) const noexcept(noexcept(std::forward<_FirstType_>(left) || std::forward<_SecondType_>(right)))
            -> decltype(std::forward<_FirstType_>(left) || std::forward<_SecondType_>(right))
    {
        return std::forward<_FirstType_>(left) || std::forward<_SecondType_>(right);
    }

    using is_transparent = int;
};

template <>
struct logical_not<void> {
    template <class _Type_>
    simd_stl_nodiscard constexpr auto operator()(_Type_&& left) const noexcept(noexcept(!std::forward<_Type_>(left))) 
        -> decltype(!std::forward<_Type_>(left))
    {
        return !std::forward<_Type_>(left);
    }

    using is_transparent = int;
};

template <>
struct bit_and<void> {
    template <
        class _FirstType_,
        class _SecondType_>
    simd_stl_nodiscard constexpr auto operator()(
        _FirstType_&&   left, 
        _SecondType_&&  right) const noexcept(noexcept(std::forward<_FirstType_>(left) & std::forward<_SecondType_>(right)))
            -> decltype(std::forward<_FirstType_>(left)& std::forward<_SecondType_>(right)) 
    {
        return std::forward<_FirstType_>(left) & std::forward<_SecondType_>(right);
    }

    using is_transparent = int;
};

template <>
struct bit_or<void> {
    template <
        class _FirstType_, 
        class _SecondType_>
    simd_stl_nodiscard constexpr auto operator()(
        _FirstType_&& left,
        _SecondType_&& right) const noexcept(noexcept(std::forward<_FirstType_>(left) | std::forward<_SecondType_>(right)))
            -> decltype(std::forward<_FirstType_>(left) | std::forward<_SecondType_>(right)) 
    {
        return std::forward<_FirstType_>(left) | std::forward<_SecondType_>(right);
    }

    using is_transparent = int;
};

template <>
struct bit_xor<void> {
    template <
        class _FirstType_,
        class _SecondType_>
    simd_stl_nodiscard constexpr auto operator()(
        _FirstType_&&   left,
        _SecondType_&&  right) const noexcept(noexcept(std::forward<_FirstType_>(left) ^ std::forward<_SecondType_>(right)))
            -> decltype(std::forward<_FirstType_>(left) ^ std::forward<_SecondType_>(right))
    {
        return std::forward<_FirstType_>(left) ^ std::forward<_SecondType_>(right);
    }

    using is_transparent = int;
};

template <>
struct bit_not<void> {
    template <class _Type_>
    simd_stl_nodiscard constexpr auto operator()(_Type_&& left) const noexcept(noexcept(~std::forward<_Type_>(left))) 
        -> decltype(~std::forward<_Type_>(left))
    {
        return ~std::forward<_Type_>(left);
    }

    using is_transparent = int;
};


__SIMD_STL_TYPE_TRAITS_NAMESPACE_END
