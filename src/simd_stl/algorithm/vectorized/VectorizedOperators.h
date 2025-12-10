#pragma once 

#include <simd_stl/numeric/BasicSimd.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <
    class _UnaryPredicate_,
    class _BasicSimd_,
    class = void>
constexpr inline bool _Is_unary_predicate_vectorizable_v = false;

template <
    class _UnaryPredicate_,
    class _BasicSimd_>
constexpr inline bool _Is_unary_predicate_vectorizable_v<_UnaryPredicate_, _BasicSimd_, std::void_t<
    decltype(std::declval<_UnaryPredicate_>()(std::declval<_BasicSimd_>()))>> = true;

template <
    class _BinaryPredicate_,
    class _BasicSimd_,
    class = void>
constexpr inline bool _Is_binary_predicate_vectorizable_v = false;

template <
    class _BinaryPredicate_,
    class _BasicSimd_>
constexpr inline bool _Is_binary_predicate_vectorizable_v<_BinaryPredicate_, _BasicSimd_, std::void_t<
    decltype(std::declval<_BinaryPredicate_>()(std::declval<_BasicSimd_>(), std::declval<_BasicSimd_>()))>> = true;


template <
    class _Predicate_,
    class _BasicSimd_> 
constexpr inline auto _Is_predicate_vectorizable_v = _Is_unary_predicate_vectorizable_v<_Predicate_, _BasicSimd_> ||
    _Is_binary_predicate_vectorizable_v<_Predicate_, _BasicSimd_>;

__SIMD_STL_ALGORITHM_NAMESPACE_END