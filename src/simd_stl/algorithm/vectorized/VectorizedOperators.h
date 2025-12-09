#pragma once 

#include <simd_stl/numeric/BasicSimd.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <class _Predicate_> 
struct _VectorizedUnaryPredicateInvoker {
    static constexpr bool _Is_predicate_vectorizable = false;

    template <class _BasicSimd_> 
    static simd_stl_always_inline _BasicSimd_ _Invoke(const _BasicSimd_& _Simd) noexcept;
};

template <>
struct _VectorizedUnaryPredicateInvoker<type_traits::negate<>> {
    static constexpr bool _Is_predicate_vectorizable = true;

    template <class _BasicSimd_>
    static simd_stl_always_inline _BasicSimd_ _Invoke(const _BasicSimd_& _Simd) noexcept {
        static_assert(numeric::_Is_valid_basic_simd_v<_BasicSimd_>);
        return (-_Simd);
    }
};

template <>
struct _VectorizedUnaryPredicateInvoker<type_traits::bit_not<>> {
    static constexpr bool _Is_predicate_vectorizable = true;

    template <class _BasicSimd_>
    static simd_stl_always_inline _BasicSimd_ _Invoke(const _BasicSimd_& _Simd) noexcept {
        static_assert(numeric::_Is_valid_basic_simd_v<_BasicSimd_>);
        return (~_Simd);
    }
};

template <class _Predicate_>
struct _VectorizedBinaryPredicateInvoker {
    static constexpr bool _Is_predicate_vectorizable = false;

    template <class _BasicSimd_>
    static simd_stl_always_inline _BasicSimd_ _Invoke(
        const _BasicSimd_& _Left,
        const _BasicSimd_& _Right) noexcept;
};

template <>
struct _VectorizedBinaryPredicateInvoker<type_traits::bit_and<>> {
    static constexpr bool _Is_predicate_vectorizable = true;

    template <class _BasicSimd_>
    static simd_stl_always_inline _BasicSimd_ _Invoke(
        const _BasicSimd_& _Left,
        const _BasicSimd_& _Right) noexcept
    {
        return (_Left & _Right);
    }
};

template <>
struct _VectorizedBinaryPredicateInvoker<type_traits::bit_or<>> {
    static constexpr bool _Is_predicate_vectorizable = true;

    template <class _BasicSimd_>
    static simd_stl_always_inline _BasicSimd_ _Invoke(
        const _BasicSimd_& _Left,
        const _BasicSimd_& _Right) noexcept
    {
        return (_Left | _Right);
    }
};

template <>
struct _VectorizedBinaryPredicateInvoker<type_traits::bit_xor<>> {
    static constexpr bool _Is_predicate_vectorizable = true;

    template <class _BasicSimd_>
    static simd_stl_always_inline _BasicSimd_ _Invoke(
        const _BasicSimd_& _Left,
        const _BasicSimd_& _Right) noexcept
    {
        return (_Left ^ _Right);
    }
};

template <>
struct _VectorizedBinaryPredicateInvoker<type_traits::plus<>> {
    static constexpr bool _Is_predicate_vectorizable = true;

    template <class _BasicSimd_>
    static simd_stl_always_inline _BasicSimd_ _Invoke(
        const _BasicSimd_& _Left,
        const _BasicSimd_& _Right) noexcept
    {
        return (_Left + _Right);
    }
};

template <>
struct _VectorizedBinaryPredicateInvoker<type_traits::minus<>> {
    static constexpr bool _Is_predicate_vectorizable = true;

    template <class _BasicSimd_>
    static simd_stl_always_inline _BasicSimd_ _Invoke(
        const _BasicSimd_& _Left,
        const _BasicSimd_& _Right) noexcept
    {
        return (_Left - _Right);
    }
};

template <class _Predicate_> 
constexpr inline auto _Is_predicate_vectorizable_v = _VectorizedBinaryPredicateInvoker<_Predicate_>::_Is_predicate_vectorizable ||
    _VectorizedUnaryPredicateInvoker<_Predicate_>::_Is_predicate_vectorizable;

template <
    class _Predicate_,
    class _BasicSimd_>
simd_stl_always_inline _BasicSimd_ _InvokeVectorizedTransformPredicate(const _BasicSimd_& _Simd) noexcept {
    static_assert(_Is_predicate_vectorizable_v<_Predicate_>);
    return _VectorizedUnaryPredicateInvoker<_Predicate_>::_Invoke(_Simd);
}

template <
    class _Predicate_,
    class _BasicSimd_>
simd_stl_always_inline _BasicSimd_ _InvokeVectorizedTransformPredicate(
    const _BasicSimd_& _Left,
    const _BasicSimd_& _Right) noexcept 
{
    static_assert(_Is_predicate_vectorizable_v<_Predicate_>);
    return _VectorizedBinaryPredicateInvoker<_Predicate_>::_Invoke(_Left, _Right);
}

__SIMD_STL_ALGORITHM_NAMESPACE_END