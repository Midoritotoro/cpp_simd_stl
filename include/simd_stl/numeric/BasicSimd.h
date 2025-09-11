#pragma once 

#include <simd_stl/numeric/BasicSimdImplementation.h>
#include <simd_stl/numeric/BasicSimdElementReference.h>


__SIMD_STL_NUMERIC_NAMESPACE_BEGIN

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_ = int>
class basic_simd {
    static_assert(type_traits::__is_generation_supported_v<_SimdGeneration_>);
    static_assert(type_traits::__is_vector_type_supported_v<_Element_>);

    friend BasicSimdElementReference;
    using __impl = BasicSimdImplementation<_SimdGeneration_, _Element_>;
public:
    static constexpr auto _Generation = _SimdGeneration_;

    using value_type    = typename __impl::value_type;
    using vector_type   = typename __impl::vector_type;

    using size_type     = typename __impl::size_type;
    using mask_type     = typename __impl::mask_type;

    basic_simd() noexcept;

    basic_simd(const value_type value) noexcept;
    basic_simd(const vector_type& other) noexcept;

    ~basic_simd() noexcept;


    simd_stl_constexpr_cxx20 simd_stl_always_inline friend basic_simd operator+(
        const basic_simd& left, 
        const basic_simd& right) noexcept;

    simd_stl_constexpr_cxx20 simd_stl_always_inline friend basic_simd operator-(
        const basic_simd& left,
        const basic_simd& right) noexcept;

    simd_stl_constexpr_cxx20 simd_stl_always_inline friend basic_simd operator*(
        const basic_simd& left,
        const basic_simd& right) noexcept;


    simd_stl_constexpr_cxx20 simd_stl_always_inline friend basic_simd operator/(
        const basic_simd& left,
        const basic_simd& right) noexcept;


    simd_stl_constexpr_cxx20 simd_stl_always_inline friend basic_simd operator&(
        const basic_simd& left,
        const basic_simd& right) noexcept;


    simd_stl_constexpr_cxx20 simd_stl_always_inline friend basic_simd operator|(
        const basic_simd& left,
        const basic_simd& right) noexcept;


    simd_stl_constexpr_cxx20 simd_stl_always_inline friend basic_simd operator^(
        const basic_simd& left,
        const basic_simd& right) noexcept;   
    
    // Unary plus
    simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd operator+() const noexcept;
    // Unary minus
    simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd operator-() const noexcept;

    simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd& operator++(int) noexcept;
    simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd& operator++() noexcept;

    simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd& operator--(int) noexcept;
    simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd& operator--() noexcept;

    simd_stl_constexpr_cxx20 simd_stl_always_inline mask_type operator!() const noexcept;
    simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd operator~() const noexcept;


    simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd& operator+=(const basic_simd& other) const noexcept;
    simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd& operator-=(const basic_simd& other) noexcept;

    simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd& operator*=(const basic_simd& other) noexcept;
    simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd& operator/=(const basic_simd& other) noexcept;

    simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd& operator%=(const basic_simd& other) noexcept;

    simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd& operator=(const basic_simd& left) noexcept;

    simd_stl_constexpr_cxx20 simd_stl_always_inline _Element_ operator[](const size_type index) const noexcept;
    simd_stl_constexpr_cxx20 simd_stl_always_inline BasicSimdElementReference<basic_simd> operator[](const size_type index) noexcept;


    simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd& operator&=(const basic_simd& other) noexcept;
    simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd& operator|=(const basic_simd& other) noexcept;
    simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd& operator^=(const basic_simd& other) noexcept;
private:
    vector_type _vector;
};

template <
    arch::CpuFeature _SimdGeneration_,
    typename _Element_>
basic_simd<_SimdGeneration_, _Element_>::basic_simd() noexcept
{
    _vector = __impl::constructZero();
}

template <
    arch::CpuFeature _SimdGeneration_,
    typename _Element_>
basic_simd<_SimdGeneration_, _Element_>::basic_simd(const vector_type& other) noexcept:
    _vector(other)
{}

template <
    arch::CpuFeature _SimdGeneration_, 
    typename _Element_>
basic_simd<_SimdGeneration_, _Element_>::basic_simd(const value_type value) noexcept {
    _vector = __impl::broadcast(value);
}

template <
    arch::CpuFeature _SimdGeneration_, 
    typename _Element_>
basic_simd<_SimdGeneration_, _Element_>::~basic_simd() noexcept
{}


template <
    arch::CpuFeature _SimdGeneration_, 
    typename _Element_>
simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_>& 
basic_simd<_SimdGeneration_, _Element_>::operator+=(const basic_simd& other) const noexcept {
    _vector = _vector + other;
    return *this;
}

template <
    arch::CpuFeature _SimdGeneration_,
    typename _Element_>
simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_>& 
basic_simd<_SimdGeneration_, _Element_>::operator-=(const basic_simd& other) noexcept
{
}


template <
    arch::CpuFeature _SimdGeneration_,
    typename _Element_>
simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_>& 
basic_simd<_SimdGeneration_, _Element_>::operator*=(const basic_simd& other) noexcept {

}

template <
    arch::CpuFeature _SimdGeneration_, 
    typename _Element_>
simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_>& 
basic_simd<_SimdGeneration_, _Element_>::operator/(const basic_simd& other) const noexcept {

}

template <
    arch::CpuFeature _SimdGeneration_, 
    typename _Element_>
simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_>& 
basic_simd<_SimdGeneration_, _Element_>::operator/=(const basic_simd& other) noexcept {

}


template <
    arch::CpuFeature _SimdGeneration_, 
    typename _Element_>
simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_>& 
basic_simd<_SimdGeneration_, _Element_>::operator%=(const basic_simd& other) noexcept {

}

template <
    arch::CpuFeature _SimdGeneration_,
    typename _Element_>
simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_>&
basic_simd<_SimdGeneration_, _Element_>::operator=(const basic_simd& left) noexcept {
    _vector = left._vector;
}

template <
    arch::CpuFeature _SimdGeneration_, 
    typename _Element_>
simd_stl_constexpr_cxx20 simd_stl_always_inline _Element_ 
basic_simd<_SimdGeneration_, _Element_>::operator[](const size_type index) const noexcept {
    return __impl::extract(_vector, index);
}

template <
    arch::CpuFeature _SimdGeneration_, 
    typename _Element_>
simd_stl_constexpr_cxx20 simd_stl_always_inline BasicSimdElementReference<basic_simd<_SimdGeneration_, _Element_>>
basic_simd<_SimdGeneration_, _Element_>::operator[](const size_type index) noexcept {
    return BasicSimdElementReference<basic_simd<_SimdGeneration_, _Element_>>(this, index);
}

template <
    arch::CpuFeature _SimdGeneration_,
    typename _Element_>
simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_>& 
basic_simd<_SimdGeneration_, _Element_>::operator++(int) noexcept {

}

template <
    arch::CpuFeature _SimdGeneration_, 
    typename _Element_>
simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_>& 
basic_simd<_SimdGeneration_, _Element_>::operator++() noexcept {
 //   _vector += 1;
    return *this;
}

template <
    arch::CpuFeature _SimdGeneration_,
    typename _Element_>
simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_>& 
basic_simd<_SimdGeneration_, _Element_>::operator--(int) noexcept {

}

template <
    arch::CpuFeature _SimdGeneration_, 
    typename _Element_>
simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_>& 
basic_simd<_SimdGeneration_, _Element_>::operator--() noexcept {

}

template <
    arch::CpuFeature _SimdGeneration_,
    typename _Element_>
simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_>::mask_type
basic_simd<_SimdGeneration_, _Element_>::operator!() const noexcept {
    return __impl::convertToMask(__impl::bitwiseNot(_vector));
}

template <
    arch::CpuFeature _SimdGeneration_,
    typename _Element_>
simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_>
basic_simd<_SimdGeneration_, _Element_>::operator~() const noexcept {
    return __impl::bitwiseNot(_vector);
}

template <
    arch::CpuFeature _SimdGeneration_,
    typename _Element_>
simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_>&
basic_simd<_SimdGeneration_, _Element_>::operator&=(const basic_simd& other) noexcept {
    _vector = __impl::bitwiseAnd(_vector, other._vector);
    return *this;
}

template <
    arch::CpuFeature _SimdGeneration_,
    typename _Element_>
simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_>& 
basic_simd<_SimdGeneration_, _Element_>::operator|=(const basic_simd& other) noexcept {
    _vector = __impl::bitwiseOr(_vector, other._vector);
    return *this;
}

template <
    arch::CpuFeature _SimdGeneration_,
    typename _Element_>
simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_>&
basic_simd<_SimdGeneration_, _Element_>::operator^=(const basic_simd& other) noexcept {
    _vector = __impl::bitwiseXor(_vector, other._vector);
    return *this;
}

template <
    arch::CpuFeature _SimdGeneration_,
    typename _Element_>
simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_> operator+(
    const basic_simd<_SimdGeneration_, _Element_>& left,
    const basic_simd<_SimdGeneration_, _Element_>& right) noexcept
{
    return basic_simd<_SimdGeneration_, _Element_>::__impl::add(left._vector, right._vector);
}

template <
    arch::CpuFeature _SimdGeneration_, 
    typename _Element_>
simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_> operator-(
    const basic_simd<_SimdGeneration_, _Element_>& left,
    const basic_simd<_SimdGeneration_, _Element_>& right) noexcept
{
    return basic_simd<_SimdGeneration_, _Element_>::__impl::sub(left._vector, right._vector);
}

template <
    arch::CpuFeature _SimdGeneration_, 
    typename _Element_>
simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_> operator*(
    const basic_simd<_SimdGeneration_, _Element_>& left,
    const basic_simd<_SimdGeneration_, _Element_>& right) noexcept 
{
    return basic_simd<_SimdGeneration_, _Element_>::__impl::mul(left._vector, right._vector);
}

template <
    arch::CpuFeature _SimdGeneration_,
    typename _Element_>
simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_> operator&(
    const basic_simd<_SimdGeneration_, _Element_>& left,
    const basic_simd<_SimdGeneration_, _Element_>& right) noexcept
{
    return basic_simd<_SimdGeneration_, _Element_>::__impl::bitwiseAnd(left._vector, right._vector);
}

template <
    arch::CpuFeature _SimdGeneration_, 
    typename _Element_>
simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_> operator|(
    const basic_simd<_SimdGeneration_, _Element_>& left,
    const basic_simd<_SimdGeneration_, _Element_>& right) noexcept
{
    return basic_simd<_SimdGeneration_, _Element_>::__impl::bitwiseAnd(left._vector, right._vector);
}

template <
    arch::CpuFeature _SimdGeneration_,
    typename _Element_>
simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd<_SimdGeneration_, _Element_> operator^(
    const basic_simd<_SimdGeneration_, _Element_>& left,
    const basic_simd<_SimdGeneration_, _Element_>& right) noexcept
{
    return basic_simd<_SimdGeneration_, _Element_>::__impl::bitwiseXor(left._vector, right._vector);
}


__SIMD_STL_NUMERIC_NAMESPACE_END
