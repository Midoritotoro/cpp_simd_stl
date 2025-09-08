#pragma once 

#include <simd_stl/numeric/BasicSimdImplementation.h>
#include <simd_stl/numeric/BasicSimdElementReference.h>

__SIMD_STL_NUMERIC_NAMESPACE_BEGIN

#if !defined(__basic_simd)
#  define __basic_simd template <arch::CpuFeature _SimdGeneration_, typename _Element_>
#endif // __basic_simd

#if !defined(__basic_simd_t)
#  define __basic_simd_t basic_simd<_SimdGeneration_, _Element_>
#endif // __basic_simd_t


struct _Basic_Simd_Plus_Operator_Tag        { _Basic_Simd_Plus_Operator_Tag()           noexcept = default; };
struct _Basic_Simd_Mul_Operator_Tag         { _Basic_Simd_Mul_Operator_Tag()            noexcept = default; };
struct _Basic_Simd_Div_Operator_Tag         { _Basic_Simd_Div_Operator_Tag()            noexcept = default; };
struct _Basic_Simd_Sub_Operator_Tag         { _Basic_Simd_Sub_Operator_Tag()            noexcept = default; };
struct _Basic_Simd_Bitwise_Or_Operator_Tag  { _Basic_Simd_Bitwise_Or_Operator_Tag()     noexcept = default; };
struct _Basic_Simd_Bitwise_Xor_Operator_Tag { _Basic_Simd_Bitwise_Xor_Operator_Tag()    noexcept = default; };
struct _Basic_Simd_Bitwise_And_Operator_Tag { _Basic_Simd_Bitwise_And_Operator_Tag()    noexcept = default; };
struct _Basic_Simd_Bitwise_Not_Operator_Tag { _Basic_Simd_Bitwise_Not_Operator_Tag()    noexcept = default; };


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

    basic_simd() noexcept;

    basic_simd(const value_type value) noexcept;
    basic_simd(const vector_type& other) noexcept;

    // Operators
    basic_simd(
        _Basic_Simd_Plus_Operator_Tag,
        const vector_type left, 
        const vector_type right) noexcept;

    ~basic_simd() noexcept;

    simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd& operator+=(const basic_simd& other) const noexcept;
    simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd& operator-=(const basic_simd& other) noexcept;

    simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd& operator*=(const basic_simd& other) noexcept;

    simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd& operator/(const basic_simd& other) const noexcept;
    simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd& operator/=(const basic_simd& other) noexcept;

    simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd& operator%=(const basic_simd& other) noexcept;

    simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd& operator=(const basic_simd& left) noexcept;

    simd_stl_constexpr_cxx20 simd_stl_always_inline _Element_ operator[](const size_type index) const noexcept;
    simd_stl_constexpr_cxx20 simd_stl_always_inline _Element_& operator[](const size_type index) noexcept;

    simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd& operator++(int) noexcept;
    simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd& operator++() noexcept;

    simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd& operator--(int) noexcept;
    simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd& operator--() noexcept;

    simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd& operator!() const noexcept;
    simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd& operator~() const noexcept;

    simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd& operator&=(const basic_simd& other) noexcept;
    simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd& operator|=(const basic_simd& other) noexcept;
    simd_stl_constexpr_cxx20 simd_stl_always_inline basic_simd& operator^=(const basic_simd& other) noexcept;
private:
    vector_type _vector;
};

__basic_simd
__basic_simd_t::basic_simd() noexcept
{
    _vector = __impl::constructZero();
}

__basic_simd
__basic_simd_t::basic_simd(const vector_type& other) noexcept:
    _vector(other)
{}

__basic_simd
__basic_simd_t::basic_simd(const value_type value) noexcept {
    _vector = __impl::broadcast(value);
}

__basic_simd
__basic_simd_t::~basic_simd() noexcept
{}

__basic_simd
__basic_simd_t::basic_simd(
    _Basic_Simd_Plus_Operator_Tag,
    const vector_type left,
    const vector_type right) noexcept
{
    _vector = __impl::mul(left, right);
}

__basic_simd
simd_stl_constexpr_cxx20 simd_stl_always_inline __basic_simd_t& __basic_simd_t::operator+=(const basic_simd& other) const noexcept {
    _vector = _vector + other;
    return *this;
}

__basic_simd
simd_stl_constexpr_cxx20 simd_stl_always_inline __basic_simd_t& __basic_simd_t::operator-=(const basic_simd& other) noexcept {
}


__basic_simd
simd_stl_constexpr_cxx20 simd_stl_always_inline __basic_simd_t& __basic_simd_t::operator*=(const basic_simd& other) noexcept {

}

__basic_simd
simd_stl_constexpr_cxx20 simd_stl_always_inline __basic_simd_t& __basic_simd_t::operator/(const basic_simd& other) const noexcept {

}

__basic_simd
simd_stl_constexpr_cxx20 simd_stl_always_inline __basic_simd_t& __basic_simd_t::operator/=(const basic_simd& other) noexcept {

}


__basic_simd
simd_stl_constexpr_cxx20 simd_stl_always_inline __basic_simd_t& __basic_simd_t::operator%=(const basic_simd& other) noexcept {

}

__basic_simd
simd_stl_constexpr_cxx20 simd_stl_always_inline __basic_simd_t& __basic_simd_t::operator=(const basic_simd& left) noexcept {
    _vector = left._vector;
}

__basic_simd
simd_stl_constexpr_cxx20 simd_stl_always_inline _Element_ __basic_simd_t::operator[](const size_type index) const noexcept {

}

__basic_simd
simd_stl_constexpr_cxx20 simd_stl_always_inline _Element_& __basic_simd_t::operator[](const size_type index) noexcept {

}

__basic_simd
simd_stl_constexpr_cxx20 simd_stl_always_inline __basic_simd_t& __basic_simd_t::operator++(int) noexcept {

}

__basic_simd
simd_stl_constexpr_cxx20 simd_stl_always_inline __basic_simd_t& __basic_simd_t::operator++() noexcept {
 //   _vector += 1;
    return *this;
}

__basic_simd
simd_stl_constexpr_cxx20 simd_stl_always_inline __basic_simd_t& __basic_simd_t::operator--(int) noexcept {

}

__basic_simd
simd_stl_constexpr_cxx20 simd_stl_always_inline __basic_simd_t& __basic_simd_t::operator--() noexcept {

}

__basic_simd
simd_stl_constexpr_cxx20 simd_stl_always_inline __basic_simd_t& __basic_simd_t::operator!() const noexcept {

}

__basic_simd
simd_stl_constexpr_cxx20 simd_stl_always_inline __basic_simd_t& __basic_simd_t::operator~() const noexcept {

}

__basic_simd
simd_stl_constexpr_cxx20 simd_stl_always_inline __basic_simd_t& __basic_simd_t::operator&=(const basic_simd& other) noexcept {

}

__basic_simd
simd_stl_constexpr_cxx20 simd_stl_always_inline __basic_simd_t& __basic_simd_t::operator|=(const basic_simd& other) noexcept {

}

__basic_simd
simd_stl_constexpr_cxx20 simd_stl_always_inline __basic_simd_t& __basic_simd_t::operator^=(const basic_simd& other) noexcept {

}

__basic_simd
simd_stl_constexpr_cxx20 inline __basic_simd_t& operator+(
    const __basic_simd_t& left, 
    const __basic_simd_t& right) noexcept
{
    basic_simd temp = left;
    temp._vector += right; 
    return temp;
}

__basic_simd
simd_stl_constexpr_cxx20 inline __basic_simd_t& operator-(
    const __basic_simd_t& left,
    const __basic_simd_t& right) noexcept 
{

}

__basic_simd
simd_stl_constexpr_cxx20 inline __basic_simd_t& operator*(
    const __basic_simd_t& left,
    const __basic_simd_t& right) noexcept
{

}

__basic_simd
simd_stl_constexpr_cxx20 inline __basic_simd_t& operator%(
    const __basic_simd_t& left,
    const __basic_simd_t& right) noexcept
{

}

__basic_simd
simd_stl_constexpr_cxx20 inline __basic_simd_t& operator&(
    const __basic_simd_t& left,
    const __basic_simd_t& right) noexcept
{

}

__basic_simd
simd_stl_constexpr_cxx20 inline __basic_simd_t& operator|(
    const __basic_simd_t& left,
    const __basic_simd_t& right) noexcept 
{

}

__basic_simd
simd_stl_constexpr_cxx20 inline __basic_simd_t& operator^(
    const __basic_simd_t& left,
    const __basic_simd_t& right) noexcept
{

}


__SIMD_STL_NUMERIC_NAMESPACE_END
