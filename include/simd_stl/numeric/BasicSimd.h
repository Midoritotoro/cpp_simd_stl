#pragma once 

#include <simd_stl/numeric/BasicSimdImplementation.h>

__SIMD_STL_NUMERIC_NAMESPACE_BEGIN

#if !defined(__basic_simd)
#  define __basic_simd template <arch::CpuFeature _SimdGeneration_, typename _Element_>
#endif // __basic_simd

#if !defined(__basic_simd_t)
#  define __basic_simd_t basic_simd<_SimdGeneration_, _Element_>
#endif // __basic_simd_t


template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_ = int>
class basic_simd {
    static_assert(__is_generation_supported_v<_SimdGeneration_>);
    static_assert(__is_vector_type_supported_v<_Element_>);
public:
    using value_type    = _Element_;
    using vector_type   = __deduce_simd_vector_type<_SimdGeneration_, _Element_>;

    using size_type = unsigned short;

    basic_simd() noexcept;

    basic_simd(const value_type value) noexcept;
    basic_simd(const vector_type& other) noexcept;

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
    _vector = __constructZero();
}

__basic_simd
__basic_simd_t::basic_simd(const vector_type& other) noexcept : _vector(other)
{
}

__basic_simd
__basic_simd_t::basic_simd(const value_type value) noexcept {
    _vector = __broadcast(value);
}

__basic_simd
__basic_simd_t::~basic_simd() noexcept
{
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
    _vector += 1;
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
static simd_stl_constexpr_cxx20 simd_stl_always_inline __basic_simd_t::vector_type __basic_simd_t::__constructZero() noexcept {
    if      constexpr (arch::__is_xmm_v<_SimdGeneration_>)
        return _mm_setzero_si128();
    else if constexpr (arch::__is_ymm_v<_SimdGeneration_>)
        return _mm256_setzero_si256();
    else if constexpr (arch::__is_zmm_v<_SimdGeneration_>)
        return _mm512_setzero_si512();
}

__basic_simd
simd_stl_constexpr_cxx20 simd_stl_always_inline __basic_simd_t::vector_type __basic_simd_t::__broadcast(const value_type value) noexcept {
    if constexpr (arch::__is_xmm_v<_SimdGeneration_>) {
        if      constexpr (sizeof(value_type) == 8)
            return _mm_set1_epi64x(value);
        else if constexpr (sizeof(value_type) == 4)
            return _mm_set1_epi32(value);
        else if constexpr (sizeof(value_type) == 2)
            return _mm_set1_epi16(value);
        else if constexpr (sizeof(value_type) == 1)
            return _mm_set1_epi8(value);
    }
    else if constexpr (arch::__is_ymm_v<_SimdGeneration_>) {
        if      constexpr (sizeof(value_type) == 8)
            return _mm256_set1_epi64x(value);
        else if constexpr (sizeof(value_type) == 4)
            return _mm256_set1_epi32(value);
        else if constexpr (sizeof(value_type) == 2)
            return _mm256_set1_epi16(value);
        else if constexpr (sizeof(value_type) == 1)
            return _mm256_set1_epi8(value);
    }
    else if constexpr (arch::__is_zmm_v<_SimdGeneration_>) {
        if      constexpr (sizeof(value_type) == 8)
            return _mm512_set1_epi64(value);
        else if constexpr (sizeof(value_type) == 4)
            return _mm512_set1_epi32(value);
        else if constexpr (sizeof(value_type) == 2)
            return _mm512_set1_epi16(value);
        else if constexpr (sizeof(value_type) == 1)
            return _mm512_set1_epi8(value);
    }
}

__basic_simd
simd_stl_constexpr_cxx20 simd_stl_always_inline __basic_simd_t::vector_type __basic_simd_t::__add(
    const vector_type& left,
    const vector_type& right) noexcept
{
    if constexpr (arch::__is_xmm_v<_SimdGeneration_>) {
        if      constexpr (sizeof(value_type) == 8)
            _vector = _mm_add_epi64(left, right);
        else if constexpr (sizeof(value_type) == 4)
            _vector = _mm_add_epi32(left, right);
        else if constexpr (sizeof(value_type) == 2)
            _vector = _mm_add_epi16(left, right);
        else if constexpr (sizeof(value_type) == 1)
            _vector = _mm_add_epi8(left, right);
    }
    else if constexpr (arch::__is_ymm_v<_SimdGeneration_>) {
        if      constexpr (sizeof(value_type) == 8)
            _vector = _mm256_add_epi64(left, right);
        else if constexpr (sizeof(value_type) == 4)
            _vector = _mm256_add_epi32(left, right);
        else if constexpr (sizeof(value_type) == 2)
            _vector = _mm256_add_epi16(left, right);
        else if constexpr (sizeof(value_type) == 1)
            _vector = _mm256_add_epi8(left, right);
    }
    else if constexpr (arch::__is_zmm_v<_SimdGeneration_>) {
        if      constexpr (sizeof(value_type) == 8)
            _vector = _mm512_add_epi64(left, right);
        else if constexpr (sizeof(value_type) == 4)
            _vector = _mm512_add_epi32(left, right);
        else if constexpr (sizeof(value_type) == 2)
            _vector = _mm512_add_epi16(left, right);
        else if constexpr (sizeof(value_type) == 1)
            _vector = _mm512_add_epi8(left, right);
    }
}

__basic_simd
simd_stl_constexpr_cxx20 simd_stl_always_inline __basic_simd_t::vector_type __basic_simd_t::__sub(
    const vector_type& left,
    const vector_type& right) noexcept
{
    if constexpr (arch::__is_xmm_v<_SimdGeneration_>) {
        if      constexpr (sizeof(value_type) == 8)
            _vector = _mm_sub_epi64(left, right);
        else if constexpr (sizeof(value_type) == 4)
            _vector = _mm_sub_epi32(left, right);
        else if constexpr (sizeof(value_type) == 2)
            _vector = _mm_sub_epi16(left, right);
        else if constexpr (sizeof(value_type) == 1)
            _vector = _mm_sub_epi8(left, right);
    }
    else if constexpr (arch::__is_ymm_v<_SimdGeneration_>) {
        if      constexpr (sizeof(value_type) == 8)
            _vector = _mm256_sub_epi64(left, right);
        else if constexpr (sizeof(value_type) == 4)
            _vector = _mm256_sub_epi32(left, right);
        else if constexpr (sizeof(value_type) == 2)
            _vector = _mm256_sub_epi16(left, right);
        else if constexpr (sizeof(value_type) == 1)
            _vector = _mm256_sub_epi8(left, right);
    }
    else if constexpr (arch::__is_zmm_v<_SimdGeneration_>) {
        if      constexpr (sizeof(value_type) == 8)
            _vector = _mm512_sub_epi64(left, right);
        else if constexpr (sizeof(value_type) == 4)
            _vector = _mm512_sub_epi32(left, right);
        else if constexpr (sizeof(value_type) == 2)
            _vector = _mm512_sub_epi16(left, right);
        else if constexpr (sizeof(value_type) == 1)
            _vector = _mm512_sub_epi8(left, right);
    }
}

__basic_simd
simd_stl_constexpr_cxx20 simd_stl_always_inline __basic_simd_t::vector_type __basic_simd_t::__mul(
    const vector_type& left,
    const vector_type& right) noexcept
{
    if constexpr (arch::__is_xmm_v<_SimdGeneration_>) {
        if      constexpr (sizeof(value_type) == 8)
            _vector = _mm_mul_epi64(left, right);
        else if constexpr (sizeof(value_type) == 4)
            _vector = _mm_mul_epi32(left, right);
        else if constexpr (sizeof(value_type) == 2)
            _vector = _mm_mul_epi16(left, right);
        else if constexpr (sizeof(value_type) == 1)
            _vector = _mm_mul_epi8(left, right);
    }
    else if constexpr (arch::__is_ymm_v<_SimdGeneration_>) {
        if      constexpr (sizeof(value_type) == 8)
            _vector = _mm256_mul_epi64(left, right);
        else if constexpr (sizeof(value_type) == 4)
            _vector = _mm256_mul_epi32(left, right);
        else if constexpr (sizeof(value_type) == 2)
            _vector = _mm256_mul_epi16(left, right);
        else if constexpr (sizeof(value_type) == 1)
            _vector = _mm256_mul_epi8(left, right);
    }
    else if constexpr (arch::__is_zmm_v<_SimdGeneration_>) {
        if      constexpr (sizeof(value_type) == 8)
            _vector = _mm512_mul_epi64(left, right);
        else if constexpr (sizeof(value_type) == 4)
            _vector = _mm512_mul_epi32(left, right);
        else if constexpr (sizeof(value_type) == 2)
            _vector = _mm512_mul_epi16(left, right);
        else if constexpr (sizeof(value_type) == 1)
            _vector = _mm512_mul_epi8(left, right);
    }
}

__basic_simd
simd_stl_constexpr_cxx20 simd_stl_always_inline __basic_simd_t::vector_type __basic_simd_t::__div(
    const vector_type& left,
    const vector_type& right) noexcept
{

}

__basic_simd
simd_stl_constexpr_cxx20 simd_stl_always_inline __basic_simd_t::vector_type __basic_simd_t::__mod(
    const vector_type& left,
    const vector_type& right) noexcept
{

}

__basic_simd
simd_stl_constexpr_cxx20 simd_stl_always_inline __basic_simd_t::vector_type __basic_simd_t::__bitwiseNot(const vector_type& vector) noexcept {

}

__basic_simd
simd_stl_constexpr_cxx20 simd_stl_always_inline __basic_simd_t::vector_type __basic_simd_t::__bitwiseXor(
    const vector_type& left,
    const vector_type& right) noexcept
{

}

__basic_simd
simd_stl_constexpr_cxx20 simd_stl_always_inline __basic_simd_t::vector_type __basic_simd_t::__bitwiseAnd(
    const vector_type& left,
    const vector_type& right) noexcept
{

}

__basic_simd
static simd_stl_constexpr_cxx20 simd_stl_always_inline __basic_simd_t::vector_type __basic_simd_t::__bitwiseOr(
    const vector_type& left,
    const vector_type& right) noexcept
{

}


// ==============================================================================================

__basic_simd
simd_stl_constexpr_cxx20 inline __basic_simd_t& operator+(
    const __basic_simd_t& left, 
    const __basic_simd_t& right) noexcept
{

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
