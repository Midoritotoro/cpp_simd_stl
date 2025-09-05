#pragma once 

#include <simd_stl/compatibility/SimdCompatibility.h>
#include <src/simd_stl/type_traits/TypeTraits.h>

#include <simd_stl/arch/CpuFeature.h>
#include <simd_stl/compatibility/Inline.h>

#include <xstring>

__SIMD_STL_NUMERIC_NAMESPACE_BEGIN

template <arch::CpuFeature _SimdGeneration_>
constexpr inline bool __is_generation_supported_v = arch::Contains<_SimdGeneration_, arch::__supportedFeatures>::value;

template <typename _VectorElementType_>
constexpr inline bool __is_vector_type_supported_v = type_traits::is_any_of_v<_VectorElementType_, int, float, double>;

template <
  arch::CpuFeature  _SimdGeneration_,
  typename      _VectorElementType_>
using __deduce_simd_vector_type = std::conditional_t<
    arch::__is_zmm_v<_SimdGeneration_>,
    std::conditional_t<
        std::is_same_v<_VectorElementType_, double>, __m512d,
        std::conditional_t<
            std::is_same_v<_VectorElementType_, float>, __m512,
            std::conditional_t<
                std::is_same_v<_VectorElementType_, int>, __m512i, void>>>,
    std::conditional_t<
        arch::__is_ymm_v<_SimdGeneration_>,
        std::conditional_t<
            std::is_same_v<_VectorElementType_, double>, __m256d,
            std::conditional_t<
                std::is_same_v<_VectorElementType_, float>, __m256,
                std::conditional_t<
                    std::is_same_v<_VectorElementType_, int>, __m256i, void>>>,
    std::conditional_t<
        arch::__is_xmm_v<_SimdGeneration_>,
        std::conditional_t<
            std::is_same_v<_VectorElementType_, double>, __m128d,
            std::conditional_t<
                std::is_same_v<_VectorElementType_, float>, __m128,
                std::conditional_t<
                    std::is_same_v<_VectorElementType_, int>, __m128i, void>>>,
        void>>;


#if !defined(__simd_stl_basic_simd)
#  define __simd_stl_basic_simd template <typename _Element_, arch::CpuFeature _SimdGeneration_>
#endif // __simd_stl_basic_simd

#if !defined(__simd_stl_basic_simd_t)
#  define __simd_stl_basic_simd_t basic_simd<_Element_, _SimdGeneration_>
#endif // __simd_stl_basic_simd_t


template <
	typename			_Element_,
	arch::CpuFeature	_SimdGeneration_>
class basic_simd {
	static_assert(__is_generation_supported_v<_SimdGeneration_>);
    static_assert(__is_vector_type_supported_v<_Element_>);
public:
	using value_t = _Element_;
    using simd_type_t = __deduce_simd_vector_type<_SimdGeneration_, _Element_>;

    using size_type = unsigned short;

    basic_simd(simd_type_t other) noexcept;
    ~basic_simd() noexcept;


    simd_stl_constexpr_cxx20 inline friend basic_simd& operator+(const basic_simd& left, const basic_simd& right) const noexcept;
    simd_stl_constexpr_cxx20 inline friend basic_simd& operator+=(const basic_simd& left, const basic_simd& right) const noexcept;

    simd_stl_constexpr_cxx20 inline friend basic_simd& operator-(const basic_simd& left, const basic_simd& right) const noexcept;
    simd_stl_constexpr_cxx20 inline friend basic_simd& operator-=(const basic_simd& left, const basic_simd& right) const noexcept;

    simd_stl_constexpr_cxx20 inline friend basic_simd& operator*(const basic_simd& left, const basic_simd& right) const noexcept;
    simd_stl_constexpr_cxx20 inline friend basic_simd& operator*=(const basic_simd& left, const basic_simd& right) const noexcept;

    simd_stl_constexpr_cxx20 inline friend basic_simd& operator/(const basic_simd& left, const basic_simd& right) const noexcept;
    simd_stl_constexpr_cxx20 inline friend basic_simd& operator/=(const basic_simd& left, const basic_simd& right) const noexcept;

    simd_stl_constexpr_cxx20 inline friend basic_simd& operator%(const basic_simd& left, const basic_simd& right) const noexcept;
    simd_stl_constexpr_cxx20 inline friend basic_simd& operator%=(const basic_simd& left, const basic_simd& right) const noexcept;

    simd_stl_constexpr_cxx20 inline friend basic_simd& operator=(const basic_simd& left) const noexcept;

    simd_stl_constexpr_cxx20 inline friend _Element_ operator[](const size_type index) const noexcept;
    simd_stl_constexpr_cxx20 inline friend _Element_& operator[](const size_type index) noexcept;

    simd_stl_constexpr_cxx20 inline friend basic_simd& operator++(int) noexcept;
    simd_stl_constexpr_cxx20 inline friend basic_simd& operator++() noexcept;

    simd_stl_constexpr_cxx20 inline friend basic_simd& operator--(int) noexcept;
    simd_stl_constexpr_cxx20 inline friend basic_simd& operator--() noexcept;

    simd_stl_constexpr_cxx20 inline friend basic_simd& operator!(int) noexcept;
    simd_stl_constexpr_cxx20 inline friend basic_simd& operator&() noexcept;

    simd_stl_constexpr_cxx20 inline friend basic_simd& operator|(int) noexcept;
    simd_stl_constexpr_cxx20 inline friend basic_simd& operator^() noexcept;

    simd_stl_constexpr_cxx20 inline friend basic_simd& operator&=() noexcept;

    simd_stl_constexpr_cxx20 inline friend basic_simd& operator|=(int) noexcept;
    simd_stl_constexpr_cxx20 inline friend basic_simd& operator^=() noexcept;
private:
    simd_type_t _vector;
};


__simd_stl_basic_simd
simd_stl_constexpr_cxx20 inline __simd_stl_basic_simd_t& __simd_stl_basic_simd_t::operator+(const __simd_stl_basic_simd_t& left, const __simd_stl_basic_simd_t& right) const noexcept {

}

simd_stl_constexpr_cxx20 inline friend basic_simd& operator+=(const basic_simd& left, const basic_simd& right) const noexcept;

simd_stl_constexpr_cxx20 inline friend basic_simd& operator-(const basic_simd& left, const basic_simd& right) const noexcept;
simd_stl_constexpr_cxx20 inline friend basic_simd& operator-=(const basic_simd& left, const basic_simd& right) const noexcept;

simd_stl_constexpr_cxx20 inline friend basic_simd& operator*(const basic_simd& left, const basic_simd& right) const noexcept;
simd_stl_constexpr_cxx20 inline friend basic_simd& operator*=(const basic_simd& left, const basic_simd& right) const noexcept;

simd_stl_constexpr_cxx20 inline friend basic_simd& operator/(const basic_simd& left, const basic_simd& right) const noexcept;
simd_stl_constexpr_cxx20 inline friend basic_simd& operator/=(const basic_simd& left, const basic_simd& right) const noexcept;

simd_stl_constexpr_cxx20 inline friend basic_simd& operator%(const basic_simd& left, const basic_simd& right) const noexcept;
simd_stl_constexpr_cxx20 inline friend basic_simd& operator%=(const basic_simd& left, const basic_simd& right) const noexcept;

simd_stl_constexpr_cxx20 inline friend basic_simd& operator=(const basic_simd& left) const noexcept;

simd_stl_constexpr_cxx20 inline friend _Element_ operator[](const size_type index) const noexcept;
simd_stl_constexpr_cxx20 inline friend _Element_& operator[](const size_type index) noexcept;

simd_stl_constexpr_cxx20 inline friend basic_simd& operator++(int) noexcept;
simd_stl_constexpr_cxx20 inline friend basic_simd& operator++() noexcept;

simd_stl_constexpr_cxx20 inline friend basic_simd& operator--(int) noexcept;
simd_stl_constexpr_cxx20 inline friend basic_simd& operator--() noexcept;

simd_stl_constexpr_cxx20 inline friend basic_simd& operator!(int) noexcept;
simd_stl_constexpr_cxx20 inline friend basic_simd& operator&() noexcept;

simd_stl_constexpr_cxx20 inline friend basic_simd& operator|(int) noexcept;
simd_stl_constexpr_cxx20 inline friend basic_simd& operator^() noexcept;

simd_stl_constexpr_cxx20 inline friend basic_simd& operator&=() noexcept;

simd_stl_constexpr_cxx20 inline friend basic_simd& operator|=(int) noexcept;
simd_stl_constexpr_cxx20 inline friend basic_simd& operator^=() noexcept;


__SIMD_STL_NUMERIC_NAMESPACE_END
