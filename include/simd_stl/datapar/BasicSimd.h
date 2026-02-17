#pragma once 

#include <src/simd_stl/datapar/Arithmetic.h>

#include <src/simd_stl/datapar/Compare.h>
#include <src/simd_stl/datapar/Shuffle.h>

#include <src/simd_stl/utility/Assert.h>

#include <simd_stl/datapar/BasicSimdMask.h>
#include <simd_stl/datapar/SimdCast.h>

#include <simd_stl/datapar/BasicSimdElementReference.h>

#include <simd_stl/datapar/SimdCompareResult.h>
#include <src/simd_stl/datapar/SimdCompareAdapters.h>

#include <src/simd_stl/datapar/Memory.h>


__SIMD_STL_DATAPAR_NAMESPACE_BEGIN


using aligned_policy    = __aligned_policy;
using unaligned_policy  = __unaligned_policy;

template <
    arch::ISA	_ISA_,
    class	    _Type_,
    uint32      _Width_ = __default_width<_ISA_>>
class simd {
    static_assert(type_traits::__is_generation_supported_v<_ISA_>);
    static_assert(type_traits::__is_vector_type_supported_v<std::decay_t<_Type_>>);

    template <typename _DesiredType_>
    using __mask_type = type_traits::__deduce_simd_mask_type<_ISA_, _Type_, _Width_>;

    template <typename _DesiredType_>
    using __reference_type = simd_element_reference<simd, _DesiredType_>;
public:
    static constexpr auto __isa = _ISA_;
    static constexpr auto __width = _Width_;
    
    using vector_type = type_traits::__deduce_simd_vector_type<_ISA_, _Type_, _Width_>;

    using value_type    = _Type_;
    using reference_type = __reference_type<value_type>;

    using mask_type     = simd_mask<_ISA_, _Type_, _Width_>;
    using size_type     = uint8;

    template <typename _DesiredType_ = value_type>
    static constexpr inline bool is_native_mask_load_supported_v = __is_native_mask_load_supported_v<
        __isa, _Width_, sizeof(_DesiredType_)>;

    template <typename _DesiredType_ = value_type>
    static constexpr inline bool is_native_mask_store_supported_v = __is_native_mask_store_supported_v<
        __isa, _Width_, sizeof(_DesiredType_)>;

    simd() noexcept;
    simd(const value_type __value) noexcept;
    ~simd() noexcept;

    template <
        typename _VectorType_,
        std::enable_if_t<__is_intrin_type_v<_VectorType_> || __is_valid_simd_v<_VectorType_>, int> = 0>
    simd(_VectorType_ __other) noexcept;

    static simd_stl_always_inline simd zero() noexcept;

    simd_stl_always_inline simd& fill(value_type __value) noexcept;
    simd_stl_always_inline value_type extract(size_type __index) const noexcept;
    simd_stl_always_inline simd_element_reference<simd> extract_wrapped(size_type __index) noexcept;

    simd_stl_always_inline void insert(
        size_type   __index,
        value_type  __value) noexcept;

    friend simd operator+ <>(const simd& __left, const value_type __right) noexcept;
    friend simd operator- <>(const simd& __left, const value_type __right) noexcept;
    friend simd operator* <>(const simd& __left, const value_type __right) noexcept;
    friend simd operator/ <>(const simd& __left, const value_type __right) noexcept;
    friend simd operator+ <>(const simd& __left, const simd& __right) noexcept;
    friend simd operator- <>(const simd& __left, const simd& __right) noexcept;
    friend simd operator* <>(const simd& __left, const simd& __right) noexcept;
    friend simd operator/ <>(const simd& __left, const simd& __right) noexcept;
    friend simd operator& <>(const simd& __left, const simd& __right) noexcept;
    friend simd operator| <>(const simd& __left, const simd& __right) noexcept;
    friend simd operator^ <>(const simd& __left, const simd& __right) noexcept;

    friend simd_compare_result<_ISA_, _Type_, _Width_> operator== <>(const simd& __left, const simd& __right) noexcept;
    friend simd_compare_result<_ISA_, _Type_, _Width_> operator!= <>(const simd& __left, const simd& __right) noexcept;
    friend simd_compare_result<_ISA_, _Type_, _Width_> operator< <>(const simd& __left, const simd& __right) noexcept;
    friend simd_compare_result<_ISA_, _Type_, _Width_> operator<= <>(const simd& __left, const simd& __right) noexcept;
    friend simd_compare_result<_ISA_, _Type_, _Width_> operator> <>(const simd& __left, const simd& __right) noexcept;
    friend simd_compare_result<_ISA_, _Type_, _Width_> operator>= <>(const simd& __left, const simd& __right) noexcept;

    simd_stl_always_inline simd& operator&=(const simd& __other) noexcept;
    simd_stl_always_inline simd& operator|=(const simd& __other) noexcept;
    simd_stl_always_inline simd& operator^=(const simd& __other) noexcept;
    simd_stl_always_inline simd& operator+=(const simd& __other) noexcept;
    simd_stl_always_inline simd& operator-=(const simd& __other) noexcept;
    simd_stl_always_inline simd& operator*=(const simd& __other) noexcept;
    simd_stl_always_inline simd& operator/=(const simd& __other) noexcept;
    simd_stl_always_inline simd& operator=(const simd& __other) noexcept;

    simd_stl_always_inline simd  operator+()     const noexcept;
    simd_stl_always_inline simd  operator-()     const noexcept;
    simd_stl_always_inline simd  operator++(int) noexcept;
    simd_stl_always_inline simd& operator++()    noexcept;
    simd_stl_always_inline simd  operator--(int) noexcept;
    simd_stl_always_inline simd& operator--()    noexcept;
    simd_stl_always_inline simd operator~() const noexcept;

    simd_stl_always_inline _Type_ operator[](const size_type __index) const noexcept;
    simd_stl_always_inline simd_element_reference<simd> operator[](const size_type __index) noexcept;

    template <typename _DesiredType_ = value_type>
    simd_stl_always_inline simd_mask<_ISA_, _DesiredType_, _Width_> to_mask() const noexcept;

    template <typename _DesiredType_ = value_type>
    simd_stl_always_inline auto to_index_mask() const noexcept;

    static simd_stl_always_inline constexpr int width() noexcept;
    static simd_stl_always_inline constexpr int size() noexcept;
    static simd_stl_always_inline constexpr int length() noexcept;

    static simd_stl_always_inline bool is_supported() noexcept;
    simd_stl_always_inline vector_type unwrap() const noexcept;
private:
    vector_type _vector;
};

__SIMD_STL_DATAPAR_NAMESPACE_END

#include <src/simd_stl/datapar/BasicSimd.inl>