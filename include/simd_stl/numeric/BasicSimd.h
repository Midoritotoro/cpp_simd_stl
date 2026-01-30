#pragma once 

#include <src/simd_stl/numeric/SimdArithmetic.h>

#include <src/simd_stl/numeric/SimdCompare.h>
#include <src/simd_stl/numeric/SimdConvert.h>

#include <src/simd_stl/utility/Assert.h>

#include <simd_stl/numeric/BasicSimdMask.h>
#include <simd_stl/numeric/SimdCast.h>

#include <simd_stl/numeric/BasicSimdElementReference.h>
#include <src/simd_stl/numeric/ZmmThreshold.h>

#include <simd_stl/numeric/SimdCompareResult.h>
#include <src/simd_stl/numeric/SimdCompareAdapters.h>


__SIMD_STL_NUMERIC_NAMESPACE_BEGIN

using aligned_policy    = __aligned_policy;
using unaligned_policy  = __unaligned_policy;

using simd_comparison = __simd_comparison;

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
class simd {
    static_assert(type_traits::__is_generation_supported_v<_SimdGeneration_>);
    static_assert(type_traits::__is_vector_type_supported_v<std::decay_t<_Element_>>);

    friend simd_element_reference;

    template <typename _DesiredType_>
    using __mask_type = type_traits::__deduce_simd_mask_type<_SimdGeneration_, _Element_, _RegisterPolicy_>;

    template <typename _DesiredType_>
    using __reference_type = simd_element_reference<simd, _DesiredType_>;
public:
    using policy_type = _RegisterPolicy_;
    static constexpr auto __generation = _SimdGeneration_;

    using value_type    = _Element_;
    using vector_type   = type_traits::__deduce_simd_vector_type<_SimdGeneration_, _Element_, _RegisterPolicy_>;

    using size_type     = uint32;
    using mask_type     = simd_mask<_SimdGeneration_, _Element_, _RegisterPolicy_>;

    using reference_type = __reference_type<value_type>;

    template <typename _DesiredType_ = value_type>
    static constexpr inline bool is_native_mask_load_supported_v = __is_native_mask_load_supported_v<
        __generation, policy_type, _DesiredType_>;

    template <typename _DesiredType_ = value_type>
    static constexpr inline bool is_native_mask_store_supported_v = __is_native_mask_store_supported_v<
        __generation, policy_type, _DesiredType_>;

    simd() noexcept;

    simd(const value_type __value) noexcept;

    template <
        typename _VectorType_,
        std::enable_if_t<__is_intrin_type_v<_VectorType_> || __is_valid_basic_simd_v<_VectorType_>, int> = 0>
    simd(_VectorType_ __other) noexcept;

    ~simd() noexcept;

    simd_stl_always_inline simd& clear() noexcept;

    template <typename _DesiredType_ = value_type>
    simd_stl_always_inline simd& fill(typename std::type_identity<_DesiredType_>::type __value) noexcept;

    template <typename _DesiredType_>
    simd_stl_always_inline _DesiredType_ extract(size_type __index) const noexcept;

    template <typename _DesiredType_>
    simd_stl_always_inline simd_element_reference<simd> extract_wrapped(size_type __index) noexcept;

    template <typename _DesiredType_>
    simd_stl_always_inline void insert(
        size_type                                         __index,
        typename std::type_identity<_DesiredType_>::type  __value) noexcept;

    template <class _AlignmentPolicy_ = unaligned_policy>
    static simd_stl_always_inline simd load(
        const void*         __address,
        _AlignmentPolicy_&& __policy = _AlignmentPolicy_{}) noexcept;

    template <class _AlignmentPolicy_ = unaligned_policy>
    simd_stl_always_inline void store(
        void*               __address,
        _AlignmentPolicy_&& __policy = _AlignmentPolicy_{}) const noexcept;

    template <
        typename    _DesiredType_       = value_type,
        class       _MaskType_          = __mask_type<_DesiredType_>,
        class       _AlignmentPolicy_   = unaligned_policy>
    static simd_stl_always_inline simd mask_load(
        const void*         __address,
        const _MaskType_&   __mask,
        _AlignmentPolicy_&& __policy = _AlignmentPolicy_{}) noexcept;

    template <
        typename    _DesiredType_       = value_type,
        class       _MaskType_          = __mask_type<_DesiredType_>,
        class       _VectorType_        = simd,
        class       _AlignmentPolicy_   = unaligned_policy,
        std::enable_if_t<__is_intrin_type_v<_VectorType_> || __is_valid_basic_simd_v<_VectorType_>, int> = 0>
    static simd_stl_always_inline simd mask_load(
        const void*         __address,
        const _MaskType_&   __mask,
        _VectorType_        __additional_source,
        _AlignmentPolicy_&& __policy = _AlignmentPolicy_{}) noexcept;

    template <
        typename    _DesiredType_       = value_type,
        class       _MaskType_          = __mask_type<_DesiredType_>,
        class       _AlignmentPolicy_   = unaligned_policy>
    simd_stl_always_inline void mask_store(
        void*               __address,
        const _MaskType_&   __mask,
        _AlignmentPolicy_&& __policy = _AlignmentPolicy_{}) const noexcept;

    simd_stl_always_inline static simd non_temporal_load(const void* __address) noexcept;
    simd_stl_always_inline void non_temporal_store(void* __address) const noexcept;

    template <
        typename    _DesiredType_       = value_type,
        class       _MaskType_          = __mask_type<_DesiredType_>,
        class       _AlignmentPolicy_   = unaligned_policy>
    simd_stl_always_inline _DesiredType_* compress_store(
        void*               __address,
        const _MaskType_&   __mask,
        _AlignmentPolicy_&& __policy = _AlignmentPolicy_{}) const noexcept;

    template <
        typename    _DesiredType_   = value_type,
        class       _MaskType_      = __mask_type<_DesiredType_>>
    simd_stl_always_inline int32 compress(const _MaskType_& __mask) noexcept;

    simd_stl_always_inline friend simd operator+  <>(const simd& __left, const value_type __right) noexcept;
    simd_stl_always_inline friend simd operator-  <>(const simd& __left, const value_type __right) noexcept;
    simd_stl_always_inline friend simd operator*  <>(const simd& __left, const value_type __right) noexcept;
    simd_stl_always_inline friend simd operator/  <>(const simd& __left, const value_type __right) noexcept;
    simd_stl_always_inline friend simd operator+  <>(const simd& __left, const simd& __right) noexcept;
    simd_stl_always_inline friend simd operator-  <>(const simd& __left, const simd& __right) noexcept;
    simd_stl_always_inline friend simd operator*  <>(const simd& __left, const simd& __right) noexcept;
    simd_stl_always_inline friend simd operator/  <>(const simd& __left, const simd& __right) noexcept;
    simd_stl_always_inline friend simd operator&  <>(const simd& __left, const simd& __right) noexcept;
    simd_stl_always_inline friend simd operator|  <>(const simd& __left, const simd& __right) noexcept;
    simd_stl_always_inline friend simd operator^  <>(const simd& __left, const simd& __right) noexcept;
    //simd_stl_always_inline friend simd operator>> <>(const simd& __left, const uint32 _Shift) noexcept;
    //simd_stl_always_inline friend simd operator<< <>(const simd& __left, const uint32 _Shift) noexcept;

    simd_stl_always_inline simd& operator&=(const simd& __other) noexcept;
    simd_stl_always_inline simd& operator|=(const simd& __other) noexcept;
    simd_stl_always_inline simd& operator^=(const simd& __other) noexcept;
    simd_stl_always_inline simd& operator+=(const simd& __other) noexcept;
    simd_stl_always_inline simd& operator-=(const simd& __other) noexcept;
    simd_stl_always_inline simd& operator*=(const simd& __other) noexcept;
    simd_stl_always_inline simd& operator/=(const simd& __other) noexcept;
    //simd_stl_always_inline simd& operator>>=(const uint32 _Shift) noexcept;
    //simd_stl_always_inline simd& operator<<=(const uint32 _Shift) noexcept;

    simd_stl_always_inline simd  operator+()     const noexcept;
    simd_stl_always_inline simd  operator-()     const noexcept;
    simd_stl_always_inline simd  operator++(int) noexcept;
    simd_stl_always_inline simd& operator++()    noexcept;
    simd_stl_always_inline simd  operator--(int) noexcept;
    simd_stl_always_inline simd& operator--()    noexcept;

    simd_stl_always_inline simd operator~() const noexcept;
    simd_stl_always_inline simd& operator=(const simd& __other) noexcept;

    simd_stl_always_inline _Element_ operator[](const size_type __index) const noexcept;
    simd_stl_always_inline simd_element_reference<simd> operator[](const size_type __index) noexcept;

    simd_stl_always_inline friend simd_compare_result<_SimdGeneration_, _Element_, _RegisterPolicy_, simd_comparison::equal> 
        operator== <>(const simd& __left, const simd& __right) noexcept;

    simd_stl_always_inline friend simd_compare_result<_SimdGeneration_, _Element_, _RegisterPolicy_, simd_comparison::not_equal> 
        operator!= <>(const simd& __left, const simd& __right) noexcept;

    simd_stl_always_inline friend simd_compare_result<_SimdGeneration_, _Element_, _RegisterPolicy_, simd_comparison::less>
        operator< <>(const simd& __left, const simd& __right) noexcept;

    simd_stl_always_inline friend simd_compare_result<_SimdGeneration_, _Element_, _RegisterPolicy_, simd_comparison::less_equal>
        operator<= <>(const simd& __left, const simd& __right) noexcept;

    simd_stl_always_inline friend simd_compare_result<_SimdGeneration_, _Element_, _RegisterPolicy_, simd_comparison::greater>
        operator> <>(const simd& __left, const simd& __right) noexcept;

    simd_stl_always_inline friend simd_compare_result<_SimdGeneration_, _Element_, _RegisterPolicy_, simd_comparison::greater_equal>
        operator>= <>(const simd& __left, const simd& __right) noexcept;

    template <typename _DesiredType_ = value_type>
    simd_stl_always_inline simd_mask<_SimdGeneration_, _DesiredType_, _RegisterPolicy_> to_mask() const noexcept;

    template <typename _DesiredType_ = value_type>
    simd_stl_always_inline auto to_index_mask() const noexcept;

    template <typename _DesiredType_ = value_type>
    simd_stl_always_inline __reduce_type<_DesiredType_> reduce_add() const noexcept;

    simd_stl_always_inline static void streaming_fence() noexcept;

    template <
        class       _MaskType_,
        typename    _DesiredType_ = _Element_>
    simd_stl_always_inline void blend(
        const simd<_SimdGeneration_, _DesiredType_, _RegisterPolicy_>&  __vector,
        const _MaskType_&                                               __mask) noexcept;

    template <typename _DesiredType_ = _Element_>
    simd_stl_always_inline simd<_SimdGeneration_, _DesiredType_, _RegisterPolicy_> vertical_min(const simd<_SimdGeneration_, _DesiredType_, _RegisterPolicy_>& __other) const noexcept;

    template <typename _DesiredType_ = _Element_>
    simd_stl_always_inline simd<_SimdGeneration_, _DesiredType_, _RegisterPolicy_> vertical_max(const simd<_SimdGeneration_, _DesiredType_, _RegisterPolicy_>& __other) const noexcept;

    template <typename _DesiredType_ = _Element_>
    simd_stl_always_inline _DesiredType_ horizontal_min() const noexcept;

    template <typename _DesiredType_ = _Element_>
    simd_stl_always_inline _DesiredType_ horizontal_max() const noexcept;

    template <typename _DesiredType_ = _Element_>
    simd_stl_always_inline simd<_SimdGeneration_, _DesiredType_, _RegisterPolicy_> abs() const noexcept;

    template <typename _DesiredType_ = _Element_>
    simd_stl_always_inline void reverse() noexcept;

    template <typename _ElementType_ = _Element_>
    static simd_stl_always_inline constexpr int width() noexcept;

    template <typename _ElementType_ = _Element_>
    static simd_stl_always_inline constexpr int size() noexcept;

    template <typename _ElementType_ = _Element_>
    static simd_stl_always_inline constexpr int length() noexcept;

    static simd_stl_always_inline bool is_supported() noexcept;

    simd_stl_always_inline vector_type unwrap() const noexcept;

    static simd_stl_always_inline __make_tail_mask_return_type<simd> make_tail_mask(uint32 __bytes) noexcept;
private:
    vector_type _vector;
};

template <arch::CpuFeature _SimdGeneration_>
struct __zero_upper_at_exit_guard {
    __zero_upper_at_exit_guard(const __zero_upper_at_exit_guard&) noexcept = delete;
    __zero_upper_at_exit_guard(__zero_upper_at_exit_guard&&) noexcept = delete;

    __zero_upper_at_exit_guard() noexcept;
    ~__zero_upper_at_exit_guard() noexcept;
};

template <arch::CpuFeature _SimdGeneration_>
simd_stl_always_inline __zero_upper_at_exit_guard<_SimdGeneration_> make_guard() noexcept;

template <class _Simd_, std::enable_if_t<__is_valid_basic_simd_v<_Simd_>, int> = 0>
simd_stl_always_inline __zero_upper_at_exit_guard<_Simd_::__generation> make_guard() noexcept;

__SIMD_STL_NUMERIC_NAMESPACE_END

#include <src/simd_stl/numeric/BasicSimd.inl>