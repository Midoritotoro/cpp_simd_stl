#pragma once 

#include <src/simd_stl/numeric/SimdArithmetic.h>

#include <src/simd_stl/numeric/SimdCompare.h>
#include <src/simd_stl/numeric/SimdConvert.h>

#include <src/simd_stl/utility/Assert.h>

#include <simd_stl/numeric/BasicSimdMask.h>
#include <simd_stl/numeric/SimdCast.h>

#include <simd_stl/numeric/BasicSimdElementReference.h>
#include <src/simd_stl/numeric/ZmmThreshold.h>


__SIMD_STL_NUMERIC_NAMESPACE_BEGIN

struct aligned_policy {
    static constexpr bool __alignment    = true;
};

struct unaligned_policy {
    static constexpr bool __alignment    = false;
};

using simd_comparison = __simd_comparison;

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
class simd {
    static_assert(type_traits::__is_generation_supported_v<_SimdGeneration_>);
    static_assert(type_traits::__is_vector_type_supported_v<std::decay_t<_Element_>>);

    friend BasicSimdElementReference;

    template <typename _DesiredType_>
    using __mask_type = type_traits::__deduce_simd_mask_type<_SimdGeneration_, _Element_, _RegisterPolicy_>;

    template <typename _DesiredType_>
    using __reference_type = BasicSimdElementReference<simd, _DesiredType_>;
public:
    using policy_type = _RegisterPolicy_;
    static constexpr auto __generation = _SimdGeneration_;

    using value_type    = _Element_;
    using vector_type   = type_traits::__deduce_simd_vector_type<_SimdGeneration_, _Element_, _RegisterPolicy_>;

    using size_type     = uint32;
    using mask_type     = __mask_type<_Element_>;

    using reference_type = __reference_type<value_type>;

    template <typename _DesiredType_ = value_type>
    static constexpr inline bool is_native_mask_load_supported_v = _Is_native_mask_load_supported_v<
        _Generation, policy_type, _DesiredType_>;

    template <typename _DesiredType_ = value_type>
    static constexpr inline bool is_native_mask_store_supported_v = _Is_native_mask_store_supported_v<
        _Generation, policy_type, _DesiredType_>;

    simd() noexcept;

    simd(const value_type __value) noexcept;

    template <
        typename _VectorType_,
        std::enable_if_t<__is_intrin_type_v<_VectorType_> || __is_valid_basic_simd_v<_VectorType_>, int> = 0>
    simd(_VectorType_ _Other) noexcept;

    //template <class _Range_>
    //simd(_Range_&& _Range) noexcept;

    ~simd() noexcept;

    simd_stl_always_inline simd& clear() noexcept;

    template <typename _DesiredType_ = value_type>
    simd_stl_always_inline simd& fill(typename std::type_identity<_DesiredType_>::type value) noexcept;

    template <typename _DesiredType_>
    simd_stl_always_inline _DesiredType_ extract(size_type index) const noexcept;

    template <typename _DesiredType_>
    simd_stl_always_inline BasicSimdElementReference<simd> extract_wrapped(size_type index) noexcept;

    template <typename _DesiredType_>
    simd_stl_always_inline void insert(
        size_type                                         _Index,
        typename std::type_identity<_DesiredType_>::type  _Value) noexcept;

    template <class _AlignmentPolicy_ = unaligned_policy>
    static simd_stl_always_inline simd load(
        const void*         _Address,
        _AlignmentPolicy_&& _Policy = _AlignmentPolicy_{}) noexcept;

    template <class _AlignmentPolicy_ = unaligned_policy>
    simd_stl_always_inline void store(
        void*               _Address,
        _AlignmentPolicy_&& _Policy = _AlignmentPolicy_{}) const noexcept;

    template <
        typename    _DesiredType_       = value_type,
        class       _MaskType_          = __mask_type<_DesiredType_>,
        class       _AlignmentPolicy_   = unaligned_policy>
    static simd_stl_always_inline simd mask_load(
        const void*         _Address,
        const _MaskType_&   _Mask,
        _AlignmentPolicy_&& _Policy = _AlignmentPolicy_{}) noexcept;

    template <
        typename    _DesiredType_       = value_type,
        class       _MaskType_          = __mask_type<_DesiredType_>,
        class       _VectorType_        = simd,
        class       _AlignmentPolicy_   = unaligned_policy,
        std::enable_if_t<__is_intrin_type_v<_VectorType_> || __is_valid_basic_simd_v<_VectorType_>, int> = 0>
    static simd_stl_always_inline simd mask_load(
        const void*         _Address,
        const _MaskType_&   _Mask,
        _VectorType_        _AdditionalSource,
        _AlignmentPolicy_&& _Policy = _AlignmentPolicy_{}) noexcept;

    template <
        typename    _DesiredType_       = value_type,
        class       _MaskType_          = __mask_type<_DesiredType_>,
        class       _AlignmentPolicy_   = unaligned_policy>
    simd_stl_always_inline void mask_store(
        void*               _Address,
        const _MaskType_&   _Mask,
        _AlignmentPolicy_&& _Policy = _AlignmentPolicy_{}) const noexcept;

    simd_stl_always_inline static simd non_temporal_load(const void* _Address) noexcept;
    simd_stl_always_inline void non_temporal_store(void* _Address) const noexcept;

    template <
        typename    _DesiredType_       = value_type,
        class       _MaskType_          = __mask_type<_DesiredType_>,
        class       _AlignmentPolicy_   = unaligned_policy>
    simd_stl_always_inline _DesiredType_* compress_store(
        void*               _Address,
        const _MaskType_&   _Mask,
        _AlignmentPolicy_&& _Policy = unaligned_policy{}) const noexcept;

    simd_stl_always_inline friend simd operator+  <>(const simd& _Left, const value_type _Right) noexcept;
    simd_stl_always_inline friend simd operator-  <>(const simd& _Left, const value_type _Right) noexcept;
    simd_stl_always_inline friend simd operator*  <>(const simd& _Left, const value_type _Right) noexcept;
    simd_stl_always_inline friend simd operator/  <>(const simd& _Left, const value_type _Right) noexcept;
    simd_stl_always_inline friend simd operator+  <>(const simd& _Left, const simd& _Right) noexcept;
    simd_stl_always_inline friend simd operator-  <>(const simd& _Left, const simd& _Right) noexcept;
    simd_stl_always_inline friend simd operator*  <>(const simd& _Left, const simd& _Right) noexcept;
    simd_stl_always_inline friend simd operator/  <>(const simd& _Left, const simd& _Right) noexcept;
    simd_stl_always_inline friend simd operator&  <>(const simd& _Left, const simd& _Right) noexcept;
    simd_stl_always_inline friend simd operator|  <>(const simd& _Left, const simd& _Right) noexcept;
    simd_stl_always_inline friend simd operator^  <>(const simd& _Left, const simd& _Right) noexcept;
    //simd_stl_always_inline friend simd operator>> <>(const simd& _Left, const uint32 _Shift) noexcept;
    //simd_stl_always_inline friend simd operator<< <>(const simd& _Left, const uint32 _Shift) noexcept;

    simd_stl_always_inline simd& operator&=(const simd& _Other) noexcept;
    simd_stl_always_inline simd& operator|=(const simd& _Other) noexcept;
    simd_stl_always_inline simd& operator^=(const simd& _Other) noexcept;
    simd_stl_always_inline simd& operator+=(const simd& _Other) noexcept;
    simd_stl_always_inline simd& operator-=(const simd& _Other) noexcept;
    simd_stl_always_inline simd& operator*=(const simd& _Other) noexcept;
    simd_stl_always_inline simd& operator/=(const simd& _Other) noexcept;
    //simd_stl_always_inline simd& operator>>=(const uint32 _Shift) noexcept;
    //simd_stl_always_inline simd& operator<<=(const uint32 _Shift) noexcept;

    simd_stl_always_inline simd  operator+()     const noexcept;
    simd_stl_always_inline simd  operator-()     const noexcept;
    simd_stl_always_inline simd  operator++(int) noexcept;
    simd_stl_always_inline simd& operator++()    noexcept;
    simd_stl_always_inline simd  operator--(int) noexcept;
    simd_stl_always_inline simd& operator--()    noexcept;

    simd_stl_always_inline simd operator~() const noexcept;
    simd_stl_always_inline simd& operator=(const simd& _Lseft) noexcept;

    simd_stl_always_inline _Element_ operator[](const size_type _Index) const noexcept;
    simd_stl_always_inline BasicSimdElementReference<simd> operator[](const size_type _Index) noexcept;

    simd_stl_always_inline friend bool operator== <>(const simd& _Left, const simd& _Right) noexcept;
    simd_stl_always_inline friend bool operator!= <>(const simd& _Left, const simd& _Right) noexcept;

    template <
        simd_comparison _Comparison_,
        typename        _DesiredType_ = value_type>
    simd_stl_always_inline simd_mask<_SimdGeneration_, _DesiredType_, _RegisterPolicy_> mask_compare(const simd& _Right) const noexcept;
    
    template <
        simd_comparison _Comparison_,
        typename        _DesiredType_ = value_type>
    simd_stl_always_inline simd<_SimdGeneration_, _DesiredType_, _RegisterPolicy_> vector_compare(const simd& _Right) const noexcept;

    template <
        simd_comparison _Comparison_,
        typename        _DesiredType_ = value_type> 
    simd_stl_always_inline _Native_compare_return_type<simd, _DesiredType_, _Predicate_> native_compare(const simd& _Right) const noexcept;

    template <typename _DesiredType_ = value_type>
    simd_stl_always_inline simd_mask<_SimdGeneration_, _DesiredType_, _RegisterPolicy_> to_mask() const noexcept;

    template <typename _DesiredType_ = value_type>
    simd_stl_always_inline _Reduce_type<_DesiredType_> reduce_add() const noexcept;

    simd_stl_always_inline static void streaming_fence() noexcept;

    template <
        class       _MaskType_,
        typename    _DesiredType_ = _Element_>
    simd_stl_always_inline void blend(
        const simd<_SimdGeneration_, _DesiredType_, _RegisterPolicy_>&  _Vector,
        const _MaskType_&                                               _Mask) noexcept;

    template <typename _DesiredType_ = _Element_>
    simd_stl_always_inline simd<_SimdGeneration_, _DesiredType_, _RegisterPolicy_> vertical_min(
        const simd<_SimdGeneration_, _DesiredType_, _RegisterPolicy_>& _Other) const noexcept;

    template <typename _DesiredType_ = _Element_>
    simd_stl_always_inline _DesiredType_ horizontal_min() const noexcept;

    template <typename _DesiredType_ = _Element_>
    simd_stl_always_inline simd<_SimdGeneration_, _DesiredType_, _RegisterPolicy_> vertical_max(
        const simd<_SimdGeneration_, _DesiredType_, _RegisterPolicy_>& _Other) const noexcept;

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

    static simd_stl_always_inline _Make_tail_mask_return_type<simd> make_tail_mask(uint32 bytes) noexcept;
private:
    vector_type _vector;
};

template <arch::CpuFeature _SimdGeneration_>
struct zero_upper_at_exit_guard {
    zero_upper_at_exit_guard(const zero_upper_at_exit_guard&) noexcept = delete;
    zero_upper_at_exit_guard(zero_upper_at_exit_guard&&) noexcept  = delete;

    zero_upper_at_exit_guard() noexcept;
    ~zero_upper_at_exit_guard() noexcept;
};

__SIMD_STL_NUMERIC_NAMESPACE_END

#include <src/simd_stl/numeric/BasicSimd.inl>