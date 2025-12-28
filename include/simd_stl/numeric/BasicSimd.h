#pragma once 

#include <src/simd_stl/numeric/SimdArithmetic.h>

#include <src/simd_stl/numeric/SimdCompare.h>
#include <src/simd_stl/numeric/SimdConvert.h>

#include <src/simd_stl/utility/Assert.h>

#include <simd_stl/numeric/BasicSimdMask.h>
#include <simd_stl/numeric/SimdCast.h>

#include <simd_stl/numeric/BasicSimdElementReference.h>


__SIMD_STL_NUMERIC_NAMESPACE_BEGIN

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
class simd {
    static_assert(type_traits::__is_generation_supported_v<_SimdGeneration_>);
    static_assert(type_traits::__is_vector_type_supported_v<std::decay_t<_Element_>>);

    friend BasicSimdElementReference;

    template <typename _DesiredType_>
    using _Mask_type = type_traits::__deduce_simd_mask_type<_SimdGeneration_, _Element_, _RegisterPolicy_>;
public:
    using policy_type = _RegisterPolicy_;
    static constexpr auto _Generation = _SimdGeneration_;

    using value_type    = _Element_;
    using vector_type   = type_traits::__deduce_simd_vector_type<_SimdGeneration_, _Element_, _RegisterPolicy_>;

    using size_type     = uint32;
    using mask_type     = _Mask_type<_Element_>;

    template <typename _DesiredType_ = value_type>
    static constexpr inline bool is_native_mask_load_supported_v = _Is_native_mask_load_supported_v<
        _Generation, policy_type, _DesiredType_>;

    template <typename _DesiredType_ = value_type>
    static constexpr inline bool is_native_mask_store_supported_v = _Is_native_mask_store_supported_v<
        _Generation, policy_type, _DesiredType_>;

    template <bool _ZeroMemset_ = false>
    simd() noexcept;

    simd(const value_type value) noexcept;

    template <
        typename _IntrinType_,
        std::enable_if_t<_Is_intrin_type_v<_IntrinType_>, int> = 0>
    simd(_IntrinType_ other) noexcept;

    template <typename _OtherType_>
    simd(const simd<_SimdGeneration_, _OtherType_, _RegisterPolicy_>& other) noexcept;

    template <
        arch::CpuFeature    _OtherFeature_,
        typename            _OtherType_>
    simd(const simd<_OtherFeature_, _OtherType_, _RegisterPolicy_>& other) noexcept;

    ~simd() noexcept;

    template <class _BasicSimdTo_>
    simd_stl_always_inline _BasicSimdTo_ convert() const noexcept;

    simd_stl_always_inline void broadcastZeros() noexcept;

    template <typename _DesiredType_ = value_type>
    simd_stl_always_inline void fill(const typename std::type_identity<_DesiredType_>::type value) noexcept;

    template <typename _DesiredType_>
    simd_stl_always_inline _DesiredType_ extract(const size_type index) const noexcept;

    template <typename _DesiredType_>
    simd_stl_always_inline BasicSimdElementReference<simd> extractWrapped(const size_type index) noexcept;

    template <typename _DesiredType_>
    simd_stl_always_inline void insert(
        const size_type                                         where,
        const typename std::type_identity<_DesiredType_>::type  value) noexcept;

    static simd_stl_always_inline simd loadUnaligned(const void* where) noexcept;
    static simd_stl_always_inline simd loadAligned(const void* where) noexcept;

    simd_stl_always_inline void storeUnaligned(void* where) const noexcept;
    simd_stl_always_inline void storeAligned(void* where) const noexcept;

    static simd_stl_always_inline simd loadUpperHalf(const void* where) noexcept;
    static simd_stl_always_inline simd loadLowerHalf(const void* where) noexcept;

    template <typename _DesiredType_ = value_type>
    static simd_stl_always_inline simd maskLoadUnaligned(
        const void*                             where,
        const _Mask_type<_DesiredType_> mask) noexcept;

    template <typename _DesiredType_ = value_type>
    static simd_stl_always_inline simd maskLoadAligned(
        const void*                             where,
        const _Mask_type<_DesiredType_> mask) noexcept;

    template <typename _DesiredType_ = value_type>
    static simd_stl_always_inline simd maskLoadUnaligned(
        const void*         where,
        const simd<_SimdGeneration_, _DesiredType_, _RegisterPolicy_>&   mask) noexcept;

    template <typename _DesiredType_ = value_type>
    static simd_stl_always_inline simd maskLoadAligned(
        const void* where,
        const simd<_SimdGeneration_, _DesiredType_, _RegisterPolicy_>& mask) noexcept;

    template <typename _DesiredType_ = value_type>
    simd_stl_always_inline void maskStoreUnaligned(
        void*                                   where,
        const _Mask_type<_DesiredType_> mask) const noexcept;

    template <typename _DesiredType_ = value_type>
    simd_stl_always_inline void maskStoreAligned(
        void*                                   where,
        const _Mask_type<_DesiredType_> mask) const noexcept;

    template <typename _DesiredType_ = value_type>
    simd_stl_always_inline void maskStoreUnaligned(
        void*                                                                   where,
        const simd<_SimdGeneration_, _DesiredType_, _RegisterPolicy_>&    mask) const noexcept;

    template <typename _DesiredType_ = value_type>
    simd_stl_always_inline void maskStoreAligned(
        void*                                                                   where,
        const simd<_SimdGeneration_, _DesiredType_, _RegisterPolicy_>&    mask) const noexcept;

    template <typename _DesiredType_ = value_type>
    simd_stl_always_inline void maskBlendStoreUnaligned(
        void*                               where,
        const _Mask_type<_DesiredType_> mask,
        const simd&                   source) const noexcept;

    template <typename _DesiredType_ = value_type>
    simd_stl_always_inline void maskBlendStoreAligned(
        void*                               where,
        const _Mask_type<_DesiredType_>    mask,
        const simd&                   source) const noexcept;

    template <typename _DesiredType_ = value_type>
    simd_stl_always_inline void maskBlendStoreUnaligned(
        void*               where,
        const simd&   mask,
        const simd&   source) const noexcept;

    template <typename _DesiredType_ = value_type>
    simd_stl_always_inline void maskBlendStoreAligned(
        void*               where,
        const simd&   mask,
        const simd&   source) const noexcept;

    simd_stl_always_inline static simd nonTemporalLoad(const void* where) noexcept;
    simd_stl_always_inline void nonTemporalStore(void* where) const noexcept;

    template <typename _DesiredType_ = _Element_> 
    simd_stl_always_inline _DesiredType_* compressStoreUnaligned(
        void*                       where,
        _Mask_type<_DesiredType_>   mask) const noexcept;

    template <typename _DesiredType_ = _Element_> 
    simd_stl_always_inline _DesiredType_* compressStoreAligned(
        void*                       where,
        _Mask_type<_DesiredType_>   mask) const noexcept;


    simd_stl_always_inline friend simd operator+<>(
        const simd&   left,
        const value_type    right) noexcept;

    simd_stl_always_inline friend simd operator-<>(
        const simd&   left,
        const value_type    right) noexcept;

    simd_stl_always_inline friend simd operator*<>(
        const simd&   left,
        const value_type    right) noexcept;
  
    simd_stl_always_inline friend simd operator/<>(
        const simd&   left,
        const value_type    right) noexcept;

    simd_stl_always_inline friend simd operator+ <>(
        const simd& left,
        const simd& right) noexcept;

    simd_stl_always_inline friend simd operator- <>(
        const simd& left,
        const simd& right) noexcept;

    simd_stl_always_inline friend simd operator* <>(
        const simd& left,
        const simd& right) noexcept;

    simd_stl_always_inline friend simd operator/ <>(
        const simd& left,
        const simd& right) noexcept;

    simd_stl_always_inline friend simd operator& <>(
        const simd& left,
        const simd& right) noexcept;

    simd_stl_always_inline friend simd operator| <>(
        const simd& left,
        const simd& right) noexcept;

    simd_stl_always_inline friend simd operator^ <>(
        const simd& left,
        const simd& right) noexcept;

    simd_stl_always_inline friend simd operator>> <>(
        const simd left,
        const uint32 shift) noexcept;

    simd_stl_always_inline friend simd operator<< <>(
        const simd left,
        const uint32 shift) noexcept;

    simd_stl_always_inline simd operator+() const noexcept;
    simd_stl_always_inline simd operator-() const noexcept;

    simd_stl_always_inline simd operator++(int) noexcept;
    simd_stl_always_inline simd& operator++() noexcept;
    simd_stl_always_inline simd operator--(int) noexcept;
    simd_stl_always_inline simd& operator--() noexcept;

    simd_stl_always_inline mask_type operator!() const noexcept;
    simd_stl_always_inline simd operator~() const noexcept;

    simd_stl_always_inline simd& operator=(const simd& left) noexcept;

    simd_stl_always_inline _Element_ operator[](const size_type index) const noexcept;
    simd_stl_always_inline BasicSimdElementReference<simd> operator[](const size_type index) noexcept;

    simd_stl_always_inline simd& operator&=(const simd& other) noexcept;
    simd_stl_always_inline simd& operator|=(const simd& other) noexcept;
    simd_stl_always_inline simd& operator^=(const simd& other) noexcept;
    simd_stl_always_inline simd& operator+=(const simd& other) noexcept;
    simd_stl_always_inline simd& operator-=(const simd& other) noexcept;
    simd_stl_always_inline simd& operator*=(const simd& other) noexcept;
    simd_stl_always_inline simd& operator/=(const simd& other) noexcept;
    simd_stl_always_inline simd& operator>>=(const uint32 shift) noexcept;
    simd_stl_always_inline simd& operator<<=(const uint32 shift) noexcept;


    simd_stl_always_inline friend bool operator== <>(
        const simd& left,
        const simd& right) noexcept;

    template <typename _DesiredType_ = value_type>
    simd_stl_always_inline bool isEqual(const simd& right) const noexcept;

    template <typename _DesiredType_ = value_type>
    simd_stl_always_inline basic_simd_mask<_SimdGeneration_, _DesiredType_, _RegisterPolicy_> maskNotEqual(const simd& right) const noexcept;

    template <typename _DesiredType_ = value_type>
    simd_stl_always_inline basic_simd_mask<_SimdGeneration_, _DesiredType_, _RegisterPolicy_> maskEqual(const simd& right) const noexcept;

    template <typename _DesiredType_ = value_type>
    simd_stl_always_inline basic_simd_mask<_SimdGeneration_, _DesiredType_, _RegisterPolicy_> maskGreater(const simd& right) const noexcept;

    template <typename _DesiredType_ = value_type>
    simd_stl_always_inline basic_simd_mask<_SimdGeneration_, _DesiredType_, _RegisterPolicy_> maskLess(const simd& right) const noexcept;

    template <typename _DesiredType_ = value_type>
    simd_stl_always_inline basic_simd_mask<_SimdGeneration_, _DesiredType_, _RegisterPolicy_> maskGreaterEqual(const simd& right) const noexcept;

    template <typename _DesiredType_ = value_type>
    simd_stl_always_inline basic_simd_mask<_SimdGeneration_, _DesiredType_, _RegisterPolicy_> maskLessEqual(const simd& right) const noexcept;

    
    template <typename _DesiredType_ = value_type>
    simd_stl_always_inline simd<_SimdGeneration_, _DesiredType_, _RegisterPolicy_> notEqual(const simd& right) const noexcept;

    template <typename _DesiredType_ = value_type>
    simd_stl_always_inline simd<_SimdGeneration_, _DesiredType_, _RegisterPolicy_> equal(const simd& right) const noexcept;

    template <typename _DesiredType_ = value_type>
    simd_stl_always_inline simd<_SimdGeneration_, _DesiredType_, _RegisterPolicy_> greater(const simd& right) const noexcept;

    template <typename _DesiredType_ = value_type>
    simd_stl_always_inline simd<_SimdGeneration_, _DesiredType_, _RegisterPolicy_> less(const simd& right) const noexcept;

    template <typename _DesiredType_ = value_type>
    simd_stl_always_inline simd<_SimdGeneration_, _DesiredType_, _RegisterPolicy_> greaterEqual(const simd& right) const noexcept;

    template <typename _DesiredType_ = value_type>
    simd_stl_always_inline simd<_SimdGeneration_, _DesiredType_, _RegisterPolicy_> lessEqual(const simd& right) const noexcept;


    template <typename _DesiredType_ = value_type>
    simd_stl_always_inline _Native_compare_return_type<simd, _DesiredType_, type_traits::not_equal_to<>> 
        nativeNotEqual(const simd& right) const noexcept;

    template <typename _DesiredType_ = value_type>
    simd_stl_always_inline _Native_compare_return_type<simd, _DesiredType_, type_traits::equal_to<>>
        nativeEqual(const simd& right) const noexcept;

    template <typename _DesiredType_ = value_type>
    simd_stl_always_inline _Native_compare_return_type<simd, _DesiredType_, type_traits::greater<>>
        nativeGreater(const simd& right) const noexcept;

    template <typename _DesiredType_ = value_type>
    simd_stl_always_inline _Native_compare_return_type<simd, _DesiredType_, type_traits::less<>>
        nativeLess(const simd& right) const noexcept;

    template <typename _DesiredType_ = value_type>
    simd_stl_always_inline _Native_compare_return_type<simd, _DesiredType_, type_traits::greater_equal<>>
        nativeGreaterEqual(const simd& right) const noexcept;

    template <typename _DesiredType_ = value_type>
    simd_stl_always_inline _Native_compare_return_type<simd, _DesiredType_, type_traits::less_equal<>>
        nativeLessEqual(const simd& right) const noexcept;

    template <typename _DesiredType_ = value_type>
    simd_stl_always_inline basic_simd_mask<_SimdGeneration_, _DesiredType_, _RegisterPolicy_> toMask() const noexcept;

    template <typename _DesiredType_ = value_type>
    simd_stl_always_inline _Reduce_type<_DesiredType_> reduce() const noexcept;

    simd_stl_always_inline static void streamingFence() noexcept;
    static simd_stl_always_inline void zeroUpper() noexcept;

    template <typename _DesiredType_ = _Element_>
    simd_stl_always_inline void blend(
        const simd<_SimdGeneration_, _DesiredType_, _RegisterPolicy_>&    _Vector,
        _Mask_type<_DesiredType_>                                               _Mask) noexcept;

    template <typename _DesiredType_ = _Element_>
    simd_stl_always_inline void blend(
        const simd<_SimdGeneration_, _DesiredType_, _RegisterPolicy_>& _Vector,
        const simd<_SimdGeneration_, _DesiredType_, _RegisterPolicy_>& _Mask) noexcept;

    template <typename _DesiredType_ = _Element_>
    simd_stl_always_inline simd<_SimdGeneration_, _DesiredType_, _RegisterPolicy_> verticalMin(
        const simd<_SimdGeneration_, _DesiredType_, _RegisterPolicy_>& _Other) const noexcept;

    template <typename _DesiredType_ = _Element_>
    simd_stl_always_inline _DesiredType_ horizontalMin() const noexcept;

    template <typename _DesiredType_ = _Element_>
    simd_stl_always_inline simd<_SimdGeneration_, _DesiredType_, _RegisterPolicy_> verticalMax(
        const simd<_SimdGeneration_, _DesiredType_, _RegisterPolicy_>& _Other) const noexcept;

    template <typename _DesiredType_ = _Element_>
    simd_stl_always_inline _DesiredType_ horizontalMax() const noexcept;

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

    static simd_stl_always_inline constexpr int registersCount() noexcept;
    static simd_stl_always_inline bool isSupported() noexcept;

    simd_stl_always_inline vector_type unwrap() const noexcept;

    static simd_stl_always_inline _Make_tail_mask_return_type<simd> makeTailMask(uint32 bytes) noexcept;
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