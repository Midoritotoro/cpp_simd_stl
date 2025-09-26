#pragma once 

#include <src/simd_stl/numeric/SimdDivisors.h>

__SIMD_STL_NUMERIC_NAMESPACE_BEGIN

template <arch::CpuFeature _SimdGeneration_>
class BasicSimdImplementation;

template <typename _Element_>
constexpr bool is_epi64_v = sizeof(_Element_) == 8 && std::is_signed_v<_Element_>;

template <typename _Element_>
constexpr bool is_epu64_v = sizeof(_Element_) == 8 && std::is_unsigned_v<_Element_>;

template <typename _Element_>
constexpr bool is_epi32_v = sizeof(_Element_) == 4 && std::is_signed_v<_Element_>;

template <typename _Element_>
constexpr bool is_epu32_v = sizeof(_Element_) == 4 && std::is_unsigned_v<_Element_>;

template <typename _Element_>
constexpr bool is_epi16_v = sizeof(_Element_) == 2 && std::is_signed_v<_Element_>;

template <typename _Element_>
constexpr bool is_epu16_v = sizeof(_Element_) == 2 && std::is_unsigned_v<_Element_>;

template <typename _Element_>
constexpr bool is_epi8_v  = sizeof(_Element_) == 1 && std::is_signed_v<_Element_>;

template <typename _Element_>
constexpr bool is_epu8_v  = sizeof(_Element_) == 1 && std::is_unsigned_v<_Element_>;

template <typename _Element_>
constexpr bool is_pd_v    = sizeof(_Element_) == 8 && type_traits::is_any_of_v<_Element_, double, long double>;

template <typename _Element_>
constexpr bool is_ps_v    = sizeof(_Element_) == 4 && std::is_same_v<_Element_, float>;


// vector operator / : divide each element by divisor

// vector of 4 32-bit signed integers
//static inline Vec4i operator / (Vec4i const a, Divisor_i const d) {
//#if INSTRSET >= 5
//    __m128i t1 = _mm_mul_epi32(a, d.getm());               // 32x32->64 bit signed multiplication of a[0] and a[2]
//    __m128i t2 = _mm_srli_epi64(t1, 32);                   // high dword of result 0 and 2
//    __m128i t3 = _mm_srli_epi64(a, 32);                    // get a[1] and a[3] into position for multiplication
//    __m128i t4 = _mm_mul_epi32(t3, d.getm());              // 32x32->64 bit signed multiplication of a[1] and a[3]
//    __m128i t7 = _mm_blend_epi16(t2, t4, 0xCC);
//    __m128i t8 = _mm_add_epi32(t7, a);                     // add
//    __m128i t9 = _mm_sra_epi32(t8, d.gets1());             // shift right arithmetic
//    __m128i t10 = _mm_srai_epi32(a, 31);                   // sign of a
//    __m128i t11 = _mm_sub_epi32(t10, d.getsign());         // sign of a - sign of d
//    __m128i t12 = _mm_sub_epi32(t9, t11);                  // + 1 if a < 0, -1 if d < 0
//    return        _mm_xor_si128(t12, d.getsign());         // change sign if divisor negative
//#else  // not SSE4.1
//    __m128i t1 = _mm_mul_epu32(a, d.getm());               // 32x32->64 bit unsigned multiplication of a[0] and a[2]
//    __m128i t2 = _mm_srli_epi64(t1, 32);                   // high dword of result 0 and 2
//    __m128i t3 = _mm_srli_epi64(a, 32);                    // get a[1] and a[3] into position for multiplication
//    __m128i t4 = _mm_mul_epu32(t3, d.getm());              // 32x32->64 bit unsigned multiplication of a[1] and a[3]
//    __m128i t5 = _mm_set_epi32(-1, 0, -1, 0);              // mask of dword 1 and 3
//    __m128i t6 = _mm_and_si128(t4, t5);                    // high dword of result 1 and 3
//    __m128i t7 = _mm_or_si128(t2, t6);                     // combine all four results of unsigned high mul into one vector
//    // convert unsigned to signed high multiplication (from: H S Warren: Hacker's delight, 2003, p. 132)
//    __m128i u1 = _mm_srai_epi32(a, 31);                    // sign of a
//    __m128i u2 = _mm_srai_epi32(d.getm(), 31);             // sign of m [ m is always negative, except for abs(d) = 1 ]
//    __m128i u3 = _mm_and_si128(d.getm(), u1);              // m * sign of a
//    __m128i u4 = _mm_and_si128(a, u2);                     // a * sign of m
//    __m128i u5 = _mm_add_epi32(u3, u4);                    // sum of sign corrections
//    __m128i u6 = _mm_sub_epi32(t7, u5);                    // high multiplication result converted to signed
//    __m128i t8 = _mm_add_epi32(u6, a);                     // add a
//    __m128i t9 = _mm_sra_epi32(t8, d.gets1());             // shift right arithmetic
//    __m128i t10 = _mm_sub_epi32(u1, d.getsign());          // sign of a - sign of d
//    __m128i t11 = _mm_sub_epi32(t9, t10);                  // + 1 if a < 0, -1 if d < 0
//    return        _mm_xor_si128(t11, d.getsign());         // change sign if divisor negative
//#endif
//}
//
//// vector of 4 32-bit unsigned integers
//static inline Vec4ui operator / (Vec4ui const a, Divisor_ui const d) {
//    __m128i t1 = _mm_mul_epu32(a, d.getm());               // 32x32->64 bit unsigned multiplication of a[0] and a[2]
//    __m128i t2 = _mm_srli_epi64(t1, 32);                   // high dword of result 0 and 2
//    __m128i t3 = _mm_srli_epi64(a, 32);                    // get a[1] and a[3] into position for multiplication
//    __m128i t4 = _mm_mul_epu32(t3, d.getm());              // 32x32->64 bit unsigned multiplication of a[1] and a[3]
//#if INSTRSET >= 5   // SSE4.1 supported
//    __m128i t7 = _mm_blend_epi16(t2, t4, 0xCC);            // blend two results
//#else
//    __m128i t5 = _mm_set_epi32(-1, 0, -1, 0);              // mask of dword 1 and 3
//    __m128i t6 = _mm_and_si128(t4, t5);                    // high dword of result 1 and 3
//    __m128i t7 = _mm_or_si128(t2, t6);                     // combine all four results into one vector
//#endif
//    __m128i t8 = _mm_sub_epi32(a, t7);                     // subtract
//    __m128i t9 = _mm_srl_epi32(t8, d.gets1());             // shift right logical
//    __m128i t10 = _mm_add_epi32(t7, t9);                   // add
//    return        _mm_srl_epi32(t10, d.gets2());           // shift right logical
//}
//
//// vector of 8 16-bit signed integers
//static inline Vec8s operator / (Vec8s const a, Divisor_s const d) {
//    __m128i t1 = _mm_mulhi_epi16(a, d.getm());             // multiply high signed words
//    __m128i t2 = _mm_add_epi16(t1, a);                     // + a
//    __m128i t3 = _mm_sra_epi16(t2, d.gets1());             // shift right arithmetic
//    __m128i t4 = _mm_srai_epi16(a, 15);                    // sign of a
//    __m128i t5 = _mm_sub_epi16(t4, d.getsign());           // sign of a - sign of d
//    __m128i t6 = _mm_sub_epi16(t3, t5);                    // + 1 if a < 0, -1 if d < 0
//    return        _mm_xor_si128(t6, d.getsign());          // change sign if divisor negative
//}
//
//// vector of 8 16-bit unsigned integers
//static inline Vec8us operator / (Vec8us const a, Divisor_us const d) {
//    __m128i t1 = _mm_mulhi_epu16(a, d.getm());             // multiply high unsigned words
//    __m128i t2 = _mm_sub_epi16(a, t1);                     // subtract
//    __m128i t3 = _mm_srl_epi16(t2, d.gets1());             // shift right logical
//    __m128i t4 = _mm_add_epi16(t1, t3);                    // add
//    return        _mm_srl_epi16(t4, d.gets2());            // shift right logical
//}
//
//
//// vector of 16 8-bit signed integers
//static inline Vec16c operator / (Vec16c const a, Divisor_s const d) {
//    // expand into two Vec8s
//    Vec8s low = extend_low(a) / d;
//    Vec8s high = extend_high(a) / d;
//    return compress(low, high);
//}
//
//// vector of 16 8-bit unsigned integers
//static inline Vec16uc operator / (Vec16uc const a, Divisor_us const d) {
//    // expand into two Vec8s
//    Vec8us low = extend_low(a) / d;
//    Vec8us high = extend_high(a) / d;
//    return compress(low, high);
//}
//
//// vector operator /= : divide
//static inline Vec8s & operator /= (Vec8s & a, Divisor_s const d) {
//    a = a / d;
//    return a;
//}
//
//// vector operator /= : divide
//static inline Vec8us & operator /= (Vec8us & a, Divisor_us const d) {
//    a = a / d;
//    return a;
//}
//
//// vector operator /= : divide
//static inline Vec4i & operator /= (Vec4i & a, Divisor_i const d) {
//    a = a / d;
//    return a;
//}
//
//// vector operator /= : divide
//static inline Vec4ui & operator /= (Vec4ui & a, Divisor_ui const d) {
//    a = a / d;
//    return a;
//}
//
//// vector operator /= : divide
//static inline Vec16c & operator /= (Vec16c & a, Divisor_s const d) {
//    a = a / d;
//    return a;
//}
//
//// vector operator /= : divide
//static inline Vec16uc & operator /= (Vec16uc & a, Divisor_us const d) {
//    a = a / d;
//    return a;
//}
//
///*****************************************************************************
//*
//*          Integer division 2: divisor is a compile-time constant
//*
//*****************************************************************************/
//
//// Divide Vec4i by compile-time constant
//template <int32_t d>
//static inline Vec4i divide_by_i(Vec4i const x) {
//    static_assert(d != 0, "Integer division by zero");     // Error message if dividing by zero
//    if constexpr (d == 1) return  x;
//    if constexpr (d == -1) return -x;
//    if constexpr (uint32_t(d) == 0x80000000u) return Vec4i(x == Vec4i(0x80000000)) & 1; // prevent overflow when changing sign
//    constexpr uint32_t d1 = d > 0 ? uint32_t(d) : uint32_t(-d);// compile-time abs(d). (force GCC compiler to treat d as 32 bits, not 64 bits)
//    if constexpr ((d1 & (d1 - 1)) == 0) {
//        // d1 is a power of 2. use shift
//        constexpr int k = bit_scan_reverse_const(d1);
//        __m128i sign;
//        if constexpr (k > 1) sign = _mm_srai_epi32(x, k-1); else sign = x;// k copies of sign bit
//        __m128i bias = _mm_srli_epi32(sign, 32 - k);       // bias = x >= 0 ? 0 : k-1
//        __m128i xpbias = _mm_add_epi32(x, bias);           // x + bias
//        __m128i q = _mm_srai_epi32(xpbias, k);             // (x + bias) >> k
//        if (d > 0)      return q;                          // d > 0: return  q
//        return _mm_sub_epi32(_mm_setzero_si128(), q);      // d < 0: return -q
//    }
//    // general case
//    constexpr int32_t sh = bit_scan_reverse_const(uint32_t(d1) - 1); // ceil(log2(d1)) - 1. (d1 < 2 handled by power of 2 case)
//    constexpr int32_t mult = int(1 + (uint64_t(1) << (32 + sh)) / uint32_t(d1) - (int64_t(1) << 32));   // multiplier
//    const Divisor_i div(mult, sh, d < 0 ? -1 : 0);
//    return x / div;
//}
//
//// define Vec4i a / const_int(d)
//template <int32_t d>
//static inline Vec4i operator / (Vec4i const a, Const_int_t<d>) {
//    return divide_by_i<d>(a);
//}
//
//// define Vec4i a / const_uint(d)
//template <uint32_t d>
//static inline Vec4i operator / (Vec4i const a, Const_uint_t<d>) {
//    static_assert(d < 0x80000000u, "Dividing signed by overflowing unsigned");
//    return divide_by_i<int32_t(d)>(a);                     // signed divide
//}
//
//// vector operator /= : divide
//template <int32_t d>
//static inline Vec4i & operator /= (Vec4i & a, Const_int_t<d> b) {
//    a = a / b;
//    return a;
//}
//
//// vector operator /= : divide
//template <uint32_t d>
//static inline Vec4i & operator /= (Vec4i & a, Const_uint_t<d> b) {
//    a = a / b;
//    return a;
//}
//
//
//// Divide Vec4ui by compile-time constant
//template <uint32_t d>
//static inline Vec4ui divide_by_ui(Vec4ui const x) {
//    static_assert(d != 0, "Integer division by zero");     // Error message if dividing by zero
//    if constexpr (d == 1) return x;                        // divide by 1
//    constexpr int b = bit_scan_reverse_const(d);           // floor(log2(d))
//    if constexpr ((uint32_t(d) & (uint32_t(d) - 1)) == 0) {
//        // d is a power of 2. use shift
//        return    _mm_srli_epi32(x, b);                    // x >> b
//    }
//    // general case (d > 2)
//    constexpr uint32_t mult = uint32_t((uint64_t(1) << (b+32)) / d); // multiplier = 2^(32+b) / d
//    constexpr uint64_t rem = (uint64_t(1) << (b+32)) - uint64_t(d)*mult; // remainder 2^(32+b) % d
//    constexpr bool round_down = (2 * rem < d);             // check if fraction is less than 0.5
//    constexpr uint32_t mult1 = round_down ? mult : mult + 1;
//    // do 32*32->64 bit unsigned multiplication and get high part of result
//#if INSTRSET >= 10
//    const __m128i multv = _mm_maskz_set1_epi32(0x05, mult1);// zero-extend mult and broadcast
//#else
//    const __m128i multv = Vec2uq(uint64_t(mult1));         // zero-extend mult and broadcast
//#endif
//    __m128i t1 = _mm_mul_epu32(x, multv);                  // 32x32->64 bit unsigned multiplication of x[0] and x[2]
//    if constexpr (round_down) {
//        t1 = _mm_add_epi64(t1, multv);                     // compensate for rounding error. (x+1)*m replaced by x*m+m to avoid overflow
//    }
//    __m128i t2 = _mm_srli_epi64(t1, 32);                   // high dword of result 0 and 2
//    __m128i t3 = _mm_srli_epi64(x, 32);                    // get x[1] and x[3] into position for multiplication
//    __m128i t4 = _mm_mul_epu32(t3, multv);                 // 32x32->64 bit unsigned multiplication of x[1] and x[3]
//    if constexpr (round_down) {
//        t4 = _mm_add_epi64(t4, multv);                     // compensate for rounding error. (x+1)*m replaced by x*m+m to avoid overflow
//    }
//#if INSTRSET >= 5   // SSE4.1 supported
//    __m128i t7 = _mm_blend_epi16(t2, t4, 0xCC);            // blend two results
//#else
//    __m128i t5 = _mm_set_epi32(-1, 0, -1, 0);              // mask of dword 1 and 3
//    __m128i t6 = _mm_and_si128(t4, t5);                    // high dword of result 1 and 3
//    __m128i t7 = _mm_or_si128(t2, t6);                     // combine all four results into one vector
//#endif
//    Vec4ui q = _mm_srli_epi32(t7, b);                      // shift right by b
//    return q;                                              // no overflow possible
//}
//
//// define Vec4ui a / const_uint(d)
//template <uint32_t d>
//static inline Vec4ui operator / (Vec4ui const a, Const_uint_t<d>) {
//    return divide_by_ui<d>(a);
//}
//
//// define Vec4ui a / const_int(d)
//template <int32_t d>
//static inline Vec4ui operator / (Vec4ui const a, Const_int_t<d>) {
//    static_assert(d < 0x80000000u, "Dividing unsigned integer by negative value is ambiguous");
//    return divide_by_ui<d>(a);                             // unsigned divide
//}
//
//// vector operator /= : divide
//template <uint32_t d>
//static inline Vec4ui & operator /= (Vec4ui & a, Const_uint_t<d> b) {
//    a = a / b;
//    return a;
//}
//
//// vector operator /= : divide
//template <int32_t d>
//static inline Vec4ui & operator /= (Vec4ui & a, Const_int_t<d> b) {
//    a = a / b;
//    return a;
//}


/* 
Каждая специализация реализует: 

1.  template <typename _DesiredType_, typename _VectorType_>
    _VectorType_ unaryMinus(_VectorType_ vector) noexcept

2.  template <typename _DesiredType_, typename _VectorType_>
    _VectorType_ shuffle(
        _VectorType_                                                                                vector,
        type_traits::__deduce_simd_shuffle_mask_type<sizeof(_VectorType_) / sizeof(_DesiredType_)>  shuffleMask) noexcept


3.  template <typename _DesiredType_, typename _VectorType_>
    _VectorType_ shuffle(
        _VectorType_                                                                                vector,
        _VectorType_                                                                                vectorSecond,
        type_traits::__deduce_simd_shuffle_mask_type<sizeof(_VectorType_) / sizeof(_DesiredType_)>  shuffleMask) noexcept

    Перестановка элементов в векторе 'vector' по битовой маске shuffleMask, тип которой определяется следующим образом
    в завимости от размера вектора в элементах(sizeof(_VectorType_) / sizeof(_DesiredType_)):
        vectorSize == 2     -> uint8    (используется два бита)
        vectorSize == 4     -> uint8
        vectorSize == 8     -> uint32   (используется 24 бита)
        vectorSize == 16    -> uint64 

4.  template <typename _VectorType_, typename _DesiredType_>
    _VectorType_ loadUnaligned(const _DesiredType_* where) noexcept;


5.  template <typename _VectorType_, typename _DesiredType_>
    _VectorType_ loadAligned(const _DesiredType_* where) noexcept;


6.  template <typename _DesiredType_, typename _VectorType_>
    void storeUnaligned(
        _DesiredType_*      where,
        const _VectorType_  vector) noexcept;

7.  template <typename _DesiredType_, typename _VectorType_>
    void storeAligned(
        _DesiredType_*      where,
        const _VectorType_  vector) noexcept;

8.  template <typename _DesiredType_, typename _VectorType_>
    void maskStoreUnaligned(
        _DesiredType_*                                      where,
        const type_traits::__deduce_simd_shuffle_mask_type<
            sizeof(_VectorType_) / sizeof(_DesiredType_)>   mask,
        const _VectorType_                                  vector) noexcept;

9.  template <typename _DesiredType_, typename _VectorType_>
    void maskStoreAligned(
        _DesiredType_*                                      where,
        const type_traits::__deduce_simd_shuffle_mask_type<
            sizeof(_VectorType_) / sizeof(_DesiredType_)>   mask,
        const _VectorType_                                  vector);

10. template <typename _DesiredType_, typename _VectorType_>
    _VectorType_ maskLoadUnaligned(
        const _DesiredType_*                                where,
        const type_traits::__deduce_simd_shuffle_mask_type<
            sizeof(_VectorType_) / sizeof(_DesiredType_)>   mask,
        const _VectorType_                                  vector) noexcept;

11. template <typename _DesiredType_, typename _VectorType_>
    void maskLoadAligned(
        const _DesiredType_*                                where,
        const type_traits::__deduce_simd_shuffle_mask_type<
            sizeof(_VectorType_) / sizeof(_DesiredType_)>   mask,
        _VectorType_                                        vector) noexcept;

12. template <typename _FromVector_, typename _ToVector_, bool _SafeCast_>
    _ToVector_ cast(_FromVector_ from) noexcept;

13. template <typename _DesiredVectorElementType_, typename _VectorType_>
    int32 maskFromVector(_VectorType_ vector) noexcept;

14. template <typename _VectorType_>
    _VectorType_ decrement(_VectorType_ vector) noexcept;

15. template <typename _VectorType_>
    _VectorType_ increment(_VectorType_ vector) noexcept;

16. template <typename _DesiredType_, typename _VectorType_>
    _DesiredType_ extract(
        _VectorType_    vector,
        const uint8     where) noexcept;

17. template <typename _VectorType_>
    _VectorType_ constructZero() noexcept;

18. template <typename _VectorType_, typename _DesiredType_>
    _VectorType_ broadcast(_DesiredType_ value) noexcept;

19. template <typename _DesiredType_, typename _VectorType_>
    _VectorType_ add(
        _VectorType_ left,
        _VectorType_ right) noexcept;

20. template <typename _DesiredType_, typename _VectorType_>
    _VectorType_ sub(
        _VectorType_ left,
        _VectorType_ right) noexcept;

21. template <typename _DesiredType_, typename _VectorType_>
    _VectorType_ mul(
        _VectorType_ left,
        _VectorType_ right) noexcept;

22. template <typename _DesiredType_, typename _VectorType_>
    _VectorType_ div(
        _VectorType_ left,
        _VectorType_ right);

23. template <typename _VectorType_>
    _VectorType_ bitwiseNot(_VectorType_ vector) noexcept;
    

24. template <typename _VectorType_>
    _VectorType_ bitwiseXor(
        const _VectorType_& left,
        const _VectorType_& right) noexcept;

25. template <typename _VectorType_>
    _VectorType_ bitwiseAnd(
        const _VectorType_& left,
        const _VectorType_& right) noexcept;

26. template <typename _VectorType_>
    _VectorType_ bitwiseOr(
        const _VectorType_& left,
        const _VectorType_& right) noexcept;

27. template <typename _MaskType_, typename _DesiredVectorElementType_,  typename _VectorType_>
    _VectorType_ maskToVector(_MaskType_ mask) noexcept;
*/

template <>
class BasicSimdImplementation<arch::CpuFeature::SSE2> {
public:
    template <
        typename        _DesiredType_,
        _DesiredType_   _Divisor_,
        typename        _VectorType_>
    static simd_stl_constexpr_cxx20 simd_stl_always_inline _VectorType_ divideByConst(_VectorType_ dividendVector) noexcept
    {
        return divideByConstHelper<_DesiredType_, _Divisor_, _VectorType_>(dividendVector);
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_constexpr_cxx20 simd_stl_always_inline _VectorType_ unaryMinus(_VectorType_ vector) noexcept {
        // 0x80000000 == 0b10000000000000000000000000000000
        if constexpr (is_ps_v<_DesiredType_>)
            return _mm_xor_ps(vector, cast<__m128i, __m128>(_mm_set1_epi32(0x80000000)));
        else if constexpr (is_pd_v<_DesiredType_>)
            return _mm_xor_pd(vector, cast<__m128i, __m128d>(_mm_setr_epi32(0, 0x80000000, 0, 0x80000000)));
        else
            return sub<_DesiredType_>(constructZero<_VectorType_>(), vector);
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_constexpr_cxx20 simd_stl_always_inline _VectorType_ shuffle(
        _VectorType_                                        vector,
        type_traits::__deduce_simd_shuffle_mask_type<
            sizeof(_VectorType_) / sizeof(_DesiredType_)>   shuffleMask) noexcept
    {
        return shuffle<_DesiredType_>(vector, vector, shuffleMask);
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_constexpr_cxx20 simd_stl_always_inline _VectorType_ shuffle(
        _VectorType_                                            vector,
        _VectorType_                                            secondVector,
        type_traits::__deduce_simd_shuffle_mask_type<
            sizeof(_VectorType_) / sizeof(_DesiredType_)>       shuffleMask) noexcept
    {
        if      constexpr (is_epi64_v<_DesiredType_> || is_epu64_v<_DesiredType_> || is_pd_v<_DesiredType_>)
            return cast<__m128d, _VectorType_>(_mm_shuffle_pd(
                cast<_VectorType_, __m128d>(vector),
                cast<_VectorType_, __m128d>(secondVector),
                shuffleMask)
            );
        else if constexpr (is_epi32_v<_DesiredType_> || is_epu32_v<_DesiredType_> || is_ps_v<_DesiredType_>)
            return cast<__m128i, _VectorType_>(_mm_shuffle_epi32(
                cast<_VectorType_, __m128i>(vector),
                shuffleMask)
            );
        else if constexpr (is_epi16_v<_DesiredType_> || is_epu16_v<_DesiredType_>) {

        }
        else if constexpr (is_epi8_v<_DesiredType_> || is_epu8_v<_DesiredType_>) {
            
        }
    }

    template <
        typename _VectorType_,
        typename _DesiredType_>
    static simd_stl_constexpr_cxx20 simd_stl_always_inline _VectorType_ loadUnaligned(const _DesiredType_* where) noexcept {
        if      constexpr (std::is_same_v<_VectorType_, __m128i>)
            return _mm_loadu_si128(reinterpret_cast<const __m128i*>(where));
        else if constexpr (std::is_same_v<_VectorType_, __m128d>)
            return _mm_loadu_pd(reinterpret_cast<const double*>(where));
        else if constexpr (std::is_same_v<_VectorType_, __m128>)
            return _mm_loadu_ps(reinterpret_cast<const float*>(where));
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_constexpr_cxx20 simd_stl_always_inline _VectorType_ loadAligned(const _DesiredType_* where) noexcept {
        if      constexpr (std::is_same_v<_VectorType_, __m128i>)
            return _mm_load_si128(reinterpret_cast<const __m128i*>(where));
        else if constexpr (std::is_same_v<_VectorType_, __m128d>)
            return _mm_load_pd(reinterpret_cast<const double*>(where));
        else if constexpr (std::is_same_v<_VectorType_, __m128>)
            return _mm_load_ps(reinterpret_cast<const float*>(where));
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_constexpr_cxx20 simd_stl_always_inline void storeUnaligned(
        _DesiredType_*          where,
        const _VectorType_      vector) noexcept
    {
        if      constexpr (std::is_same_v<_VectorType_, __m128i>)
            return _mm_storeu_si128(reinterpret_cast<__m128i*>(where), vector);
        else if constexpr (std::is_same_v<_VectorType_, __m128d>)
            return _mm_storeu_pd(reinterpret_cast<double*>(where), vector);
        else if constexpr (std::is_same_v<_VectorType_, __m128>)
            return _mm_storeu_ps(reinterpret_cast<float*>(where), vector);
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_constexpr_cxx20 simd_stl_always_inline void storeAligned(
        _DesiredType_*          where,
        const _VectorType_      vector) noexcept
    {
        if      constexpr (std::is_same_v<_VectorType_, __m128i>)
            return _mm_store_si128(reinterpret_cast<__m128i*>(where), vector);
        else if constexpr (std::is_same_v<_VectorType_, __m128d>)
            return _mm_store_pd(reinterpret_cast<double*>(where), vector);
        else if constexpr (std::is_same_v<_VectorType_, __m128>)
            return _mm_store_ps(reinterpret_cast<float*>(where), vector);
    }


    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_constexpr_cxx20 simd_stl_always_inline void maskStoreUnaligned(
        _DesiredType_*                                      where,
        const type_traits::__deduce_simd_shuffle_mask_type<
            sizeof(_VectorType_) / sizeof(_DesiredType_)>   mask,
        const _VectorType_                                  vector) noexcept
    {
        storeUnaligned(where, shuffle<_DesiredType_>(loadUnaligned<_VectorType_>(where), vector, mask));
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_constexpr_cxx20 simd_stl_always_inline void maskStoreAligned(
        _DesiredType_*                                      where,
        const type_traits::__deduce_simd_shuffle_mask_type<
            sizeof(_VectorType_) / sizeof(_DesiredType_)>   mask,
        const _VectorType_                                  vector) noexcept
    {
        storeAligned(where, shuffle<_DesiredType_>(loadAligned<_VectorType_>(where), vector, mask));
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_constexpr_cxx20 simd_stl_always_inline _VectorType_ maskLoadUnaligned(
        const _DesiredType_*                                where,
        const type_traits::__deduce_simd_shuffle_mask_type<
            sizeof(_VectorType_) / sizeof(_DesiredType_)>   mask,
        const _VectorType_                                  vector) noexcept
    {
        return shuffle<_DesiredType_>(loadUnaligned<_VectorType_>(where), vector, mask);
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_constexpr_cxx20 simd_stl_always_inline void maskLoadAligned(
        const _DesiredType_*                                where,
        const type_traits::__deduce_simd_shuffle_mask_type<
            sizeof(_VectorType_) / sizeof(_DesiredType_)>   mask,
        _VectorType_                                        vector) noexcept
    {
        return shuffle<_DesiredType_>(loadAligned<_VectorType_>(where), vector, mask);
    }

    template <
        typename    _FromVector_,
        typename    _ToVector_,
        bool        _SafeCast_ = false>
    static simd_stl_constexpr_cxx20 simd_stl_always_inline _ToVector_ cast(_FromVector_ from) noexcept {
        if constexpr (std::is_same_v<_FromVector_, _ToVector_>)
            return from;

        else if constexpr (std::is_same_v<_FromVector_, __m128> && std::is_same_v<_ToVector_, __m128i>)
            return _mm_castps_si128(from);
        else if constexpr (std::is_same_v<_FromVector_, __m128> && std::is_same_v<_ToVector_, __m128d>)
            return _mm_castps_pd(from);

        else if constexpr (std::is_same_v<_FromVector_, __m128d> && std::is_same_v<_ToVector_, __m128>)
            return _mm_castpd_ps(from);
        else if constexpr (std::is_same_v<_FromVector_, __m128d> && std::is_same_v<_ToVector_, __m128i>)
            return _mm_castpd_si128(from);

        else if constexpr (std::is_same_v<_FromVector_, __m128i> && std::is_same_v<_ToVector_, __m128>)
            return _mm_castsi128_ps(from);
        else if constexpr (std::is_same_v<_FromVector_, __m128i> && std::is_same_v<_ToVector_, __m128d>)
            return _mm_castsi128_pd(from);
    }

    template <typename _VectorType_> 
    static simd_stl_constexpr_cxx20 simd_stl_always_inline int32 convertToMask(_VectorType_ vector) noexcept {
        return 0;
    }

    template <typename _VectorType_>
    static simd_stl_constexpr_cxx20 simd_stl_always_inline _VectorType_ decrement(_VectorType_ vector) noexcept {
        return sub(vector, broadcast(1));
    }

    template <typename _VectorType_>
    static simd_stl_constexpr_cxx20 simd_stl_always_inline _VectorType_ increment(_VectorType_ vector) noexcept {
        return add(vector, broadcast(1));
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_constexpr_cxx20 simd_stl_always_inline _DesiredType_ extract(
        _VectorType_    vector,
        const uint8     where) noexcept
    {
        
    }

    template <typename _VectorType_>
    static simd_stl_constexpr_cxx20 simd_stl_always_inline _VectorType_ constructZero() noexcept {
        if      constexpr (std::is_same_v<_VectorType_, __m128d>)
            return _mm_setzero_pd();
        else if constexpr (std::is_same_v<_VectorType_, __m128i>)
            return _mm_setzero_si128();
        else if constexpr (std::is_same_v<_VectorType_, __m128>)
            return _mm_setzero_ps();
    }

    template <
        typename _VectorType_,
        typename _DesiredType_>
    static simd_stl_constexpr_cxx20 simd_stl_always_inline _VectorType_ broadcast(_DesiredType_ value) noexcept {
        if constexpr (is_epi64_v<_DesiredType_> || is_epu64_v<_DesiredType_>)
            return cast<__m128i, _VectorType_>(_mm_set1_epi64x(value));
        else if constexpr (is_epi32_v<_DesiredType_> || is_epu32_v<_DesiredType_>)
            return cast<__m128i, _VectorType_>(_mm_set1_epi32(value));
        else if constexpr (is_epi16_v<_DesiredType_> || is_epu16_v<_DesiredType_>)
            return cast<__m128i, _VectorType_>(_mm_set1_epi16(value));
        else if constexpr (is_epi8_v<_DesiredType_> || is_epu8_v<_DesiredType_>)
            return cast<__m128i, _VectorType_>(_mm_set1_epi8(value));
        else if constexpr (is_ps_v<_DesiredType_>)
            return cast<__m128, _VectorType_>(_mm_set1_ps(value));
        else if constexpr (is_pd_v<_DesiredType_>)
            return cast<__m128d, _VectorType_>(_mm_set1_pd(value));
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_constexpr_cxx20 simd_stl_always_inline _VectorType_ add(
        _VectorType_ left,
        _VectorType_ right) noexcept
    {
        if constexpr (is_epi64_v<_DesiredType_> || is_epu64_v<_DesiredType_>)
            return cast<__m128i, _VectorType_>(_mm_add_epi64(
                cast<_VectorType_, __m128i>(left),
                cast<_VectorType_, __m128i>(right)));
        else if constexpr (is_epi32_v<_DesiredType_> || is_epu32_v<_DesiredType_>)
            return cast<__m128i, _VectorType_>(_mm_add_epi32(
                cast<_VectorType_, __m128i>(left),
                cast<_VectorType_, __m128i>(right)));
        else if constexpr (is_epi16_v<_DesiredType_> || is_epu16_v<_DesiredType_>)
            return cast<__m128i, _VectorType_>(_mm_add_epi16(
                cast<_VectorType_, __m128i>(left),
                cast<_VectorType_, __m128i>(right)));
        else if constexpr (is_epi8_v<_DesiredType_> || is_epu8_v<_DesiredType_>)
            return cast<__m128i, _VectorType_>(_mm_add_epi8(
                cast<_VectorType_, __m128i>(left),
                cast<_VectorType_, __m128i>(right)));
        else if constexpr (is_ps_v<_DesiredType_>)
            return cast<__m128, _VectorType_>(_mm_add_ps(
                cast<_VectorType_, __m128>(left),
                cast<_VectorType_, __m128>(right)));
        else if constexpr (is_pd_v<_DesiredType_>)
            return cast<__m128d, _VectorType_>(_mm_add_pd(
                cast<_VectorType_, __m128d>(left),
                cast<_VectorType_, __m128d>(right)));
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_constexpr_cxx20 simd_stl_always_inline _VectorType_ sub(
        _VectorType_ left,
        _VectorType_ right) noexcept
    {
        if constexpr (is_epi64_v<_DesiredType_> || is_epu64_v<_DesiredType_>)
            return cast<__m128i, _VectorType_>(_mm_sub_epi64(
                cast<_VectorType_, __m128i>(left),
                cast<_VectorType_, __m128i>(right)));
        else if constexpr (is_epi32_v<_DesiredType_> || is_epu32_v<_DesiredType_>)
            return cast<__m128i, _VectorType_>(_mm_sub_epi32(
                cast<_VectorType_, __m128i>(left),
                cast<_VectorType_, __m128i>(right)));
        else if constexpr (is_epi16_v<_DesiredType_> || is_epu16_v<_DesiredType_>)
            return cast<__m128i, _VectorType_>(_mm_sub_epi16(
                cast<_VectorType_, __m128i>(left),
                cast<_VectorType_, __m128i>(right)));
        else if constexpr (is_epi8_v<_DesiredType_> || is_epu8_v<_DesiredType_>)
            return cast<__m128i, _VectorType_>(_mm_sub_epi8(
                cast<_VectorType_, __m128i>(left),
                cast<_VectorType_, __m128i>(right)));
        else if constexpr (is_ps_v<_DesiredType_>)
            return cast<__m128, _VectorType_>(_mm_sub_ps(
                cast<_VectorType_, __m128>(left),
                cast<_VectorType_, __m128>(right)));
        else if constexpr (is_pd_v<_DesiredType_>)
            return cast<__m128d, _VectorType_>(_mm_sub_pd(
                cast<_VectorType_, __m128d>(left),
                cast<_VectorType_, __m128d>(right)));
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_constexpr_cxx20 simd_stl_always_inline _VectorType_ mul(
        _VectorType_ left,
        _VectorType_ right) noexcept
    {
        // ? 
       /* if      constexpr (is_epi64_v<_DesiredType_>)
            return cast<__m128i, _VectorType_>(_mm_mul_epi64(
                cast<_VectorType_, __m128i>(left),
                cast<_VectorType_, __m128i>(right)));
        else if constexpr (is_epu64_v<_DesiredType_>)
            return cast<__m128i, _VectorType_>(_mm_mul_epu64(
                cast<_VectorType_, __m128i>(left),
                cast<_VectorType_, __m128i>(right)));
        else if constexpr (is_epi32_v<_DesiredType_>)
            return cast<__m128i, _VectorType_>(_mm_mul_epi32(
                cast<_VectorType_, __m128i>(left),
                cast<_VectorType_, __m128i>(right)));
        else */if constexpr (is_epu32_v<_DesiredType_>)
            return cast<__m128i, _VectorType_>(_mm_mul_epu32(
                cast<_VectorType_, __m128i>(left),
                cast<_VectorType_, __m128i>(right)));
        //else if constexpr (is_epi16_v<_DesiredType_>)
        //    return cast<__m128i, _VectorType_>(_mm_mul_epi16(
        //        cast<_VectorType_, __m128i>(left),
        //        cast<_VectorType_, __m128i>(right)));
        //else if constexpr (is_epu16_v<_DesiredType_>)
        //    return cast<__m128i, _VectorType_>(_mm_mul_epu16(
        //        cast<_VectorType_, __m128i>(left),
        //        cast<_VectorType_, __m128i>(right)));
        //else if constexpr (is_epi8_v<_DesiredType_>)
        //    return cast<__m128i, _VectorType_>(_mm_mul_epi8(
        //        cast<_VectorType_, __m128i>(left),
        //        cast<_VectorType_, __m128i>(right)));
        //else if constexpr (is_epu8_v<_DesiredType_>)
        //    return cast<__m128i, _VectorType_>(_mm_mul_epi8(
        //        cast<_VectorType_, __m128i>(left),
        //        cast<_VectorType_, __m128i>(right)));
        //else if constexpr (is_ps_v<_DesiredType_>)
        //    return _mm_mul_ps(left, right);
        else if constexpr (is_pd_v<_DesiredType_>)
            return _mm_mul_pd(left, right);

        return broadcast<_VectorType_>(0);
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_constexpr_cxx20 simd_stl_always_inline _VectorType_ div(
        _VectorType_ left,
        _VectorType_ right) noexcept
    {
        if constexpr (is_pd_v<_DesiredType_>)
            return cast<__m128d, _VectorType_>(_mm_div_pd(
                cast<_VectorType_, __m128d>(left),
                cast<_VectorType_, __m128d>(right)));
        else if constexpr (is_ps_v<_DesiredType_>)
            return cast<__m128, _VectorType_>(_mm_div_ps(
                cast<_VectorType_, __m128>(left),
                cast<_VectorType_, __m128>(right)));
        else if constexpr (is_epi32_v<_DesiredType_>) {
        
        }
    }

    template <typename _VectorType_>
    static simd_stl_constexpr_cxx20 simd_stl_always_inline _VectorType_ bitwiseNot(_VectorType_ vector) noexcept {
        if      constexpr (std::is_same_v<_VectorType_, __m128d>)
            return _mm_xor_pd(vector, _mm_cmpeq_pd(vector, vector));
        else if constexpr (std::is_same_v<_VectorType_, __m128i>)
            return _mm_xor_si128(vector, _mm_cmpeq_epi32(vector, vector));
        else if constexpr (std::is_same_v<_VectorType_, __m128>)
            return _mm_xor_ps(vector, _mm_cmpeq_ps(vector, vector));
    }

    template <typename _VectorType_>
    static simd_stl_constexpr_cxx20 simd_stl_always_inline _VectorType_ bitwiseXor(
        const _VectorType_& left,
        const _VectorType_& right) noexcept
    {
        if      constexpr (std::is_same_v<_VectorType_, __m128d>)
            return _mm_xor_pd(left, right);
        else if constexpr (std::is_same_v<_VectorType_, __m128i>)
            return _mm_xor_si128(left, right);
        else if constexpr (std::is_same_v<_VectorType_, __m128>)
            return _mm_xor_ps(left, right);
    }

    template <typename _VectorType_>
    static simd_stl_constexpr_cxx20 simd_stl_always_inline _VectorType_ bitwiseAnd(
        const _VectorType_& left,
        const _VectorType_& right) noexcept
    {
        if      constexpr (std::is_same_v<_VectorType_, __m128d>)
            return _mm_and_pd(left, right);
        else if constexpr (std::is_same_v<_VectorType_, __m128i>)
            return _mm_and_si128(left, right);
        else if constexpr (std::is_same_v<_VectorType_, __m128>)
            return _mm_and_ps(left, right);
    }

    template <typename _VectorType_>
    static simd_stl_constexpr_cxx20 simd_stl_always_inline _VectorType_ bitwiseOr(
        const _VectorType_& left,
        const _VectorType_& right) noexcept
    {
        if      constexpr (std::is_same_v<_VectorType_, __m128d>)
            return _mm_or_pd(left, right);
        else if constexpr (std::is_same_v<_VectorType_, __m128i>)
            return _mm_or_si128(left, right);
        else if constexpr (std::is_same_v<_VectorType_, __m128>)
            return _mm_or_ps(left, right);
    }
private: 
    template <
        typename        _DesiredType_,
        _DesiredType_   _Divisor_,
        typename        _VectorType_>
    static simd_stl_constexpr_cxx20 simd_stl_always_inline _VectorType_ divideByConstHelper(_VectorType_ dividendVector) noexcept
    {
        static_assert(_Divisor_ != 0, "Integer division by zero");

        if constexpr (_Divisor_ == 1)
            return dividendVector;

        if constexpr (is_epu32_v<_DesiredType_>) {
            const auto dividendAsInt32 = cast<_VectorType_, __m128i>(dividendVector);
            constexpr auto trailingZerosInDivisor = math::CountTrailingZeroBits(_Divisor_);

            if constexpr ((uint32(_Divisor_) & (uint32(_Divisor_) - 1)) == 0)
                return _mm_srli_epi32(dividendAsInt32, trailingZerosInDivisor);

            constexpr uint32 magicMultiplier = uint32((uint64(1) << (trailingZerosInDivisor + 32)) / _Divisor_);
            constexpr uint64 magicRemainder = (uint64(1) << (trailingZerosInDivisor + 32)) - uint64(_Divisor_) * magicMultiplier;

            constexpr bool useRoundDown = (2 * magicRemainder < _Divisor_);
            constexpr uint32 adjustedMultiplier = useRoundDown ? magicMultiplier : magicMultiplier + 1;

            const auto multiplierBroadcasted = broadcast<_VectorType_>(uint64(adjustedMultiplier));

            auto lowProduct = _mm_mul_epu32(dividendAsInt32, multiplierBroadcasted);    // Умножаем элементы [0] и [2] на multiplier

            if constexpr (useRoundDown)
                lowProduct = _mm_add_epi64(lowProduct, multiplierBroadcasted);

            auto lowProductShifted = _mm_srli_epi64(lowProduct, 32);                   // Получаем старшие 32 бита результата умножения
            auto highParts = _mm_srli_epi64(dividendAsInt32, 32);              // Получаем элементы [1] и [3] из исходного вектора
            auto highProduct = _mm_mul_epu32(highParts, multiplierBroadcasted);  // Умножаем элементы [1] и [3] на multiplier

            if constexpr (useRoundDown)
                highProduct = _mm_add_epi64(highProduct, multiplierBroadcasted);

            auto low32BitMask = _mm_set_epi32(-1, 0, -1, 0);
            auto highProductMasked = _mm_and_si128(highProduct, low32BitMask);

            auto combinedProduct = _mm_or_si128(lowProductShifted, highProductMasked);

            return cast<__m128i, _VectorType_>(_mm_srli_epi32(combinedProduct, trailingZerosInDivisor));
        }
        else if constexpr (is_epi32_v<_DesiredType_>) {
            if constexpr (_Divisor_ == -1)
                return unaryMinus<_DesiredType_>(dividendVector);

            constexpr uint32 absoluteDivisor = _Divisor_ > 0 ? uint32(_Divisor_) : uint32(-_Divisor_);

            if constexpr ((absoluteDivisor & (absoluteDivisor - 1)) == 0) {
                constexpr auto shiftAmount = math::CountTrailingZeroBits(absoluteDivisor);
                __m128i signBits;

                if constexpr (shiftAmount > 1)
                    signBits = _mm_srai_epi32(dividendVector, shiftAmount - 1);
                else
                    signBits = dividendVector;

                auto roundingBias = _mm_srli_epi32(signBits, 32 - shiftAmount);
                auto biasedDividend = _mm_add_epi32(dividendVector, roundingBias);

                auto quotient = _mm_srai_epi32(biasedDividend, shiftAmount);

                if constexpr (_Divisor_ > 0)
                    return quotient;

                return _mm_sub_epi32(_mm_setzero_si128(), quotient);
            }

            constexpr int32 shiftForMagic = math::CountTrailingZeroBits(uint32_t(absoluteDivisor) - 1);
            constexpr int32 magicMultiplier = int32(1 + ((uint64(1) << (32 + shiftForMagic))
                / uint32(absoluteDivisor)) - (int64(1) << 32));

            SimdDivisor<arch::CpuFeature::SSE2, int32> divisorParams(
                magicMultiplier, shiftForMagic, _Divisor_ < 0 ? -1 : 0);

            const auto productLowEven = _mm_mul_epu32(dividendVector, divisorParams.getMultiplier()); // dividendVector[0], dividendVector[2]
            const auto productHighEven = _mm_srli_epi64(productLowEven, 32);

            const auto shiftedDividendOdd = _mm_srli_epi64(dividendVector, 32); // dividendVector[1], dividendVector[3]
            const auto productLowOdd = _mm_mul_epu32(shiftedDividendOdd, divisorParams.getMultiplier());

            const auto oddMask = _mm_set_epi32(-1, 0, -1, 0);
            const auto productHighOdd = _mm_and_si128(productLowOdd, oddMask);

            const auto combinedHighProduct = _mm_or_si128(productHighEven, productHighOdd);

            const auto dividendSignMask = _mm_srai_epi32(dividendVector, 31);
            const auto multiplierSignMask = _mm_srai_epi32(divisorParams.getMultiplier(), 31);

            const auto correctionFromMultiplier = _mm_and_si128(divisorParams.getMultiplier(), dividendSignMask);
            const auto correctionFromDividend = _mm_and_si128(dividendVector, multiplierSignMask);

            const auto totalCorrection = _mm_add_epi32(correctionFromMultiplier, correctionFromDividend);
            const auto signedProduct = _mm_sub_epi32(combinedHighProduct, totalCorrection);

            const auto adjustedSum = _mm_add_epi32(signedProduct, dividendVector);
            const auto shiftedQuotient = _mm_sra_epi32(adjustedSum, divisorParams.getFirstShiftCount());

            const auto signDifference = _mm_sub_epi32(dividendSignMask, divisorParams.getDivisorSign());
            const auto correctedQuotient = _mm_sub_epi32(shiftedQuotient, signDifference);

            return _mm_xor_si128(correctedQuotient, divisorParams.getDivisorSign());
        }
        else if constexpr (is_epi16_v<_DesiredType_>) {
            if constexpr (_Divisor_ == -1)
                return unaryMinus<_DesiredType_>(dividendVector);

            constexpr uint32 absoluteDivisor = _Divisor_ > 0 ? uint32_t(_Divisor_) : uint32_t(-_Divisor_);

            if constexpr ((absoluteDivisor & (absoluteDivisor - 1)) == 0) {
                // Делитель — степень двойки
                constexpr auto shiftAmount = math::CountTrailingZeroBits(absoluteDivisor);
                __m128i signBits;

                if constexpr (shiftAmount > 1)
                    signBits = _mm_srai_epi32(dividendVector, shiftAmount - 1);
                else
                    signBits = dividendVector;

                const auto roundingBias = _mm_srli_epi32(signBits, 32 - shiftAmount);
                const auto biasedDividend = _mm_add_epi32(dividendVector, roundingBias);
                const auto quotient = _mm_srai_epi32(biasedDividend, shiftAmount);

                if constexpr (_Divisor_ > 0)
                    return quotient;

                return _mm_sub_epi32(_mm_setzero_si128(), quotient);
            }

            constexpr auto shiftForMagic = math::CountTrailingZeroBits(uint32(absoluteDivisor) - 1);
            constexpr auto magicMultiplier = int32(1 + ((uint64(1) << (32 + shiftForMagic)) / uint32(absoluteDivisor)) - (int64(1) << 32));

            const SimdDivisor<arch::CpuFeature::SSE2, int32_t> divisorParams(
                magicMultiplier, shiftForMagic, _Divisor_ < 0 ? -1 : 0);

            const auto productLowEven = _mm_mul_epu32(dividendVector, divisorParams.getMultiplier()); // dividendVector[0], dividendVector[2]
            const auto productHighEven = _mm_srli_epi64(productLowEven, 32);

            const auto shiftedDividendOdd = _mm_srli_epi64(dividendVector, 32); // dividendVector[1], dividendVector[3]
            const auto productLowOdd = _mm_mul_epu32(shiftedDividendOdd, divisorParams.getMultiplier());

            const auto oddMask = _mm_set_epi32(-1, 0, -1, 0);
            const auto productHighOdd = _mm_and_si128(productLowOdd, oddMask);

            const auto combinedHighProduct = _mm_or_si128(productHighEven, productHighOdd);

            const auto dividendSignMask = _mm_srai_epi32(dividendVector, 31);
            const auto multiplierSignMask = _mm_srai_epi32(divisorParams.getMultiplier(), 31);

            const auto correctionFromMultiplier = _mm_and_si128(divisorParams.getMultiplier(), dividendSignMask);
            const auto correctionFromDividend = _mm_and_si128(dividendVector, multiplierSignMask);

            const auto totalCorrection = _mm_add_epi32(correctionFromMultiplier, correctionFromDividend);
            const auto signedProduct = _mm_sub_epi32(combinedHighProduct, totalCorrection);

            const auto adjustedSum = _mm_add_epi32(signedProduct, dividendVector);
            const auto shiftedQuotient = _mm_sra_epi32(adjustedSum, divisorParams.getFirstShiftCount());

            const auto signDifference = _mm_sub_epi32(dividendSignMask, divisorParams.getDivisorSign());
            const auto correctedQuotient = _mm_sub_epi32(shiftedQuotient, signDifference);

            return _mm_xor_si128(correctedQuotient, divisorParams.getDivisorSign());
        }
        else if constexpr (is_epu16_v<_DesiredType_>) {
            constexpr int trailingZeroBitCount = math::CountTrailingZeroBits(_Divisor_);

            if constexpr ((_Divisor_ & (_Divisor_ - 1u)) == 0)
                return _mm_srli_epi16(dividendVector, trailingZeroBitCount);

            constexpr auto magicDivisionMultiplier = uint16((1u << uint32(trailingZeroBitCount + 16)) / _Divisor_);

            constexpr uint32_t magicDivisionRemainder = ((uint32_t(1) << uint32_t(trailingZeroBitCount + 16)) 
                - uint32_t(_Divisor_) * magicDivisionMultiplier);

            constexpr bool shouldRoundDown = (2u * magicDivisionRemainder < _Divisor_);

            if (shouldRoundDown)
                dividendVector = dividendVector + _mm_set1_epi16(1);

            constexpr uint16 adjustedMagicMultiplier = shouldRoundDown
                ? magicDivisionMultiplier
                : magicDivisionMultiplier + 1;

            const auto multiplierVector = _mm_set1_epi16(static_cast<int16_t>(adjustedMagicMultiplier));

            auto highProductVector = _mm_mulhi_epu16(dividendVector, multiplierVector);
            auto quotientVector = _mm_srli_epi16(highProductVector, trailingZeroBitCount);

            if constexpr (shouldRoundDown) {
                auto isDividendZeroMask = _mm_cmpeq_epi16(dividendVector, _mm_setzero_si128());

                return _mm_or_si128(
                    _mm_and_si128(
                        isDividendZeroMask,
                        broadcast<__m128i>(uint16_t(adjustedMagicMultiplier >> trailingZeroBitCount))
                    ),
                    _mm_andnot_si128(quotientVector, _mm_set1_epi16(trailingZeroBitCount))
                );
            }
            else
                return quotientVector;
        
        }
    }
};

template <>
class BasicSimdImplementation<arch::CpuFeature::SSE3>: 
    public BasicSimdImplementation<arch::CpuFeature::SSE2> 
{

};

template <>
class BasicSimdImplementation<arch::CpuFeature::SSSE3>:
    public BasicSimdImplementation<arch::CpuFeature::SSE3>
{
    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_constexpr_cxx20 simd_stl_always_inline _VectorType_ shuffle(
        _VectorType_                                        vector,
        type_traits::__deduce_simd_shuffle_mask_type<
        sizeof(_VectorType_) / sizeof(_DesiredType_)>   shuffleMask) noexcept
    {
        return shuffle<_DesiredType_>(vector, vector, shuffleMask);
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_constexpr_cxx20 simd_stl_always_inline _VectorType_ shuffle(
        _VectorType_                                            vector,
        _VectorType_                                            secondVector,
        type_traits::__deduce_simd_shuffle_mask_type<
            sizeof(_VectorType_) / sizeof(_DesiredType_)>       shuffleMask) noexcept
    {
        if      constexpr (is_epi64_v<_DesiredType_> || is_epu64_v<_DesiredType_> || is_pd_v<_DesiredType_>)
            return cast<__m128d, _VectorType_>(_mm_shuffle_pd(
                cast<_VectorType_, __m128d>(vector),
                cast<_VectorType_, __m128d>(secondVector),
                shuffleMask));
        else if constexpr (is_epi32_v<_DesiredType_> || is_epu32_v<_DesiredType_> || is_ps_v<_DesiredType_>)
                return cast<__m128i, _VectorType_>(_mm_shuffle_epi32(
                    cast<_VectorType_, __m128i>(vector),
                    cast<_VectorType_, __m128i>(secondVector),
                    shuffleMask)
                );
        else if constexpr (is_epi16_v<_DesiredType_> || is_epu16_v<_DesiredType_>) {
            const auto rawMask = shuffleMask & 0xFFFFFF;

            const int8 index0 = (rawMask >> 0) & 0x7;
            const int8 index1 = (rawMask >> 3) & 0x7;

            const int8 index2 = (rawMask >> 6) & 0x7;
            const int8 index3 = (rawMask >> 9) & 0x7;

            const int8 index4 = (rawMask >> 12) & 0x7;
            const int8 index5 = (rawMask >> 15) & 0x7;

            const int8 index6 = (rawMask >> 18) & 0x7;
            const int8 index7 = (rawMask >> 21) & 0x7;

            const int8 low0 = index0 << 1;
            const int8 low1 = index1 << 1;

            const int8 low2 = index2 << 1;
            const int8 low3 = index3 << 1;

            const int8 low4 = index4 << 1;
            const int8 low5 = index5 << 1;

            const int8 low6 = index6 << 1;
            const int8 low7 = index7 << 1;

            const auto byteMask = _mm_set_epi8(
                low7 + 1, low7, low6 + 1, low6, low5 + 1, low5, low4 + 1, low4,
                low3 + 1, low3, low2 + 1, low2, low1 + 1, low1, low0 + 1, low0
            );

            return _mm_shuffle_epi8(vector, byteMask);
        }
        else if constexpr (is_epi8_v<_DesiredType_> || is_epu8_v<_DesiredType_>) {
           // return _mm_shuffle_epi8(vector, (shuffleMask));
        }
    }
};

template <>
class BasicSimdImplementation<arch::CpuFeature::SSE41>: 
    public BasicSimdImplementation<arch::CpuFeature::SSSE3> 
{};

template <>
class BasicSimdImplementation<arch::CpuFeature::SSE42>: 
    public BasicSimdImplementation<arch::CpuFeature::SSE41>
{};

template <>
class BasicSimdImplementation<arch::CpuFeature::AVX2> {
public:
    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_constexpr_cxx20 simd_stl_always_inline _VectorType_ shuffle(
        _VectorType_                                        vector,
        type_traits::__deduce_simd_shuffle_mask_type<
            sizeof(_VectorType_) / sizeof(_DesiredType_)>   shuffleMask) noexcept
    {
        return shuffle<_DesiredType_>(vector, vector, shuffleMask);
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_constexpr_cxx20 simd_stl_always_inline _VectorType_ shuffle(
        _VectorType_                                            vector,
        _VectorType_                                            vectorSecond,
        type_traits::__deduce_simd_shuffle_mask_type<
            sizeof(_VectorType_) / sizeof(_DesiredType_)>       shuffleMask) noexcept
    {
        if constexpr (is_epi64_v<_DesiredType_> || is_epu64_v<_DesiredType_> || is_pd_v<_DesiredType_>)
            return _mm256_shuffle_pd(
                cast<_VectorType_, __m256d>(vector),
                cast<_VectorType_, __m256d>(vectorSecond),
                shuffleMask
            );
        else if constexpr (is_ps_v<_DesiredType_>)
            return _mm256_shuffle_ps(
                cast<_VectorType_, __m256>(vector),
                cast<_VectorType_, __m256>(vectorSecond),
                shuffleMask
            );
        else if constexpr (is_epi32_v<_DesiredType_> || is_epu32_v<_DesiredType_>)
            return _mm256_shuffle_epi32(
                cast<_VectorType_, __m256i>(vector),
                cast<_VectorType_, __m256i>(vectorSecond),
                shuffleMask
            );
        else if constexpr (is_epi16_v<_DesiredType_> || is_epu16_v<_DesiredType_>) {
            const auto shuffledHigh = _mm256_shufflehi_epi16(vector, shuffleMask);
            const auto shuffledLow  = _mm256_shufflelo_epi16(vectorSecond, shuffleMask);

            return _mm256_set_epi64x(
                _mm_cvtsd_f64(cast<_VectorType_, __m128d>(shuffledLow)),
                _mm_cvtsd_f64(cast<_VectorType_, __m128d>(shuffledHigh)),
                extract<int64>(shuffledHigh, 2),
                extract<int64>(shuffledLow, 2)
            );
        }
        else if constexpr (is_epi8_v<_DesiredType_> || is_epu8_v<_DesiredType_>) {
            return _mm256_shuffle_epi8(
                cast<_VectorType_, __m256i>(vector),
                cast<_VectorType_, __m256i>(vectorSecond),
                shuffleMask
            );
        }
    }


    template <
        typename _DesiredVectorElementType_,
        typename _VectorType_>
    static simd_stl_constexpr_cxx20 simd_stl_always_inline int32 maskFromVector(_VectorType_ vector) noexcept {
        if constexpr (is_pd_v<_DesiredVectorElementType_> || is_epi64_v<_DesiredVectorElementType_> || is_epu64_v<_DesiredVectorElementType_>)
            return _mm256_movemask_pd(cast<_VectorType_, __m256d>(vector));
        else if constexpr (is_ps_v<_DesiredVectorElementType_> || is_epi32_v<_DesiredVectorElementType_> || is_epu32_v<_DesiredVectorElementType_>)
            return _mm256_movemask_ps(cast<_VectorType_, __m256>(vector));
        else if constexpr (is_epi16_v<_DesiredVectorElementType_> || is_epu16_v<_DesiredVectorElementType_>) {

        }
        else if constexpr (is_epi8_v<_DesiredVectorElementType_> || is_epu8_v<_DesiredVectorElementType_>) {
            return _mm256_movemask_epi8(cast<_VectorType_, __m256i>(vector));
        }
    }

    template <
        typename _MaskType_,
        typename _DesiredVectorElementType_,
        typename _VectorType_> 
    static simd_stl_constexpr_cxx20 simd_stl_always_inline _VectorType_ maskToVector(_MaskType_ mask) noexcept {
        if constexpr (is_epi64_v<_DesiredVectorElementType_> || is_epu64_v<_DesiredVectorElementType_> || is_pd_v<_DesiredVectorElementType_>) {
            _DesiredVectorElementType_ arrayTemp[4];

            arrayTemp[0] = (mask        & 1) ? 0xFFFFFFFFFFFFFFFF : 0;
            arrayTemp[1] = ((mask >> 1) & 1) ? 0xFFFFFFFFFFFFFFFF : 0;
            arrayTemp[2] = ((mask >> 2) & 1) ? 0xFFFFFFFFFFFFFFFF : 0;
            arrayTemp[3] = ((mask >> 3) & 1) ? 0xFFFFFFFFFFFFFFFF : 0;

            return loadUnaligned<_VectorType_>(arrayTemp);
        }
        else if constexpr (is_epi32_v<_DesiredVectorElementType_> || is_epu32_v<_DesiredVectorElementType_> || is_ps_v<_DesiredVectorElementType_>) {
            const auto vshiftСount = _mm256_set_epi32(24, 25, 26, 27, 28, 29, 30, 31);
            auto bcast = _mm256_set1_epi32(mask);
            // Старший бит каждого элемента - соответствующий бит в маске
            auto shifted = _mm256_sllv_epi32(bcast, vshiftСount); // AVX2
            return shifted;
        }
        else if constexpr (is_epi16_v<_DesiredVectorElementType_> || is_epu16_v<_DesiredVectorElementType_>) {
            /*const auto shuffle = _mm256_setr_epi32(0, 0, 0x01010101, 0x01010101, 0, 0, 0x01010101, 0x01010101);
            auto v = _mm256_shuffle_epi8(broadcast(mask), shuffle);

            const auto bitselect = _mm256_setr_epi8(
                1, 1 << 1, 1 << 2, 1 << 3, 1 << 4, 1 << 5, 1 << 6, 1U << 7, 1U << 8, 1U << 9, 1U << 10, 1U << 11, 1U << 12, 1U << 13, 1U << 14, 1U << 15
                1, 1 << 1, 1 << 2, 1 << 3, 1 << 4, 1 << 5, 1 << 6, 1U << 7, 1U << 8, 1U << 9, 1U << 10, 1U << 11, 1U << 12, 1U << 13, 1U << 14, 1U << 15);

            v = _mm256_and_si256(v, bitselect);
            v = _mm256_min_epu8(v, _mm256_set1_epi8(1));

            return v;*/
        }
        else if constexpr (is_epi8_v<_DesiredVectorElementType_> || is_epu8_v<_DesiredVectorElementType_>) {
            auto vmask = _mm256_set1_epi32(mask);
            const auto shuffle = _mm256_setr_epi64x(
                0x0000000000000000, 0x0101010101010101,
                0x0202020202020202, 0x0303030303030303);

            vmask = _mm256_shuffle_epi8(vmask, shuffle);
            const auto bitMask = _mm256_set1_epi64x(0x7fbfdfeff7fbfdfe);

            vmask = _mm256_or_si256(vmask, bitMask);
            return _mm256_cmpeq_epi8(vmask, _mm256_set1_epi64x(-1));
        }
    }
    
    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_constexpr_cxx20 simd_stl_always_inline _VectorType_ loadUnaligned(const _DesiredType_* where) noexcept {
        return _mm256_lddqu_si256(reinterpret_cast<const __m256i*>(where));
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_constexpr_cxx20 simd_stl_always_inline _VectorType_ loadAligned(const _DesiredType_* where) noexcept {
        return _mm256_load_si256(reinterpret_cast<const __m256i*>(where));
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_constexpr_cxx20 simd_stl_always_inline void storeUnaligned(
        _DesiredType_*      where,
        const _VectorType_  vector) noexcept
    {
        return _mm256_storeu_si256(reinterpret_cast<__m256i*>(where), vector);
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_constexpr_cxx20 simd_stl_always_inline void storeAligned(
        _DesiredType_*      where,
        const _VectorType_  vector) noexcept
    {
        return _mm256_store_si256(reinterpret_cast<__m256i*>(where), vector);
    }
    
    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_constexpr_cxx20 simd_stl_always_inline void maskStoreUnaligned(
        _DesiredType_*                                      where,
        const type_traits::__deduce_simd_shuffle_mask_type<
            sizeof(_VectorType_) / sizeof(_DesiredType_)>   mask,
        const _VectorType_                                  vector) noexcept
    {
        _mm256_maskstore_ps(
            reinterpret_cast<float*>(where), 
            maskToVector(mask), cast<_VectorType_, __m256>(vector));
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_constexpr_cxx20 simd_stl_always_inline void maskStoreAligned(
        _DesiredType_*                                          where,
        const type_traits::__deduce_simd_shuffle_mask_type<
            sizeof(_VectorType_) / sizeof(_DesiredType_)>       mask,
        const _VectorType_                                      vector) noexcept
    {
        _mm256_maskstore_ps(
            reinterpret_cast<float*>(where),
            maskToVector(mask), cast<_VectorType_, __m256>(vector));
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_constexpr_cxx20 simd_stl_always_inline _VectorType_ maskLoadUnaligned(
        const _DesiredType_*                                where,
        const type_traits::__deduce_simd_shuffle_mask_type<
            sizeof(_VectorType_) / sizeof(_DesiredType_)>   mask) noexcept
    {
        return cast<__m256, _VectorType_>(
            _mm256_maskload_ps(
                reinterpret_cast<const float*>(where),
                maskToVector(mask)));
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_constexpr_cxx20 simd_stl_always_inline _VectorType_ maskLoadAligned(
        const _DesiredType_*                                where,
        const type_traits::__deduce_simd_shuffle_mask_type<
            sizeof(_VectorType_) / sizeof(_DesiredType_)>   mask) noexcept
    {
        return cast<__m256, _VectorType_>(
            _mm256_maskload_ps(
                reinterpret_cast<const float*>(where),
                maskToVector(mask)));
    }

    template <
        typename    _FromVector_,
        typename    _ToVector_,
        bool        _SafeCast_ = false>
    static simd_stl_constexpr_cxx20 simd_stl_always_inline _ToVector_ cast(_FromVector_ from) noexcept {
        if constexpr (std::is_same_v<_ToVector_, _FromVector_>)
            return from;


        else if constexpr (std::is_same_v<_FromVector_, __m256> && std::is_same_v<_ToVector_, __m256i>)
            return _mm256_castps_si256(from);
        else if constexpr (std::is_same_v<_FromVector_, __m256> && std::is_same_v<_ToVector_, __m256d>)
            return _mm256_castps_pd(from);

        else if constexpr (std::is_same_v<_FromVector_, __m256d> && std::is_same_v<_ToVector_, __m256>)
            return _mm256_castpd_ps(from);
        else if constexpr (std::is_same_v<_FromVector_, __m256d> && std::is_same_v<_ToVector_, __m256i>)
            return _mm256_castpd_si256(from);

        else if constexpr (std::is_same_v<_FromVector_, __m256i> && std::is_same_v<_ToVector_, __m256>)
            return _mm256_castsi256_ps(from);
        else if constexpr (std::is_same_v<_FromVector_, __m256i> && std::is_same_v<_ToVector_, __m256d>)
            return _mm256_castsi256_pd(from);


        else if constexpr (std::is_same_v<_FromVector_, __m128> && std::is_same_v<_ToVector_, __m128i>)
            return _mm_castps_si128(from);
        else if constexpr (std::is_same_v<_FromVector_, __m128> && std::is_same_v<_ToVector_, __m128d>)
            return _mm_castps_pd(from);

        else if constexpr (std::is_same_v<_FromVector_, __m128d> && std::is_same_v<_ToVector_, __m128>)
            return _mm_castpd_ps(from);
        else if constexpr (std::is_same_v<_FromVector_, __m128d> && std::is_same_v<_ToVector_, __m128i>)
            return _mm_castpd_si128(from);

        else if constexpr (std::is_same_v<_FromVector_, __m128i> && std::is_same_v<_ToVector_, __m128>)
            return _mm_castsi128_ps(from);
        else if constexpr (std::is_same_v<_FromVector_, __m128i> && std::is_same_v<_ToVector_, __m128d>)
            return _mm_castsi128_pd(from);

        // Zero extend
        else if constexpr (std::is_same_v<_FromVector_, __m128> && std::is_same_v<_ToVector_, __m256>   && _SafeCast_ == true)
            return _mm256_insertf128_ps(_mm256_castps128_ps256(from), _mm_setzero_ps(), 1);
        else if constexpr (std::is_same_v<_FromVector_, __m128d> && std::is_same_v<_ToVector_, __m256d> && _SafeCast_ == true)
            return _mm256_insertf128_pd(_mm256_castpd128_pd256(from), _mm_setzero_pd(), 1);
        else if constexpr (std::is_same_v<_FromVector_, __m128i> && std::is_same_v<_ToVector_, __m256i> && _SafeCast_ == true)
            return _mm256_insertf128_si256(_mm256_castsi128_si256(from), _mm_setzero_si128(), 1);

        // Undefined
        else if constexpr (std::is_same_v<_FromVector_, __m128> && std::is_same_v<_ToVector_, __m256>   && _SafeCast_ == false)
            return _mm256_castps128_ps256(from);
        else if constexpr (std::is_same_v<_FromVector_, __m128d> && std::is_same_v<_ToVector_, __m256d> && _SafeCast_ == false)
            return _mm256_castpd128_pd256(from);
        else if constexpr (std::is_same_v<_FromVector_, __m128i> && std::is_same_v<_ToVector_, __m256i> && _SafeCast_ == false)
            return _mm256_castsi128_si256(from);

        // Truncate
        else if constexpr (std::is_same_v<_FromVector_, __m256> && std::is_same_v<_ToVector_, __m128>)
            return _mm256_castps256_ps128(from);
        else if constexpr (std::is_same_v<_FromVector_, __m256d> && std::is_same_v<_ToVector_, __m128d>)
            return _mm256_castpd256_pd128(from);
        else if constexpr (std::is_same_v<_FromVector_, __m256i> && std::is_same_v<_ToVector_, __m128i>)
            return _mm256_castsi256_si128(from);
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_constexpr_cxx20 simd_stl_always_inline _VectorType_ decrement(_VectorType_ vector) noexcept {
        return sub(vector, broadcast(1));
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_constexpr_cxx20 simd_stl_always_inline _VectorType_ increment(_VectorType_ vector) noexcept {
        return add(vector, broadcast(1));
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_constexpr_cxx20 simd_stl_always_inline _DesiredType_ extract(
        _VectorType_    vector,
        uint8           where) noexcept
    {
        if constexpr (is_epi64_v<_DesiredType_> || is_epu64_v<_DesiredType_> || is_pd_v<_DesiredType_>) {
            _DesiredType_ x[4];
            storeUnaligned(x);
            return x[where & 3];
        }
        else if constexpr (is_epi32_v<_DesiredType_> || is_epu32_v<_DesiredType_> || is_ps_v<_DesiredType_>) {
            _DesiredType_ x[8];
            storeUnaligned(x);
            return x[where & 7];
        }
        else if constexpr (is_epi16_v<_DesiredType_> || is_epu16_v<_DesiredType_>) {
            _DesiredType_ x[16];
            storeUnaligned(x);
            return x[where & 0x0F];
        }
        else if constexpr (is_epi8_v<_DesiredType_> || is_epu8_v<_DesiredType_>) {
            _DesiredType_ x[32];
            storeUnaligned(x);
            return x[where & 0x1F];
        }
    }

    template <typename _VectorType_>
    static simd_stl_constexpr_cxx20 simd_stl_always_inline _VectorType_ constructZero() noexcept {
        if      constexpr (std::is_same_v<_VectorType_, __m256d>)
            return _mm256_setzero_pd();
        else if constexpr (std::is_same_v<_VectorType_, __m256i>)
            return _mm256_setzero_si256();
        else if constexpr (std::is_same_v<_VectorType_, __m256>)
            return _mm256_setzero_ps();
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_constexpr_cxx20 simd_stl_always_inline _VectorType_ broadcast(_DesiredType_ value) noexcept {
        if constexpr (is_epi64_v<_DesiredType_> || is_epu64_v<_DesiredType_>)
            return cast<__m256i, _VectorType_>(_mm256_set1_epi64x(value));
        else if constexpr (is_epi32_v<_DesiredType_> || is_epu32_v<_DesiredType_>)
            return cast<__m256i, _VectorType_>(_mm256_set1_epi32(value));
        else if constexpr (is_epi16_v<_DesiredType_> || is_epu16_v<_DesiredType_>)
            return cast<__m256i, _VectorType_>(_mm256_set1_epi16(value));
        else if constexpr (is_epi8_v<_DesiredType_> || is_epu8_v<_DesiredType_>)
            return cast<__m256i, _VectorType_>(_mm256_set1_epi8(value));
        else if constexpr (is_ps_v<_DesiredType_>)
            return cast<__m256, _VectorType_>(_mm256_set1_ps(value));
        else if constexpr (is_pd_v<_DesiredType_>)
            return cast<__m256d, _VectorType_>(_mm256_set1_pd(value));
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_constexpr_cxx20 simd_stl_always_inline _VectorType_ add(
        _VectorType_ left,
        _VectorType_ right) noexcept
    {
        if constexpr (is_epi64_v<_DesiredType_> || is_epu64_v<_DesiredType_>)
            return _mm256_add_epi64(left, right);
        else if constexpr (is_epi32_v<_DesiredType_> || is_epu32_v<_DesiredType_>)
            return _mm256_add_epi32(left, right);
        else if constexpr (is_epi16_v<_DesiredType_> || is_epu16_v<_DesiredType_>)
            return _mm256_add_epi16(left, right);
        else if constexpr (is_epi8_v<_DesiredType_> || is_epu8_v<_DesiredType_>)
            return _mm256_add_epi8(left, right);
        else if constexpr (is_ps_v<_DesiredType_>)
            return _mm256_add_ps(left, right);
        else if constexpr (is_pd_v<_DesiredType_>)
            return _mm256_add_pd(left, right);
    }
    

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_constexpr_cxx20 simd_stl_always_inline _VectorType_ sub(
        _VectorType_ left,
        _VectorType_ right) noexcept
    {
        if constexpr (is_epi64_v<_DesiredType_> || is_epu64_v<_DesiredType_>)
            return _mm256_sub_epi64(left, right);
        else if constexpr (is_epi32_v<_DesiredType_> || is_epu32_v<_DesiredType_>)
            return _mm256_sub_epi32(left, right);
        else if constexpr (is_epi16_v<_DesiredType_> || is_epu16_v<_DesiredType_>)
            return _mm256_sub_epi16(left, right);
        else if constexpr (is_epi8_v<_DesiredType_> || is_epu8_v<_DesiredType_>)
            return _mm256_sub_epi8(left, right);
        else if constexpr (is_ps_v<_DesiredType_>)
            return _mm256_sub_ps(left, right);
        else if constexpr (is_pd_v<_DesiredType_>)
            return _mm256_sub_pd(left, right);
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_constexpr_cxx20 simd_stl_always_inline _VectorType_ mul(
        _VectorType_ left,
        _VectorType_ right) noexcept
    {
        if constexpr (is_epi64_v<_DesiredType_> || is_epu64_v<_DesiredType_>) {
            auto ymm2 = _mm256_mul_epu32(_mm256_srli_epi64(right, 32), left);
            auto ymm3 = _mm256_mul_epu32(_mm256_srli_epi64(left, 32), right);

            ymm2 = _mm256_slli_epi64(_mm256_add_epi64(ymm3, ymm2), 32);
            return _mm256_add_epi64(_mm256_mul_epu32(right, left), ymm2);
        }
        else if constexpr (is_epi32_v<_DesiredType_>)
            return _mm256_mul_epi32(left, right);
        else if constexpr (is_epu32_v<_DesiredType_>)
            return _mm256_mul_epu32(left, right);
        else if constexpr (is_epi16_v<_DesiredType_> || is_epu16_v<_DesiredType_>)
            return _mm256_mullo_epi16(left, right);
        else if constexpr (is_epi8_v<_DesiredType_> || is_epu8_v<_DesiredType_>) {
            auto ymm3 = _mm256_unpacklo_epi8(right, right);
            auto ymm2 = _mm256_unpacklo_epi8(left, left);

            left = _mm256_unpackhi_epi8(left, left);
            right = _mm256_unpackhi_epi8(right, right);

            ymm2 = _mm256_mullo_epi16(ymm2, ymm3);
            left = _mm256_mullo_epi16(left, right);

            ymm2 = _mm256_shuffle_epi8(ymm2, ymm2);
            left = _mm256_shuffle_epi8(left, left);

            return _mm256_blend_epi32(left, ymm2, 0x33);
        }
        else if constexpr (is_ps_v<_DesiredType_>)
            return _mm256_mul_ps(left, right);
        else if constexpr (is_pd_v<_DesiredType_>)
            return _mm256_mul_pd(left, right);
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_constexpr_cxx20 simd_stl_always_inline _VectorType_ div(
        _VectorType_ left,
        _VectorType_ right) noexcept
    {
        if      constexpr (is_epi64_v<_DesiredType_> || is_epu64_v<_DesiredType_>)
            return cast<__m256d, _VectorType_>(_mm256_div_pd(left, right));
        else if constexpr (is_epi32_v<_DesiredType_> || is_epu32_v<_DesiredType_> || is_ps_v<_DesiredType_>)
            return cast<__m256, _VectorType_>(_mm256_div_ps(left, right));
        else if constexpr (is_epi16_v<_DesiredType_> || is_epu16_v<_DesiredType_>)
            return cast<__m256i, _VectorType_>(div_u16(left, right));
        else if constexpr (is_epi8_v<_DesiredType_> || is_epu8_v<_DesiredType_>)
            return cast<__m256i, _VectorType_>(div_u8(left, right));
    }

    template <typename _VectorType_>
    static simd_stl_constexpr_cxx20 simd_stl_always_inline _VectorType_ bitwiseNot(_VectorType_ vector) noexcept {
        if      constexpr (std::is_same_v<_VectorType_, __m256d>)
            return _mm256_xor_pd(vector, _mm256_cmp_pd(vector, vector, _CMP_EQ_OQ));
        else if constexpr (std::is_same_v<_VectorType_, __m256>)
            return _mm256_xor_ps(vector, _mm256_cmp_ps(vector, vector, _CMP_EQ_OQ));
        else if constexpr (std::is_same_v<_VectorType_, __m256i>)
            return _mm256_xor_si256(vector, _mm256_cmpeq_epi64(vector, vector));
    }

    template <typename _VectorType_>
    static simd_stl_constexpr_cxx20 simd_stl_always_inline _VectorType_ bitwiseXor(
        _VectorType_ left,
        _VectorType_ right) noexcept
    {
        if      constexpr (std::is_same_v<_VectorType_, __m256d>)
            return _mm256_xor_pd(left, right);
        else if constexpr (std::is_same_v<_VectorType_, __m256>)
            return _mm256_xor_ps(left, right);
        else if constexpr (std::is_same_v<_VectorType_, __m256i>)
            return _mm256_xor_si256(left, right);
    }

    template <typename _VectorType_>
    static simd_stl_constexpr_cxx20 simd_stl_always_inline _VectorType_ bitwiseAnd(
        _VectorType_ left,
        _VectorType_ right) noexcept
    {
        if      constexpr (std::is_same_v<_VectorType_, __m256d>)
            return _mm256_and_pd(left, right);
        else if constexpr (std::is_same_v<_VectorType_, __m256>)
            return _mm256_and_ps(left, right);
        else if constexpr (std::is_same_v<_VectorType_, __m256i>)
            return _mm256_and_si256(left, right);
    }

    template <typename _VectorType_>
    static simd_stl_constexpr_cxx20 simd_stl_always_inline _VectorType_ bitwiseOr(
        _VectorType_ left,
        _VectorType_ right) noexcept
    {
        if      constexpr (std::is_same_v<_VectorType_, __m256d>)
            return _mm256_or_pd(left, right);
        else if constexpr (std::is_same_v<_VectorType_, __m256>)
            return _mm256_or_ps(left, right);
        else if constexpr (std::is_same_v<_VectorType_, __m256>)
            return _mm256_or_si256(left, right);
    }

private:
    static simd_stl_constexpr_cxx20 simd_stl_always_inline __m256i divLow_u8_i32x8(
        __m256i left,
        __m256i right, 
        float mul) noexcept 
    {
        const auto af = _mm256_cvtepi32_ps(left);
        const auto bf = _mm256_cvtepi32_ps(right);

        const auto m1 = _mm256_mul_ps(af, _mm256_set1_ps(1.001f * mul));
        const auto m2 = _mm256_rcp_ps(bf);

        return _mm256_cvttps_epi32(_mm256_mul_ps(m1, m2));
    }

    static simd_stl_constexpr_cxx20 simd_stl_always_inline __m256i div_u8(
        __m256i left,
        __m256i right) noexcept 
    {
        const auto m0 = _mm256_set1_epi32(0x000000ff);
        const auto m1 = _mm256_set1_epi32(0x0000ff00);
        const auto m2 = _mm256_set1_epi32(0x00ff0000);

        const auto r0 = divLow_u8_i32x8(_mm256_and_si256(left, m0), _mm256_and_si256(right, m0), 1);
        auto r1 = divLow_u8_i32x8(_mm256_and_si256(left, m1), _mm256_and_si256(right, m1), 1);
        r1 = _mm256_slli_epi32(r1, 8);

        const auto r2 = divLow_u8_i32x8(_mm256_and_si256(left, m2), _mm256_and_si256(right, m2), 1 << 16);
        auto r3 = divLow_u8_i32x8(_mm256_srli_epi32(left, 24), _mm256_srli_epi32(right, 24), 1);

        r3 = _mm256_slli_epi32(r3, 24);

        const auto r01 = _mm256_or_si256(r0, r1);
        const auto r23 = _mm256_or_si256(r2, r3);

        return _mm256_blend_epi16(r01, r23, 0xAA);
    }

    static simd_stl_constexpr_cxx20 simd_stl_always_inline __m256i div_u16(
        const __m256i left, 
        const __m256i right) noexcept
    {
        const auto mask_lo = _mm256_set1_epi32(0x0000ffff);

        const auto a_lo_u32 = _mm256_and_si256(left, mask_lo);
        const auto b_lo_u32 = _mm256_and_si256(right, mask_lo);

        const auto a_hi_u32 = _mm256_srli_epi32(left, 16);
        const auto b_hi_u32 = _mm256_srli_epi32(right, 16);

        const auto  a_lo_f32 = _mm256_cvtepi32_ps(a_lo_u32);
        const auto  a_hi_f32 = _mm256_cvtepi32_ps(a_hi_u32);
        const auto  b_lo_f32 = _mm256_cvtepi32_ps(b_lo_u32);
        const auto  b_hi_f32 = _mm256_cvtepi32_ps(b_hi_u32);

        const auto  c_lo_f32 = _mm256_div_ps(a_lo_f32, b_lo_f32);
        const auto  c_hi_f32 = _mm256_div_ps(a_hi_f32, b_hi_f32);

        const auto c_lo_i32 = _mm256_cvttps_epi32(c_lo_f32); // values in the u16 range
        const auto c_hi_i32_0 = _mm256_cvttps_epi32(c_hi_f32); // values in the u16 range
        const auto c_hi_i32 = _mm256_slli_epi32(c_hi_i32_0, 16);

        return _mm256_or_si256(c_lo_i32, c_hi_i32);
    }
};

template <>
class BasicSimdImplementation<arch::CpuFeature::AVX512F> {
public:
    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_constexpr_cxx20 simd_stl_always_inline _VectorType_ shuffle(
        _VectorType_                                        vector,
        type_traits::__deduce_simd_shuffle_mask_type<
            sizeof(_VectorType_) / sizeof(_DesiredType_)>   shuffleMask) noexcept
    {
        return shuffle<_DesiredType_>(vector, vector, shuffleMask);
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_constexpr_cxx20 simd_stl_always_inline _VectorType_ shuffle(
        _VectorType_                                            vector,
        _VectorType_                                            vectorSecond,
        type_traits::__deduce_simd_shuffle_mask_type<
            sizeof(_VectorType_) / sizeof(_DesiredType_)>       shuffleMask) noexcept
    {
        if constexpr (is_epi64_v<_DesiredType_> || is_epu64_v<_DesiredType_> || is_pd_v<_DesiredType_>)
            return cast<__m512d, _VectorType_>(
                _mm512_shuffle_pd(
                    cast<_VectorType_, __m512d>(vector), 
                    cast<_VectorType_, __m512d>(vectorSecond), 
                    shuffleMask));
        else if constexpr (is_epi32_v<_DesiredType_> || is_epu32_v<_DesiredType_> || is_ps_v<_DesiredType_>)
            return cast<__m512, _VectorType_>(_mm512_shuffle_ps(
                cast<_VectorType_, __m512>(vector),
                cast<_VectorType_, __m512>(vectorSecond),
                shuffleMask
            ));
        else if constexpr (
            is_epi16_v<_DesiredType_> || is_epu16_v<_DesiredType_> ||
            is_epi8_v<_DesiredType_> || is_epu8_v<_DesiredType_>)
        {
            const auto shuffled1 = BasicSimdImplementation<arch::CpuFeature::AVX2>::template shuffle
                <_DesiredType_>(
                    cast<_VectorType_, __m256i>(vector),
                    cast<_VectorType_, __m256i>(vectorSecond),
                    shuffleMask
                );

            const auto shuffled2 = BasicSimdImplementation<arch::CpuFeature::AVX2>::template shuffle
                <_DesiredType_>(
                    _mm512_extracti32x8_epi32(vector, 1), 
                    _mm512_extracti32x8_epi32(vectorSecond, 1),
                    shuffleMask
                );

            return cast<__m512i, _VectorType_>(
                _mm512_inserti32x8(cast<__m256i, __m512i>(shuffled1), shuffled2, 1));
        }
    }
//
//    
//    template <
//        typename _DesiredVectorElementType_,
//        typename _VectorType_>
//    static simd_stl_constexpr_cxx20 simd_stl_always_inline int32 maskFromVector(_VectorType_ vector) noexcept {
//        if constexpr (is_pd_v<_DesiredVectorElementType_> || is_epi64_v<_DesiredVectorElementType_> || is_epu64_v<_DesiredVectorElementType_>)
//        {
//        }
//        else if constexpr (is_ps_v<_DesiredVectorElementType_> || is_epi32_v<_DesiredVectorElementType_> || is_epu32_v<_DesiredVectorElementType_>)
//        {
//        }
//        else if constexpr (is_epi16_v<_DesiredVectorElementType_> || is_epu16_v<_DesiredVectorElementType_>) {
//
//        }
//        else if constexpr (is_epi8_v<_DesiredVectorElementType_> || is_epu8_v<_DesiredVectorElementType_>) {
//
//        }
//    }
//
//    template <
//        typename _MaskType_,
//        typename _DesiredVectorElementType_,
//        typename _VectorType_> 
//    static simd_stl_constexpr_cxx20 simd_stl_always_inline _VectorType_ maskToVector(_MaskType_ mask) noexcept {
//        if constexpr (is_epi64_v<_DesiredVectorElementType_> || is_epu64_v<_DesiredVectorElementType_> || is_pd_v<_DesiredVectorElementType_>) {
//            _DesiredVectorElementType_ arrayTemp[4];
//
//            arrayTemp[0] = (mask & 1) ? 0xFFFFFFFFFFFFFFFF : 0;
//            arrayTemp[1] = ((mask >> 1) & 1) ? 0xFFFFFFFFFFFFFFFF : 0;
//            arrayTemp[2] = ((mask >> 2) & 1) ? 0xFFFFFFFFFFFFFFFF : 0;
//            arrayTemp[3] = ((mask >> 3) & 1) ? 0xFFFFFFFFFFFFFFFF : 0;
//            arrayTemp[4] = ((mask >> 4) & 1) ? 0xFFFFFFFFFFFFFFFF : 0;
//            arrayTemp[5] = ((mask >> 5) & 1) ? 0xFFFFFFFFFFFFFFFF : 0;
//            arrayTemp[6] = ((mask >> 6) & 1) ? 0xFFFFFFFFFFFFFFFF : 0;
//            arrayTemp[7] = ((mask >> 7) & 1) ? 0xFFFFFFFFFFFFFFFF : 0;
//
//            return loadUnaligned<_VectorType_>(arrayTemp);
//        }
//        else if constexpr (is_epi32_v<_DesiredVectorElementType_> || is_epu32_v<_DesiredVectorElementType_> || is_ps_v<_DesiredVectorElementType_>) {
//            const auto vshiftСount = _mm512_set_epi32(24, 25, 26, 27, 28, 29, 30, 31);
//            auto bcast = _mm512_set1_epi32(mask);
//            // Старший бит каждого элемента - соответствующий бит в маске
//            auto shifted = _mm512_sllv_epi32(bcast, vshiftСount); // AVX2
//            return shifted;
//        }
//        else if constexpr (is_epi16_v<_DesiredVectorElementType_> || is_epu16_v<_DesiredVectorElementType_>) {
//            const auto shuffled = _mm512_setr_epi32(0, 0, 0x01010101, 0x01010101);
//            auto v = shuffle<int8>(broadcast(mask), shuffle);
//
//            const auto bitselect = _mm512_setr_epi8(
//                1, 1 << 1, 1 << 2, 1 << 3, 1 << 4, 1 << 5, 1 << 6, 1U << 7,
//                1, 1 << 1, 1 << 2, 1 << 3, 1 << 4, 1 << 5, 1 << 6, 1U << 7);
//
//            v = _mm512_and_si512(v, bitselect);
//            v = _mm512_min_epu8(v, _mm512_set1_epi8(1));
//
//            return v;
//        }
//        else if constexpr (is_epi8_v<_DesiredVectorElementType_> || is_epu8_v<_DesiredVectorElementType_>) {
//            /* auto vmask = _mm512_set1_epi32(mask);
//             const auto shuffle = _mm512_setr_epi64x(
//                 0x0000000000000000, 0x0101010101010101,
//                 0x0202020202020202, 0x0303030303030303);
//
//             vmask = _mm256_shuffle_epi8(vmask, shuffle);
//             const auto bitMask = _mm256_set1_epi64x(0x7fbfdfeff7fbfdfe);
//
//             vmask = _mm256_or_si256(vmask, bitMask);
//             return _mm256_cmpeq_epi8(vmask, _mm256_set1_epi64x(-1));*/
//        }
//
//        template <
//            typename _DesiredType_,
//            typename _VectorType_>
//        static simd_stl_constexpr_cxx20 simd_stl_always_inline _VectorType_ loadUnaligned(const _DesiredType_ * where) noexcept {
//            return _mm512_loadu_si512(reinterpret_cast<const __m512i*>(where));
//        }
//
//        template <
//            typename _DesiredType_,
//            typename _VectorType_>
//        static simd_stl_constexpr_cxx20 simd_stl_always_inline _VectorType_ loadAligned(const _DesiredType_ * where) noexcept {
//            return _mm512_load_si512(reinterpret_cast<const __m512i*>(where));
//        }
//
//        template <
//            typename _DesiredType_,
//            typename _VectorType_>
//        simd_stl_constexpr_cxx20 simd_stl_always_inline void storeUnaligned(
//            _DesiredType_ * where,
//            const _VectorType_  vector) noexcept
//        {
//            return _mm512_storeu_si512(reinterpret_cast<__m512i*>(where), vector);
//        }
//
//        template <
//            typename _DesiredType_,
//            typename _VectorType_>
//        simd_stl_constexpr_cxx20 simd_stl_always_inline void storeAligned(
//            _DesiredType_ * where,
//            const _VectorType_  vector) noexcept
//        {
//            return _mm512_store_si512(reinterpret_cast<__m512i*>(where), vector);
//        }
//
//        template <
//            typename _DesiredType_,
//            typename _VectorType_>
//        simd_stl_constexpr_cxx20 simd_stl_always_inline void maskStoreUnaligned(
//            _DesiredType_ * where,
//            const uint64 /* ??? */  mask,
//            const _VectorType_      vector) noexcept
//        {
//            if constexpr (is_epi64_v<_DesiredType_> || is_epu64_v<_DesiredType_>)
//                return _mm512_mask_storeu_epi64(where, mask, cast<_VectorType_, __m512i>(vector));
//            else if constexpr (is_epi32_v<_DesiredType_> || is_epu32_v<_DesiredType_>)
//                return _mm512_mask_storeu_epi32(where, mask, cast<_VectorType_, __m512i>(vector));
//            else
//                return _mm512_storeu_si512(where, cast<_VectorType_, __m512i>(shuffle<_DesiredType_>(vector, mask)));
//        }
//
//        template <
//            typename _DesiredType_,
//            typename _VectorType_>
//        simd_stl_constexpr_cxx20 simd_stl_always_inline void maskStoreAligned(
//            _DesiredType_ * where,
//            const uint64 /* ??? */  mask,
//            const _VectorType_      vector) noexcept
//        {
//            if constexpr (is_epi64_v<_DesiredType_> || is_epu64_v<_DesiredType_>)
//                return _mm512_mask_store_epi64(where, mask, cast<_VectorType_, __m512i>(vector));
//            else if constexpr (is_epi32_v<_DesiredType_> || is_epu32_v<_DesiredType_>)
//                return _mm512_mask_store_epi32(where, mask, cast<_VectorType_, __m512i>(vector));
//            else
//                return _mm512_store_si512(where, cast<_VectorType_, __m512i>(shuffle<_DesiredType_>(vector, mask)));
//        }
//
//        template <
//            typename _DesiredType_,
//            typename _VectorType_>
//        simd_stl_constexpr_cxx20 simd_stl_always_inline _VectorType_ maskLoadUnaligned(
//            const _DesiredType_ * where,
//            const uint64            mask,
//            const _VectorType_      vector) noexcept
//        {
//            if constexpr (is_epi64_v<_DesiredType_> || is_epu64_v<_DesiredType_>)
//                return _mm512_mask_loadu_epi64(cast<_VectorType_, __m512i>(vector), mask, where);
//            else if constexpr (is_epi32_v<_DesiredType_> || is_epu32_v<_DesiredType_>)
//                return _mm512_mask_loadu_epi32(cast<_VectorType_, __m512i>(vector), mask, where);
//            else
//                return cast<_VectorType_, __m512i>(shuffle<_DesiredType_>(_mm512_loadu_si512(where), mask));
//        }
//
//        template <
//            typename _DesiredType_,
//            typename _VectorType_>
//        simd_stl_constexpr_cxx20 simd_stl_always_inline void maskLoadAligned(
//            const _DesiredType_ * where,
//            const uint64            mask,
//            const _VectorType_      vector) noexcept
//        {
//            if constexpr (is_epi64_v<_DesiredType_> || is_epu64_v<_DesiredType_>)
//                return _mm512_mask_load_epi64(cast<_VectorType_, __m512i>(vector), mask, where);
//            else if constexpr (is_epi32_v<_DesiredType_> || is_epu32_v<_DesiredType_>)
//                return _mm512_mask_load_epi32(cast<_VectorType_, __m512i>(vector), mask, where);
//            else
//                return cast<_VectorType_, __m512i>(shuffle<_DesiredType_>(_mm512_load_si512(where), mask));
//        }
//
//
        template <
            typename    _FromVector_,
            typename    _ToVector_,
            bool        _SafeCast_ = false>
        static simd_stl_constexpr_cxx20 simd_stl_always_inline _ToVector_ cast(const _FromVector_ from) noexcept {
            if constexpr (std::is_same_v<_ToVector_, _FromVector_>)
                return from;


            else if constexpr (std::is_same_v<_FromVector_, __m128> && std::is_same_v<_ToVector_, __m128i>)
                return _mm_castps_si128(from);
            else if constexpr (std::is_same_v<_FromVector_, __m128> && std::is_same_v<_ToVector_, __m128d>)
                return _mm_castps_pd(from);

            else if constexpr (std::is_same_v<_FromVector_, __m128d> && std::is_same_v<_ToVector_, __m128>)
                return _mm_castpd_ps(from);
            else if constexpr (std::is_same_v<_FromVector_, __m128d> && std::is_same_v<_ToVector_, __m128i>)
                return _mm_castpd_si128(from);

            else if constexpr (std::is_same_v<_FromVector_, __m128i> && std::is_same_v<_ToVector_, __m128>)
                return _mm_castsi128_ps(from);
            else if constexpr (std::is_same_v<_FromVector_, __m128i> && std::is_same_v<_ToVector_, __m128d>)
                return _mm_castsi128_pd(from);


            else if constexpr (std::is_same_v<_FromVector_, __m256> && std::is_same_v<_ToVector_, __m256i>)
                return _mm256_castps_si256(from);
            else if constexpr (std::is_same_v<_FromVector_, __m256> && std::is_same_v<_ToVector_, __m256d>)
                return _mm256_castps_pd(from);

            else if constexpr (std::is_same_v<_FromVector_, __m256d> && std::is_same_v<_ToVector_, __m256>)
                return _mm256_castpd_ps(from);
            else if constexpr (std::is_same_v<_FromVector_, __m256d> && std::is_same_v<_ToVector_, __m256i>)
                return _mm256_castpd_si256(from);

            else if constexpr (std::is_same_v<_FromVector_, __m256i> && std::is_same_v<_ToVector_, __m256>)
                return _mm256_castsi256_ps(from);
            else if constexpr (std::is_same_v<_FromVector_, __m256i> && std::is_same_v<_ToVector_, __m256d>)
                return _mm256_castsi256_pd(from);


            else if constexpr (std::is_same_v<_FromVector_, __m512> && std::is_same_v<_ToVector_, __m512i>)
                return _mm512_castps_si512(from);
            else if constexpr (std::is_same_v<_FromVector_, __m512> && std::is_same_v<_ToVector_, __m512d>)
                return _mm512_castps_pd(from);

            else if constexpr (std::is_same_v<_FromVector_, __m512d> && std::is_same_v<_ToVector_, __m512>)
                return _mm512_castpd_ps(from);
            else if constexpr (std::is_same_v<_FromVector_, __m512d> && std::is_same_v<_ToVector_, __m512i>)
                return _mm512_castpd_si512(from);

            else if constexpr (std::is_same_v<_FromVector_, __m512i> && std::is_same_v<_ToVector_, __m512>)
                return _mm512_castsi512_ps(from);
            else if constexpr (std::is_same_v<_FromVector_, __m512i> && std::is_same_v<_ToVector_, __m512d>)
                return _mm512_castsi512_pd(from);


            // Zero extend
            else if constexpr (std::is_same_v<_FromVector_, __m128> && std::is_same_v<_ToVector_, __m256> && _SafeCast_ == true)
                return _mm256_insertf128_ps(_mm256_castps128_ps256(from), _mm_setzero_ps(), 1);
            else if constexpr (std::is_same_v<_FromVector_, __m128d> && std::is_same_v<_ToVector_, __m256d> && _SafeCast_ == true)
                return _mm256_insertf128_pd(_mm256_castpd128_pd256(from), _mm_setzero_pd(), 1);
            else if constexpr (std::is_same_v<_FromVector_, __m128i> && std::is_same_v<_ToVector_, __m256i> && _SafeCast_ == true)
                return _mm256_insertf128_si256(_mm256_castsi128_si256(from), _mm_setzero_si128(), 1);

            // Zero extend
            else if constexpr (std::is_same_v<_FromVector_, __m128> && std::is_same_v<_ToVector_, __m256> && _SafeCast_ == false)
                return _mm256_castps128_ps256(from);
            else if constexpr (std::is_same_v<_FromVector_, __m128d> && std::is_same_v<_ToVector_, __m256d> && _SafeCast_ == false)
                return _mm256_castpd128_pd256(from);
            else if constexpr (std::is_same_v<_FromVector_, __m128i> && std::is_same_v<_ToVector_, __m256i> && _SafeCast_ == false)
                return _mm256_castsi128_si256(from);


            // Truncate
            else if constexpr (std::is_same_v<_FromVector_, __m256> && std::is_same_v<_ToVector_, __m128>)
                return _mm256_castps256_ps128(from);
            else if constexpr (std::is_same_v<_FromVector_, __m256d> && std::is_same_v<_ToVector_, __m128d>)
                return _mm256_castpd256_pd128(from);
            else if constexpr (std::is_same_v<_FromVector_, __m256i> && std::is_same_v<_ToVector_, __m128i>)
                return _mm256_castsi256_si128(from);

            // Zero extend
            else if constexpr (std::is_same_v<_FromVector_, __m128> && std::is_same_v<_ToVector_, __m512> && _SafeCast_ == true)
                return _mm512_insertf128_ps(_mm512_castps128_ps512(from), _mm_setzero_ps(), 1);
            else if constexpr (std::is_same_v<_FromVector_, __m128d> && std::is_same_v<_ToVector_, __m512d> && _SafeCast_ == true)
                return _mm512_insertf128_pd(_mm512_castpd128_pd512(from), _mm_setzero_pd(), 1);
            else if constexpr (std::is_same_v<_FromVector_, __m128i> && std::is_same_v<_ToVector_, __m512i> && _SafeCast_ == true)
                return _mm512_insertf128_si512(_mm512_castsi128_si512(from), _mm_setzero_si128(), 1);


            // Undefined
            else if constexpr (std::is_same_v<_FromVector_, __m128> && std::is_same_v<_ToVector_, __m512> && _SafeCast_ == false)
                return _mm512_castps128_ps512(from);
            else if constexpr (std::is_same_v<_FromVector_, __m128d> && std::is_same_v<_ToVector_, __m512d> && _SafeCast_ == false)
                return _mm512_castpd128_pd512(from);
            else if constexpr (std::is_same_v<_FromVector_, __m128i> && std::is_same_v<_ToVector_, __m512i> && _SafeCast_ == false)
                return _mm512_castsi128_si512(from);


            // Truncate
            else if constexpr (std::is_same_v<_FromVector_, __m512> && std::is_same_v<_ToVector_, __m128>)
                return _mm512_castps512_ps128(from);
            else if constexpr (std::is_same_v<_FromVector_, __m512d> && std::is_same_v<_ToVector_, __m128d>)
                return _mm512_castpd512_pd128(from);
            else if constexpr (std::is_same_v<_FromVector_, __m512i> && std::is_same_v<_ToVector_, __m128i>)
                return _mm512_castsi512_si128(from);

            // Zero extend
            else if constexpr (std::is_same_v<_FromVector_, __m256> && std::is_same_v<_ToVector_, __m512> && _SafeCast_ == true)
                return _mm512_insertf256_ps(_mm512_castps256_ps512(from), _mm256_setzero_ps(), 1);
            else if constexpr (std::is_same_v<_FromVector_, __m256d> && std::is_same_v<_ToVector_, __m512d> && _SafeCast_ == true)
                return _mm512_insertf256_pd(_mm512_castpd256_pd512(from), _mm256_setzero_pd(), 1);
            else if constexpr (std::is_same_v<_FromVector_, __m256i> && std::is_same_v<_ToVector_, __m512i> && _SafeCast_ == true)
                return _mm512_insertf256_si512(_mm512_castsi256_si512(from), _mm256_setzero_si256(), 1);

            // Undefined
            else if constexpr (std::is_same_v<_FromVector_, __m256> && std::is_same_v<_ToVector_, __m512> && _SafeCast_ == false)
                return _mm512_castps256_ps512(from);
            else if constexpr (std::is_same_v<_FromVector_, __m256d> && std::is_same_v<_ToVector_, __m512d> && _SafeCast_ == false)
                return _mm512_castpd256_pd512(from);
            else if constexpr (std::is_same_v<_FromVector_, __m256i> && std::is_same_v<_ToVector_, __m512i> && _SafeCast_ == false)
                return _mm512_castsi256_si512(from);

            // Truncate
            else if constexpr (std::is_same_v<_FromVector_, __m512> && std::is_same_v<_ToVector_, __m256>)
                return _mm512_castps512_ps256(from);
            else if constexpr (std::is_same_v<_FromVector_, __m512d> && std::is_same_v<_ToVector_, __m256d>)
                return _mm512_castpd512_pd256(from);
            else if constexpr (std::is_same_v<_FromVector_, __m512i> && std::is_same_v<_ToVector_, __m256i>)
                return _mm512_castsi512_si256(from);
        }

//        template <
//            typename _DesiredType_,
//            typename _VectorType_>
//        static simd_stl_constexpr_cxx20 simd_stl_always_inline _VectorType_ decrement(_VectorType_ vector) noexcept {
//            return sub(vector, broadcast(1));
//        }
//
//        template <
//            typename _DesiredType_,
//            typename _VectorType_>
//        static simd_stl_constexpr_cxx20 simd_stl_always_inline _VectorType_ increment(_VectorType_ vector) noexcept {
//            return add(vector, broadcast(1));
//        }
//
//        template <
//            typename _DesiredType_,
//            typename _VectorType_>
//        static simd_stl_constexpr_cxx20 simd_stl_always_inline _DesiredType_ extract(
//            _VectorType_    vector,
//            uint8           where) noexcept
//        {
//            if constexpr (is_epi64_v<_DesiredType_> || is_epu64_v<_DesiredType_> || is_pd_v<_DesiredType_>) {
//                _DesiredType_ x[4];
//                storeUnaligned(x);
//                return x[where & 3];
//            }
//            else if constexpr (is_epi32_v<_DesiredType_> || is_epu32_v<_DesiredType_> || is_ps_v<_DesiredType_>) {
//                _DesiredType_ x[8];
//                storeUnaligned(x);
//                return x[where & 7];
//            }
//            else if constexpr (is_epi16_v<_DesiredType_> || is_epu16_v<_DesiredType_>) {
//                _DesiredType_ x[16];
//                storeUnaligned(x);
//                return x[where & 0x0F];
//            }
//            else if constexpr (is_epi8_v<_DesiredType_> || is_epu8_v<_DesiredType_>) {
//                _DesiredType_ x[32];
//                storeUnaligned(x);
//                return x[where & 0x1F];
//            }
//        }
//
//        template <typename _VectorType_>
//        static simd_stl_constexpr_cxx20 simd_stl_always_inline _VectorType_ constructZero() noexcept {
//            if      constexpr (std::is_same_v<_VectorType_, __m512d>)
//                return _mm512_setzero_pd();
//            else if constexpr (std::is_same_v<_VectorType_, __m512i>)
//                return _mm512_setzero_si512();
//            else if constexpr (std::is_same_v<_VectorType_, __m512>)
//                return _mm512_setzero_ps();
//        }
//
//        template <
//            typename _DesiredType_,
//            typename _VectorType_>
//        static simd_stl_constexpr_cxx20 simd_stl_always_inline _VectorType_ broadcast(_DesiredType_ value) noexcept {
//
//        }
//
//        template <
//            typename _DesiredType_,
//            typename _VectorType_>
//        static simd_stl_constexpr_cxx20 simd_stl_always_inline _VectorType_ add(
//            _VectorType_ left,
//            _VectorType_ right) noexcept
//        {
//
//        }
//
//
//        template <
//            typename _DesiredType_,
//            typename _VectorType_>
//        static simd_stl_constexpr_cxx20 simd_stl_always_inline _VectorType_ sub(
//            _VectorType_ left,
//            _VectorType_ right) noexcept
//        {
//
//        }
//
//        template <
//            typename _DesiredType_,
//            typename _VectorType_>
//        static simd_stl_constexpr_cxx20 simd_stl_always_inline _VectorType_ mul(
//            _VectorType_ left,
//            _VectorType_ right) noexcept
//        {
//
//        }
//
//        template <
//            typename _DesiredType_,
//            typename _VectorType_>
//        static simd_stl_constexpr_cxx20 simd_stl_always_inline _VectorType_ div(
//            _VectorType_ left,
//            _VectorType_ right) noexcept
//        {
//            if      constexpr (is_epi64_v<_DesiredType_> || is_epu64_v<_DesiredType_>)
//                return cast<__m512d, _VectorType_>(_mm512_div_pd(
//                    cast<_VectorType_, __m512d>(left),
//                    cast<_VectorType_, __m512d>(right)));
//            else if constexpr (is_epi32_v<_DesiredType_> || is_epu32_v<_DesiredType_> || is_ps_v<_DesiredType_>)
//                return cast<__m512, _VectorType_>(_mm512_div_ps(
//                    cast<_VectorType_, __m512>(left),
//                    cast<_VectorType_, __m512>(right)));
//            else if constexpr (is_epi16_v<_DesiredType_> || is_epu16_v<_DesiredType_>)
//                return cast<__m512, _VectorType_>(div_u16(
//                    cast<_VectorType_, __m512i>(left),
//                    cast<_VectorType_, __m512i>(right)));
//            else if constexpr (is_epi8_v<_DesiredType_> || is_epu8_v<_DesiredType_>)
//                return cast<__m512i, _VectorType_>(div_u8(
//                    cast<_VectorType_, __m512i>(left),
//                    cast<_VectorType_, __m512i>(right)));
//        }
//
//        template <typename _VectorType_>
//        static simd_stl_constexpr_cxx20 simd_stl_always_inline _VectorType_ bitwiseNot(_VectorType_ vector) noexcept {
//            return cast<__m512i, _VectorType_>(
//                bitwiseXor(vector, _mm512_cmpeq_epi32(
//                    cast<_VectorType_, __m512i>(vector))));
//        }
//
//        template <typename _VectorType_>
//        static simd_stl_constexpr_cxx20 simd_stl_always_inline _VectorType_ bitwiseXor(
//            _VectorType_ left,
//            _VectorType_ right) noexcept
//        {
//            return cast<__m512i, _VectorType_>(_mm512_xor_si512(
//                cast<_VectorType_, __m512i>(left),
//                cast<_VectorType_, __m512i>(right)));
//        }
//
//        template <typename _VectorType_>
//        static simd_stl_constexpr_cxx20 simd_stl_always_inline _VectorType_ bitwiseAnd(
//            _VectorType_ left,
//            _VectorType_ right) noexcept
//        {
//            return cast<__m512i, _VectorType_>(_mm512_and_si512(
//                cast<_VectorType_, __m512i>(left),
//                cast<_VectorType_, __m512i>(right)));
//        }
//
//        template <typename _VectorType_>
//        static simd_stl_constexpr_cxx20 simd_stl_always_inline _VectorType_ bitwiseOr(
//            _VectorType_ left,
//            _VectorType_ right) noexcept
//        {
//            return cast<__m512i, _VectorType_>(_mm512_or_si512(
//                cast<_VectorType_, __m512i>(left),
//                cast<_VectorType_, __m512i>(right)));
//        }
//private:
//    static simd_stl_constexpr_cxx20 simd_stl_always_inline __m512i divLow_u8_i32x16(
//        __m512i left,
//        __m512i right,
//        float   mul) noexcept
//    {
//        const auto af = _mm512_cvtepi32_ps(left);
//        const auto bf = _mm512_cvtepi32_ps(right);
//
//        const auto m1 = _mm512_mul_ps(af, _mm512_set1_ps(1.001f * mul));
//        const auto m2 = _mm512_rcp14_ps(bf);
//
//        return _mm512_cvttps_epi32(_mm512_mul_ps(m1, m2));
//    }
//
//    static simd_stl_constexpr_cxx20 simd_stl_always_inline __m512i div_u8(
//        __m512i left,
//        __m512i right) noexcept
//    {
//        const auto m0 = _mm512_set1_epi32(0x000000ff);
//        const auto m1 = _mm512_set1_epi32(0x0000ff00);
//        const auto m2 = _mm512_set1_epi32(0x00ff0000);
//
//        const auto r0 = divLow_u8_i32x16(_mm512_and_si512(left, m0), _mm512_and_si512(right, m0), 1);
//        auto r1 = divLow_u8_i32x16(_mm512_and_si512(left, m1), _mm512_and_si512(right, m1), 1);
//        r1 = _mm512_slli_epi32(r1, 8);
//
//        const auto r2 = divLow_u8_i32x16(_mm512_and_si512(left, m2), _mm512_and_si512(right, m2), 1 << 16);
//        auto r3 = divLow_u8_i32x16(_mm512_srli_epi32(left, 24), _mm512_srli_epi32(right, 24), 1);
//
//        r3 = _mm512_slli_epi32(r3, 24);
//
//        auto r01 = _mm512_or_si512(r0, r1);
//        auto r23 = _mm512_or_si512(r2, r3);
//
//        return shuffle<int16>(r01, r23, maskToVector(0xAA));
//    }
//
//    static simd_stl_constexpr_cxx20 simd_stl_always_inline __m512i div_u16(
//        const __m512i left,
//        const __m512i right) noexcept
//    {
//        const auto mask_lo = _mm512_set1_epi32(0x0000ffff);
//
//        const auto a_lo_u32 = _mm512_and_si512(left, mask_lo);
//        const auto b_lo_u32 = _mm512_and_si512(right, mask_lo);
//
//        const auto a_hi_u32 = _mm512_srli_epi32(left, 16);
//        const auto b_hi_u32 = _mm512_srli_epi32(right, 16);
//
//        const auto  a_lo_f32 = _mm512_cvtepi32_ps(a_lo_u32);
//        const auto  a_hi_f32 = _mm512_cvtepi32_ps(a_hi_u32);
//        const auto  b_lo_f32 = _mm512_cvtepi32_ps(b_lo_u32);
//        const auto  b_hi_f32 = _mm512_cvtepi32_ps(b_hi_u32);
//
//        const auto  c_lo_f32 = _mm512_div_ps(a_lo_f32, b_lo_f32);
//        const auto  c_hi_f32 = _mm512_div_ps(a_hi_f32, b_hi_f32);
//
//        const auto c_lo_i32 = _mm512_cvttps_epi32(c_lo_f32); // values in the u16 range
//        const auto c_hi_i32_0 = _mm512_cvttps_epi32(c_hi_f32); // values in the u16 range
//        const auto c_hi_i32 = _mm512_slli_epi32(c_hi_i32_0, 16);
//
//        return _mm512_or_si512(c_lo_i32, c_hi_i32);
//    }
};

template <>
class BasicSimdImplementation<arch::CpuFeature::AVX512BW>:
    public BasicSimdImplementation<arch::CpuFeature::AVX512F> 
{

};

__SIMD_STL_NUMERIC_NAMESPACE_END