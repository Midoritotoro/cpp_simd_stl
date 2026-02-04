#pragma once 


#include <simd_stl/arch/CpuFeature.h>
#include <simd_stl/math/BitMath.h>


__SIMD_STL_DATAPAR_NAMESPACE_BEGIN

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Element_>
class SimdDivisor;

template <>
class SimdDivisor<arch::CpuFeature::SSE2, int32> {
protected:
    __m128i multiplier;
    __m128i firstShift;
    
    __m128i sign;
public:
    SimdDivisor() = default;
    SimdDivisor(int32 d) noexcept {
        set(d);
    }

    SimdDivisor(int32 m, int32 s1, int32 sgn) noexcept {
        multiplier = _mm_set1_epi32(m);
        firstShift = _mm_cvtsi32_si128(s1);

        sign = _mm_set1_epi32(sgn);
    }

    void set(int32 d) noexcept {
        int32 sh, m;
        const int32 d1 = ::abs(d);

        if (uint32(d) == 0x80000000u) {
            m = 0x80000001;
            sh = 30;
        }
        else if (d1 > 1) {
            sh = math::count_leading_zero_bits(uint32(d1 - 1));
            m = int32((int64(1) << (32 + sh)) / d1 - ((int64(1) << 32) - 1));
        }
        else {
            m = 1;
            sh = 0;

            if (d == 0) 
                m /= d;
        }

        multiplier = _mm_set1_epi32(m);
        firstShift = _mm_cvtsi32_si128(sh);

        if (d < 0) 
            sign = _mm_set1_epi32(-1); 
        else
            sign = _mm_set1_epi32(0);
    }

    __m128i getMultiplier() const noexcept {
        return multiplier;
    }

    __m128i getFirstShiftCount() const noexcept {
        return firstShift;
    }

    __m128i getDivisorSign() const noexcept {
        return sign;
    }
};

template <>
class SimdDivisor<arch::CpuFeature::SSE2, uint32> {
protected:
    __m128i multiplier;
    
    __m128i firstShift;
    __m128i secondShift;
public:
    SimdDivisor() = default;
    SimdDivisor(uint32 d) noexcept {
        set(d);
    }

    SimdDivisor(uint32 m, int32 s1, int32 s2) noexcept {
        multiplier = _mm_set1_epi32((int32)m);

        firstShift = _mm_setr_epi32(s1, 0, 0, 0);
        secondShift = _mm_setr_epi32(s2, 0, 0, 0);
    }

    void set(uint32 d) noexcept {
        uint32 L, L2, sh1, sh2, m;
        switch (d) {
        case 0:
            m = sh1 = sh2 = 1 / d;
            break;
        case 1:
            m = 1; sh1 = sh2 = 0;
            break;
        case 2:
            m = 1; sh1 = 1; sh2 = 0;
            break;
        default:
            L = math::count_trailing_zero_bits(d - 1) + 1;
            L2 = uint32(L < 32 ? 1 << L : 0);
        
            m = 1 + uint32((uint64_t(L2 - d) << 32) / d);
            sh1 = 1;  sh2 = L - 1;
        }
        
        multiplier = _mm_set1_epi32((int32)m);
    
        firstShift = _mm_setr_epi32((int32)sh1, 0, 0, 0);
        secondShift = _mm_setr_epi32((int32)sh2, 0, 0, 0);
    }

    __m128i getMultiplier() const noexcept {
        return multiplier;
    }

    __m128i getFirstShiftCount() const noexcept {
        return firstShift;
    }

    __m128i getSecondShiftCount() const noexcept {
        return secondShift;
    }
};

template <>
class SimdDivisor<arch::CpuFeature::SSE2, int16> {
protected:
    __m128i multiplier;
    __m128i firstShift;

    __m128i sign;
public:
    SimdDivisor() = default;
    SimdDivisor(int16 d) noexcept {
        set(d);
    }

    SimdDivisor(int16 m, int s1, int sgn) noexcept {
        multiplier = _mm_set1_epi16(m);
        firstShift = _mm_setr_epi32(s1, 0, 0, 0);

        sign = _mm_set1_epi32(sgn);
    }

    void set(int16 d) noexcept {
        const int32 d1 = ::abs(d);
        int32 sh, m;

        if (uint16(d) == 0x8000u) {
            m = 0x8001;
            sh = 14;
        }
        else if (d1 > 1) {
            sh = (int32)math::count_trailing_zero_bits(uint32(d1 - 1));
            m = ((int32(1) << (16 + sh)) / d1 - ((int32(1) << 16) - 1));
        }
        else {
            m = 1;
            sh = 0;
            if (d == 0) m /= d;
        }

        multiplier = _mm_set1_epi16(int16_t(m));
        firstShift = _mm_setr_epi32(sh, 0, 0, 0);

        sign = _mm_set1_epi32(d < 0 ? -1 : 0);
    }
    
    __m128i getMultiplier() const noexcept {
        return multiplier;
    }

    __m128i getFirstShiftCount() const noexcept {
        return firstShift;
    }

    __m128i getDivisorSign() const noexcept {
        return sign;
    }
};


template <>
class SimdDivisor<arch::CpuFeature::SSE2, uint16> {
protected:
    __m128i multiplier;

    __m128i firstShift;
    __m128i secondShift;
public:
    SimdDivisor() = default;
    SimdDivisor(uint16 d) noexcept {
        set(d);
    }

    SimdDivisor(uint16 m, int s1, int s2) noexcept  {
        multiplier = _mm_set1_epi16((int16)m);

        firstShift = _mm_setr_epi32(s1, 0, 0, 0);
        secondShift = _mm_setr_epi32(s2, 0, 0, 0);
    }

    void set(uint16 d) noexcept {
        uint16 L, L2, sh1, sh2, m;

        switch (d) {

        case 0:
            m = sh1 = sh2 = 1u / d;
            break;

        case 1:
            m = 1; sh1 = sh2 = 0;
            break;

        case 2:
            m = 1; sh1 = 1; sh2 = 0;
            break;

        default:
            L = (uint16)math::count_trailing_zero_bits(d - 1u) + 1u;
            L2 = uint16(1 << L);

            m = 1u + uint16((uint32(L2 - d) << 16) / d);
            sh1 = 1;  sh2 = L - 1u;
        }

        multiplier = _mm_set1_epi16((int16)m);

        firstShift = _mm_setr_epi32((int32)sh1, 0, 0, 0);
        secondShift = _mm_setr_epi32((int32)sh2, 0, 0, 0);
    }

    __m128i getMultiplier() const noexcept {
        return multiplier;
    }

    __m128i getFirstShiftCount() const noexcept {
        return firstShift;
    }

    __m128i getSecondShiftCount() const noexcept {
        return secondShift;
    }
};

__SIMD_STL_DATAPAR_NAMESPACE_END
