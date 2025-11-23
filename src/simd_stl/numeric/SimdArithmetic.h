#pragma once 


#include <src/simd_stl/numeric/SimdElementAccess.h>
#include <src/simd_stl/numeric/SimdDivisors.h>


__SIMD_STL_NUMERIC_NAMESPACE_BEGIN

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_>
class _SimdArithmetic;

template <class _RegisterPolicy_>
class _SimdArithmetic<arch::CpuFeature::SSE2, _RegisterPolicy_> {
   using _Cast_             = _SimdCast<arch::CpuFeature::SSE2, _RegisterPolicy_>;
   using _ElementAccess_    = _SimdElementAccess<arch::CpuFeature::SSE2, _RegisterPolicy_>;
   using _MemoryAccess_     = _SimdMemoryAccess<arch::CpuFeature::SSE2, _RegisterPolicy_>;
public:
    template <
        typename _DesiredType_,
        typename _DesiredOutputType_,
        typename _VectorType_>
    static simd_stl_always_inline _DesiredOutputType_ reduce(_VectorType_ vector) noexcept {
        constexpr auto vectorLength = sizeof(_VectorType_) / sizeof(_DesiredType_);

        _DesiredType_ vectorArray[vectorLength];
        _MemoryAccess_::template storeUnaligned<_DesiredType_>(vectorArray, vector);

        _DesiredOutputType_ out = 0;

        for (auto i = 0; i < vectorLength; ++i)
            out += vectorArray[i];

        return out;
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ shiftRight(
        _VectorType_    vector,
        uint32          shift) noexcept
    {
        if constexpr (is_epi64_v<_DesiredType_> || is_epu64_v<_DesiredType_>)
            return _Cast_::template cast<__m128i, _VectorType_>(_mm_srli_epi64(
                _Cast_::template cast<_VectorType_, __m128i>(vector), shift));

        else if constexpr (is_epi32_v<_DesiredType_> || is_epu32_v<_DesiredType_>)
            return _Cast_::template cast<__m128i, _VectorType_>(_mm_srli_epi32(
                _Cast_::template cast<_VectorType_, __m128i>(vector), shift));

        else if constexpr (is_epi16_v<_DesiredType_> || is_epu16_v<_DesiredType_>)
            return _Cast_::template cast<__m128i, _VectorType_>(_mm_srli_epi16(
                _Cast_::template cast<_VectorType_, __m128i>(vector), shift));

        else if constexpr (is_epi8_v<_DesiredType_> || is_epu8_v<_DesiredType_>) {
            auto evenVector = _mm_slli_epi16(vector, 8);
            evenVector = _mm_sra_epi16(evenVector, _mm_cvtsi32_si128(shift + 8));

            const auto oddVector = _mm_sra_epi16(vector, _mm_cvtsi32_si128(shift));
            const auto mask = _mm_set1_epi32(0x00FF00FF);

            return _mm_or_si128(_mm_and_si128(mask, evenVector), _mm_andnot_si128(mask, oddVector));
        }
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ shiftLeft(
        _VectorType_    vector,
        uint32          shift) noexcept
    {
        if constexpr (is_epi64_v<_DesiredType_> || is_epu64_v<_DesiredType_>)
            return _Cast_::template cast<__m128i, _VectorType_>(_mm_slli_epi64(
                _Cast_::template cast<_VectorType_, __m128i>(vector), shift));

        else if constexpr (is_epi32_v<_DesiredType_> || is_epu32_v<_DesiredType_>)
            return _Cast_::template cast<__m128i, _VectorType_>(_mm_slli_epi32(
                _Cast_::template cast<_VectorType_, __m128i>(vector), shift));

        else if constexpr (is_epi16_v<_DesiredType_> || is_epu16_v<_DesiredType_>)
            return _Cast_::template cast<__m128i, _VectorType_>(_mm_slli_epi16(
                _Cast_::template cast<_VectorType_, __m128i>(vector), shift));

        else if constexpr (is_epi8_v<_DesiredType_> || is_epu8_v<_DesiredType_>) {
            uint32 mask = (uint32)0xFF >> (uint32)shift;
            const auto andMask = _mm_and_si128(_Cast_::template cast<_VectorType_, __m128i>(vector), _mm_set1_epi8((char)mask));

            return _Cast_::template cast<__m128i, _VectorType_>(_mm_sll_epi16(andMask, _mm_cvtsi32_si128(shift)));
        }
    }

    template <
        typename        _DesiredType_,
        _DesiredType_   _Divisor_,
        typename        _VectorType_>
    static simd_stl_always_inline _VectorType_ divideByConst(_VectorType_ dividendVector) noexcept {
        return divideByConstHelper<_DesiredType_, _Divisor_, _VectorType_>(dividendVector);
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ unaryMinus(_VectorType_ vector) noexcept {
        // 0x80000000 == 0b10000000000000000000000000000000

        if constexpr (is_ps_v<_DesiredType_>)
            return _mm_xor_ps(vector, _Cast_::template cast<__m128i, __m128>(
                _mm_set1_epi32(0x80000000)));

        else if constexpr (is_pd_v<_DesiredType_>)
            return _mm_xor_pd(vector, _Cast_::template cast<__m128i, __m128d>(
                _mm_setr_epi32(0, 0x80000000, 0, 0x80000000)));

        else
            return sub<_DesiredType_>(_ElementAccess_::template constructZero<_VectorType_>(), vector);
    }

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ decrement(_VectorType_ vector) noexcept {
        return sub(vector, _ElementAccess_::template broadcast(1));
    }

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ increment(_VectorType_ vector) noexcept {
       return add(vector, _ElementAccess_::template broadcast(1));
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ add(
        _VectorType_ left,
        _VectorType_ right) noexcept
    {
        if constexpr (is_epi64_v<_DesiredType_> || is_epu64_v<_DesiredType_>)
            return _Cast_::template cast<__m128i, _VectorType_>(_mm_add_epi64(
                _Cast_::template cast<_VectorType_, __m128i>(left),
                _Cast_::template cast<_VectorType_, __m128i>(right)));
        else if constexpr (is_epi32_v<_DesiredType_> || is_epu32_v<_DesiredType_>)
            return _Cast_::template cast<__m128i, _VectorType_>(_mm_add_epi32(
                _Cast_::template cast<_VectorType_, __m128i>(left),
                _Cast_::template cast<_VectorType_, __m128i>(right)));
        else if constexpr (is_epi16_v<_DesiredType_> || is_epu16_v<_DesiredType_>)
            return _Cast_::template cast<__m128i, _VectorType_>(_mm_add_epi16(
                _Cast_::template cast<_VectorType_, __m128i>(left),
                _Cast_::template cast<_VectorType_, __m128i>(right)));
        else if constexpr (is_epi8_v<_DesiredType_> || is_epu8_v<_DesiredType_>)
            return _Cast_::template cast<__m128i, _VectorType_>(_mm_add_epi8(
                _Cast_::template cast<_VectorType_, __m128i>(left),
                _Cast_::template cast<_VectorType_, __m128i>(right)));
        else if constexpr (is_ps_v<_DesiredType_>)
            return _Cast_::template cast<__m128, _VectorType_>(_mm_add_ps(
                _Cast_::template cast<_VectorType_, __m128>(left),
                _Cast_::template cast<_VectorType_, __m128>(right)));
        else if constexpr (is_pd_v<_DesiredType_>)
            return _Cast_::template cast<__m128d, _VectorType_>(_mm_add_pd(
                _Cast_::template cast<_VectorType_, __m128d>(left),
                _Cast_::template cast<_VectorType_, __m128d>(right)));
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ sub(
        _VectorType_ left,
        _VectorType_ right) noexcept
    {
        if constexpr (is_epi64_v<_DesiredType_> || is_epu64_v<_DesiredType_>)
            return _Cast_::template cast<__m128i, _VectorType_>(_mm_sub_epi64(
                _Cast_::template cast<_VectorType_, __m128i>(left),
                _Cast_::template cast<_VectorType_, __m128i>(right)));
        else if constexpr (is_epi32_v<_DesiredType_> || is_epu32_v<_DesiredType_>)
            return _Cast_::template cast<__m128i, _VectorType_>(_mm_sub_epi32(
                _Cast_::template cast<_VectorType_, __m128i>(left),
                _Cast_::template cast<_VectorType_, __m128i>(right)));
        else if constexpr (is_epi16_v<_DesiredType_> || is_epu16_v<_DesiredType_>)
            return _Cast_::template cast<__m128i, _VectorType_>(_mm_sub_epi16(
                _Cast_::template cast<_VectorType_, __m128i>(left),
                _Cast_::template cast<_VectorType_, __m128i>(right)));
        else if constexpr (is_epi8_v<_DesiredType_> || is_epu8_v<_DesiredType_>)
            return _Cast_::template cast<__m128i, _VectorType_>(_mm_sub_epi8(
                _Cast_::template cast<_VectorType_, __m128i>(left),
                _Cast_::template cast<_VectorType_, __m128i>(right)));
        else if constexpr (is_ps_v<_DesiredType_>)
            return _Cast_::template cast<__m128, _VectorType_>(_mm_sub_ps(
                _Cast_::template cast<_VectorType_, __m128>(left),
                _Cast_::template cast<_VectorType_, __m128>(right)));
        else if constexpr (is_pd_v<_DesiredType_>)
            return _Cast_::template cast<__m128d, _VectorType_>(_mm_sub_pd(
                _Cast_::template cast<_VectorType_, __m128d>(left),
                _Cast_::template cast<_VectorType_, __m128d>(right)));
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ mul(
        _VectorType_ left,
        _VectorType_ right) noexcept
    {
        // ? 
       if      constexpr (is_epi64_v<_DesiredType_>)
            return _Cast_::template cast<__m128i, _VectorType_>(_mm_mul_epi64(
                _Cast_::template cast<_VectorType_, __m128i>(left),
                _Cast_::template cast<_VectorType_, __m128i>(right)));
        else if constexpr (is_epu64_v<_DesiredType_>)
            return _Cast_::template cast<__m128i, _VectorType_>(_mm_mul_epu64(
                _Cast_::template cast<_VectorType_, __m128i>(left),
                _Cast_::template cast<_VectorType_, __m128i>(right)));
        else if constexpr (is_epi32_v<_DesiredType_>)
            return _Cast_::template cast<__m128i, _VectorType_>(_mm_mul_epi32(
                _Cast_::template cast<_VectorType_, __m128i>(left),
                _Cast_::template cast<_VectorType_, __m128i>(right)));
        else if constexpr (is_epu32_v<_DesiredType_>)
            return _Cast_::template cast<__m128i, _VectorType_>(_mm_mul_epu32(
                _Cast_::template cast<_VectorType_, __m128i>(left),
                _Cast_::template cast<_VectorType_, __m128i>(right)));
        else if constexpr (is_epi16_v<_DesiredType_>)
            return _Cast_::template cast<__m128i, _VectorType_>(_mm_mul_epi16(
                _Cast_::template cast<_VectorType_, __m128i>(left),
                _Cast_::template cast<_VectorType_, __m128i>(right)));
        else if constexpr (is_epu16_v<_DesiredType_>)
            return _Cast_::template cast<__m128i, _VectorType_>(_mm_mul_epu16(
                _Cast_::template cast<_VectorType_, __m128i>(left),
                _Cast_::template cast<_VectorType_, __m128i>(right)));
        else if constexpr (is_epi8_v<_DesiredType_>)
            return _Cast_::template cast<__m128i, _VectorType_>(_mm_mul_epi8(
                _Cast_::template cast<_VectorType_, __m128i>(left),
                _Cast_::template cast<_VectorType_, __m128i>(right)));
        else if constexpr (is_epu8_v<_DesiredType_>)
            return _Cast_::template cast<__m128i, _VectorType_>(_mm_mul_epi8(
                _Cast_::template cast<_VectorType_, __m128i>(left),
                _Cast_::template cast<_VectorType_, __m128i>(right)));
        else if constexpr (is_ps_v<_DesiredType_>)
            return _mm_mul_ps(left, right);
        /*else if constexpr (is_pd_v<_DesiredType_>)
            return _mm_mul_pd(left, right);*/
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ div(
        _VectorType_ left,
        _VectorType_ right) noexcept
    {
        if constexpr (is_pd_v<_DesiredType_>)
            return _Cast_::template cast<__m128d, _VectorType_>(_mm_div_pd(
                _Cast_::template cast<_VectorType_, __m128d>(left),
                _Cast_::template cast<_VectorType_, __m128d>(right)));
        else if constexpr (is_ps_v<_DesiredType_>)
            return _Cast_::template cast<__m128, _VectorType_>(_mm_div_ps(
                _Cast_::template cast<_VectorType_, __m128>(left),
                _Cast_::template cast<_VectorType_, __m128>(right)));
        else if constexpr (is_epi32_v<_DesiredType_>) {

        }
    }

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ bitwiseNot(_VectorType_ vector) noexcept {
        if      constexpr (std::is_same_v<_VectorType_, __m128d>)
            return _mm_xor_pd(vector, _mm_cmpeq_pd(vector, vector));
        else if constexpr (std::is_same_v<_VectorType_, __m128i>)
            return _mm_xor_si128(vector, _mm_cmpeq_epi32(vector, vector));
        else if constexpr (std::is_same_v<_VectorType_, __m128>)
            return _mm_xor_ps(vector, _mm_cmpeq_ps(vector, vector));
    }

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ bitwiseXor(
        _VectorType_ left,
        _VectorType_ right) noexcept
    {
        if      constexpr (std::is_same_v<_VectorType_, __m128d>)
            return _mm_xor_pd(left, right);
        else if constexpr (std::is_same_v<_VectorType_, __m128i>)
            return _mm_xor_si128(left, right);
        else if constexpr (std::is_same_v<_VectorType_, __m128>)
            return _mm_xor_ps(left, right);
    }

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ bitwiseAnd(
        _VectorType_ left,
        _VectorType_ right) noexcept
    {
        if      constexpr (std::is_same_v<_VectorType_, __m128d>)
            return _mm_and_pd(left, right);
        else if constexpr (std::is_same_v<_VectorType_, __m128i>)
            return _mm_and_si128(left, right);
        else if constexpr (std::is_same_v<_VectorType_, __m128>)
            return _mm_and_ps(left, right);
    }

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ bitwiseOr(
        _VectorType_ left,
        _VectorType_ right) noexcept
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
    static simd_stl_always_inline _VectorType_ divideByConstHelper(_VectorType_ dividendVector) noexcept
    {
        static_assert(_Divisor_ != 0, "Integer division by zero");

        if constexpr (_Divisor_ == 1)
            return dividendVector;

        if constexpr (is_epu32_v<_DesiredType_>) {
            const auto dividendAsInt32 = _Cast_::template cast<_VectorType_, __m128i>(dividendVector);
            constexpr auto trailingZerosInDivisor = math::CountTrailingZeroBits(_Divisor_);

            if constexpr ((uint32(_Divisor_) & (uint32(_Divisor_) - 1)) == 0)
                return _mm_srli_epi32(dividendAsInt32, trailingZerosInDivisor);

            constexpr uint32 magicMultiplier = uint32((uint64(1) << (trailingZerosInDivisor + 32)) / _Divisor_);
            constexpr uint64 magicRemainder = (uint64(1) << (trailingZerosInDivisor + 32)) - uint64(_Divisor_) * magicMultiplier;

            constexpr bool useRoundDown = (2 * magicRemainder < _Divisor_);
            constexpr uint32 adjustedMultiplier = useRoundDown ? magicMultiplier : magicMultiplier + 1;

            const auto multiplierBroadcasted = _ElementAccess_::template broadcast<_VectorType_>(uint64(adjustedMultiplier));

            auto lowProduct = _mm_mul_epu32(dividendAsInt32, multiplierBroadcasted);    // �������� �������� [0] � [2] �� multiplier

            if constexpr (useRoundDown)
                lowProduct = _mm_add_epi64(lowProduct, multiplierBroadcasted);

            auto lowProductShifted = _mm_srli_epi64(lowProduct, 32);                   // �������� ������� 32 ���� ���������� ���������
            auto highParts = _mm_srli_epi64(dividendAsInt32, 32);              // �������� �������� [1] � [3] �� ��������� �������
            auto highProduct = _mm_mul_epu32(highParts, multiplierBroadcasted);  // �������� �������� [1] � [3] �� multiplier

            if constexpr (useRoundDown)
                highProduct = _mm_add_epi64(highProduct, multiplierBroadcasted);

            auto low32BitMask = _mm_set_epi32(-1, 0, -1, 0);
            auto highProductMasked = _mm_and_si128(highProduct, low32BitMask);

            auto combinedProduct = _mm_or_si128(lowProductShifted, highProductMasked);

            return _Cast_::template cast<__m128i, _VectorType_>(_mm_srli_epi32(combinedProduct, trailingZerosInDivisor));
        }
        else if constexpr (is_epi32_v<_DesiredType_>) {
            if constexpr (_Divisor_ == -1)
                return unaryMinus<_DesiredType_>(dividendVector);

            constexpr uint32 absoluteDivisor = _Divisor_ > 0 ? uint32(_Divisor_) : uint32(-_Divisor_);

            if constexpr ((absoluteDivisor & (absoluteDivisor - 1)) == 0) {
                constexpr auto shiftAmount = math::CountLeadingZeroBits(absoluteDivisor);
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

            constexpr int32 shiftForMagic = math::CountLeadingZeroBits(uint32_t(absoluteDivisor) - 1);
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

            return _Cast_::template cast<__m128i, _VectorType_>(_mm_xor_si128(correctedQuotient, divisorParams.getDivisorSign()));
        }
        else if constexpr (is_epi16_v<_DesiredType_>) {
            if constexpr (_Divisor_ == -1)
                return unaryMinus<_DesiredType_>(dividendVector);

            constexpr uint32 absoluteDivisor = _Divisor_ > 0 ? uint32_t(_Divisor_) : uint32_t(-_Divisor_);

            if constexpr ((absoluteDivisor & (absoluteDivisor - 1)) == 0) {
                // �������� � ������� ������
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
                        _ElementAccess_::template broadcast<__m128i>(uint16_t(adjustedMagicMultiplier >> trailingZeroBitCount))
                    ),
                    _mm_andnot_si128(quotientVector, _mm_set1_epi16(trailingZeroBitCount))
                );
            }
            else
                return quotientVector;
        }
    }
};

template <class _RegisterPolicy_>
class _SimdArithmetic<arch::CpuFeature::SSE3, _RegisterPolicy_> :
    public _SimdArithmetic<arch::CpuFeature::SSE2, _RegisterPolicy_>
{};

template <class _RegisterPolicy_>
class _SimdArithmetic<arch::CpuFeature::SSSE3, _RegisterPolicy_> :
    public _SimdArithmetic<arch::CpuFeature::SSE3, _RegisterPolicy_>
{};

template <class _RegisterPolicy_>
class _SimdArithmetic<arch::CpuFeature::SSE41, _RegisterPolicy_> :
    public _SimdArithmetic<arch::CpuFeature::SSSE3, _RegisterPolicy_>
{};

template <class _RegisterPolicy_>
class _SimdArithmetic<arch::CpuFeature::SSE42, _RegisterPolicy_> :
    public _SimdArithmetic<arch::CpuFeature::SSE41, _RegisterPolicy_>
{
    using _MemoryAccess_    = _SimdMemoryAccess<arch::CpuFeature::SSE42, _RegisterPolicy_>;
    using _Cast_            = _SimdCast<arch::CpuFeature::SSE42, _RegisterPolicy_>;
public:
    template <
        typename _DesiredType_, 
        typename _DesiredOutputType_,
        typename _VectorType_> 
    static simd_stl_always_inline _DesiredOutputType_ reduce(_VectorType_ vector) noexcept { 
        if constexpr (is_epi64_v<_DesiredType_> || is_epu64_v<_DesiredType_>) {
    #ifdef simd_stl_processor_x86_32
            return static_cast<_DesiredOutputType_>(_mm_cvtsi128_si32(vector)) +
                    static_cast<_DesiredOutputType_>(_mm_extract_epi32(vector, 2));
    #else
            return static_cast<_DesiredOutputType_>(_mm_cvtsi128_si64(vector)) + 
                static_cast<_DesiredOutputType_>(_mm_extract_epi64(vector, 1));
    #endif // simd_stl_processor_x86_32
        }
        else if constexpr (is_epi32_v<_DesiredType_> || is_epu32_v<_DesiredType_>) { 
            const auto reduce4 = _mm_hadd_epi32(vector, _mm_setzero_si128());    // (0+1),(2+3),0,0
            const auto reduce5 = _mm_hadd_epi32(reduce4, _mm_setzero_si128());   // (0+...+3),0,0,0

            return static_cast<_DesiredOutputType_>(_mm_cvtsi128_si32(reduce5));
        }
        else if constexpr (is_epi16_v<_DesiredType_> || is_epu16_v<_DesiredType_>) { 
            const auto reduce2 = _mm_hadd_epi16(vector, _mm_setzero_si128());
            const auto reduce3 = _mm_unpacklo_epi16(reduce2, _mm_setzero_si128());

            const auto reduce4 = _mm_hadd_epi32(reduce3, _mm_setzero_si128()); // (0+1),(2+3),0,0
            const auto reduce5 = _mm_hadd_epi32(reduce4, _mm_setzero_si128()); // (0+...+3),0,0,0

            return static_cast<_DesiredOutputType_>(_mm_cvtsi128_si32(reduce5));
        }
        else if constexpr (is_epi8_v<_DesiredType_> || is_epu8_v<_DesiredType_>) {
            const auto reduce1 = _mm_sad_epu8(vector, _mm_setzero_si128());

#ifdef simd_stl_processor_x86_32
            return static_cast<_DesiredOutputType_>(_mm_cvtsi128_si32(reduce1)) +
                static_cast<_DesiredOutputType_>(_mm_extract_epi32(reduce1, 2));
#else
            return static_cast<_DesiredOutputType_>(_mm_cvtsi128_si64(reduce1)) +
                static_cast<_DesiredOutputType_>(_mm_extract_epi64(reduce1, 1));
#endif
        }
        else { 
            constexpr auto vectorLength = sizeof(_VectorType_) / sizeof(_DesiredType_);

            _DesiredType_ vectorArray[vectorLength];
            _MemoryAccess_::template storeUnaligned<_DesiredType_>(vectorArray, vector);

            _DesiredOutputType_ out = 0;

            for (auto i = 0; i < vectorLength; ++i)
                out += vectorArray[i];

            return out;
        }
    }

};

__SIMD_STL_NUMERIC_NAMESPACE_END
