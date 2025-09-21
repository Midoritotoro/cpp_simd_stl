#pragma once 

#include <src/simd_stl/math/BitMath.h>


__SIMD_STL_NUMERIC_NAMESPACE_BEGIN

/**
 * @class basic_simd_shuffle_mask
 * @brief Предоставляет интерфейс для создания битовой маски перестановки элементов вектора.
 *
 * @tparam _VectorLength_ Количество элементов вектора.
 * @tparam _Indices_ Индексы перестановки.
 */
template <uint8 ... _Indices_>
class basic_simd_shuffle_mask;

template <uint8 A, uint8 B>
class basic_simd_shuffle_mask<A, B> {
public:
    using mask_type = uint8;

    static constexpr mask_type unwrap() noexcept {
        return _mask;
    }
private:
    static constexpr mask_type compute() noexcept {
        return (A << 2) | B;
    }

    static constexpr mask_type _mask = compute();
};

template <uint8 A, uint8 B, uint8 C, uint8 D>
class basic_simd_shuffle_mask<A, B, C, D> {
public:
    using mask_type = uint8;

    static constexpr mask_type unwrap() noexcept {
        return _mask;
    }

private:
    static constexpr mask_type compute() noexcept {
        return (A << 6) | (B << 4) | (C << 2) | D;
    }

    static constexpr mask_type _mask = compute();
};

template <
    uint8 A, uint8 B, uint8 C, uint8 D,
    uint8 E, uint8 F, uint8 G, uint8 H>
class basic_simd_shuffle_mask<A, B, C, D, E, F, G, H> {
public:
    using mask_type = uint32;

    static constexpr mask_type unwrap() noexcept {
        return _mask;
    }

private:
    static constexpr mask_type compute() noexcept {
        return (static_cast<mask_type>(A) << 21) | (static_cast<mask_type>(B) << 18) | (static_cast<mask_type>(C) << 15) 
            | (static_cast<mask_type>(D) << 12) | (static_cast<mask_type>(E) << 9) | (static_cast<mask_type>(F) << 6) |
            (static_cast<mask_type>(G) << 3) | static_cast<mask_type>(H);
    }

    static constexpr mask_type _mask = compute();
};

template <
    uint8 A, uint8 B, uint8 C, uint8 D,
    uint8 E, uint8 F, uint8 G, uint8 H,
    uint8 I, uint8 J, uint8 K, uint8 L,
    uint8 M, uint8 N, uint8 O, uint8 P>
class basic_simd_shuffle_mask<A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P> {
public:
    using mask_type = unsigned long long;

    static constexpr mask_type unwrap() noexcept {
        return _mask;
    }

private:
    static constexpr mask_type compute() noexcept {
        return (static_cast<mask_type>(A) << 60) | (static_cast<mask_type>(B) << 56) | (static_cast<mask_type>(C) << 52) | (static_cast<mask_type>(D) << 48)
            | (static_cast<mask_type>(E) << 44) | (static_cast<mask_type>(F) << 40) | (static_cast<mask_type>(G) << 36) | (static_cast<mask_type>(H) << 32)
            | (static_cast<mask_type>(I) << 28) | (static_cast<mask_type>(J) << 24) | (static_cast<mask_type>(K) << 20) | (static_cast<mask_type>(L) << 16)
            | (static_cast<mask_type>(M) << 12) | (static_cast<mask_type>(N) << 8) | (static_cast<mask_type>(O) << 4) | static_cast<mask_type>(P);
    }

    static constexpr mask_type _mask = compute();
};

__SIMD_STL_NUMERIC_NAMESPACE_END

