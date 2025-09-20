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
template <
    sizetype    _VectorLength_, 
    uint8 ...   _Indices_>
class basic_simd_shuffle_mask;

template <uint8 ... _Indices_>
class basic_simd_shuffle_mask<2, _Indices_...> {
public:
    using mask_type = uint8;

    static constexpr mask_type unwrap() noexcept {
        return _mask;
    }
private:
    static constexpr mask_type compute() noexcept {
        static_assert(sizeof...(_Indices_) == 2, "Shuffle mask for length 2 must have exactly 2 indices.");
        constexpr uint8 indices[] = { _Indices_... };

        return (indices[0] << 2) | (indices[1]);
    }

    static constexpr mask_type _mask = compute();
};

template <uint32 A, uint32 B, uint32 C, uint32 D>
class basic_simd_shuffle_mask<4, A, B, C, D> {
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
    uint32 A, uint32 B, uint32 C, uint32 D,
    uint32 E, uint32 F, uint32 G, uint32 H>
class basic_simd_shuffle_mask<8, A, B, C, D, E, F, G, H> {
public:
    using mask_type = uint32;

    static constexpr mask_type unwrap() noexcept {
        return _mask;
    }

private:
    static constexpr mask_type compute() noexcept {
        return (A << 21) | (B << 18) | (C << 15) | (D << 12) 
            | (E << 9) | (F << 6) | (G << 3) | H;
    }

    static constexpr mask_type _mask = compute();
};

template <
    uint32 A, uint32 B, uint32 C, uint32 D,
    uint32 E, uint32 F, uint32 G, uint32 H,
    uint32 I, uint32 J, uint32 K, uint32 L,
    uint32 M, uint32 N, uint32 O, uint32 P>
class basic_simd_shuffle_mask<16, A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P> {
public:
    using mask_type = uint64;

    static constexpr mask_type unwrap() noexcept {
        return _mask;
    }

private:
    static constexpr mask_type compute() noexcept {
        return (A << 45) | (B << 42) | (C << 39) | (D << 36)
            | (E << 33) | (F << 30) | (G << 27) | (H << 24)
            | (I << 21) | (J << 18) | (K << 15) | (L << 12)
            | (M << 9) | (N << 6) | (O << 3) | P;
    }

    static constexpr mask_type _mask = compute();
};


__SIMD_STL_NUMERIC_NAMESPACE_END

