#pragma once 

#include <simd_stl/math/BitMath.h>
#include <array>

__SIMD_STL_NUMERIC_NAMESPACE_BEGIN

/**
 * @class basic_simd_permute_mask
 * @brief Предоставляет интерфейс для создания битовой маски перестановки элементов вектора.
 *
 * @tparam _VectorLength_ Количество элементов вектора.
 * @tparam _Indices_ Индексы перестановки.
 */
template <uint8 ... _Indices_>
class basic_simd_permute_mask {
	static constexpr std::array<uint8, sizeof(_Indices_)> indices = { _Indices... }
};

//template <uint8 A, uint8 B>
//class basic_simd_permute_mask<A, B> {
//public:
//    static constexpr std::array<uint8, 2> indices = { A, B };
//};
//
//template <uint8 A, uint8 B, uint8 C, uint8 D>
//class basic_simd_permute_mask<A, B, C, D> {
//public:
//    static constexpr std::array<uint8, 4> indices = { A, B, C, D };
//};
//
//template <
//    uint8 A, uint8 B, uint8 C, uint8 D,
//    uint8 E, uint8 F, uint8 G, uint8 H>
//class basic_simd_permute_mask<A, B, C, D, E, F, G, H> {
//public:
//    static constexpr std::array<uint8, 8> indices = { A, B, C, D, E, F, G, H };
//};
//
//template <
//    uint8 A, uint8 B, uint8 C, uint8 D,
//    uint8 E, uint8 F, uint8 G, uint8 H,
//    uint8 I, uint8 J, uint8 K, uint8 L,
//    uint8 M, uint8 N, uint8 O, uint8 P>
//class basic_simd_permute_mask<A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P> {
//public:
//    static constexpr std::array<uint8, 16> indices = { A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P };
//};
//
//template <
//    uint8 A0,  uint8 A1,  uint8 A2,  uint8 A3,
//    uint8 A4,  uint8 A5,  uint8 A6,  uint8 A7,
//    uint8 A8,  uint8 A9,  uint8 A10, uint8 A11,
//    uint8 A12, uint8 A13, uint8 A14, uint8 A15,
//    uint8 A16, uint8 A17, uint8 A18, uint8 A19,
//    uint8 A20, uint8 A21, uint8 A22, uint8 A23,
//    uint8 A24, uint8 A25, uint8 A26, uint8 A27,
//    uint8 A28, uint8 A29, uint8 A30, uint8 A31>
//class basic_simd_permute_mask<
//    A0,  A1,  A2,  A3,  A4,  A5,  A6,  A7,
//    A8,  A9,  A10, A11, A12, A13, A14, A15,
//    A16, A17, A18, A19, A20, A21, A22, A23,
//    A24, A25, A26, A27, A28, A29, A30, A31> {
//public:
//    static constexpr std::array<uint8, 32> indices = {
//        A0,  A1,  A2,  A3,  A4,  A5,  A6,  A7,
//        A8,  A9,  A10, A11, A12, A13, A14, A15,
//        A16, A17, A18, A19, A20, A21, A22, A23,
//        A24, A25, A26, A27, A28, A29, A30, A31
//    };
//};
//
//template <
//    uint8 A0,  uint8 A1,  uint8 A2,  uint8 A3,
//    uint8 A4,  uint8 A5,  uint8 A6,  uint8 A7,
//    uint8 A8,  uint8 A9,  uint8 A10, uint8 A11,
//    uint8 A12, uint8 A13, uint8 A14, uint8 A15,
//    uint8 A16, uint8 A17, uint8 A18, uint8 A19,
//    uint8 A20, uint8 A21, uint8 A22, uint8 A23,
//    uint8 A24, uint8 A25, uint8 A26, uint8 A27,
//    uint8 A28, uint8 A29, uint8 A30, uint8 A31,
//    uint8 A32, uint8 A33, uint8 A34, uint8 A35,
//    uint8 A36, uint8 A37, uint8 A38, uint8 A39,
//    uint8 A40, uint8 A41, uint8 A42, uint8 A43,
//    uint8 A44, uint8 A45, uint8 A46, uint8 A47,
//    uint8 A48, uint8 A49, uint8 A50, uint8 A51,
//    uint8 A52, uint8 A53, uint8 A54, uint8 A55,
//    uint8 A56, uint8 A57, uint8 A58, uint8 A59,
//    uint8 A60, uint8 A61, uint8 A62, uint8 A63>
//class basic_simd_permute_mask<
//    A0,  A1,  A2,  A3,  A4,  A5,  A6,  A7,
//    A8,  A9,  A10, A11, A12, A13, A14, A15,
//    A16, A17, A18, A19, A20, A21, A22, A23,
//    A24, A25, A26, A27, A28, A29, A30, A31,
//    A32, A33, A34, A35, A36, A37, A38, A39,
//    A40, A41, A42, A43, A44, A45, A46, A47,
//    A48, A49, A50, A51, A52, A53, A54, A55,
//    A56, A57, A58, A59, A60, A61, A62, A63> {
//public:
//    static constexpr std::array<uint8, 64> indices = {
//        A0,  A1,  A2,  A3,  A4,  A5,  A6,  A7,
//        A8,  A9,  A10, A11, A12, A13, A14, A15,
//        A16, A17, A18, A19, A20, A21, A22, A23,
//        A24, A25, A26, A27, A28, A29, A30, A31,
//        A32, A33, A34, A35, A36, A37, A38, A39,
//        A40, A41, A42, A43, A44, A45, A46, A47,
//        A48, A49, A50, A51, A52, A53, A54, A55,
//        A56, A57, A58, A59, A60, A61, A62, A63
//    };
//};


__SIMD_STL_NUMERIC_NAMESPACE_END

