#pragma once 

#include <simd_stl/math/BitMath.h>
#include <simd_stl/compatibility/SimdCompatibility.h>

#include <array>


__SIMD_STL_NUMERIC_NAMESPACE_BEGIN

/**
 * @class basic_simd_permute_mask
 * @brief Предоставляет интерфейс для создания битовой маски перестановки элементов вектора.
 *
 * @tparam _Indices_ Индексы перестановки.
 */
template <uint8 ... _Indices_>
class basic_simd_permute_mask {
//	static_assert(
//		sizeof...(_Indices_) == 2 || sizeof...(_Indices_) == 4 || sizeof...(_Indices_) == 8 ||
//		sizeof...(_Indices_) == 16 || sizeof...(_Indices_) == 32 || sizeof...(_Indices_) == 64);
//
//	static constexpr std::array<uint8, sizeof...(_Indices_)> indices = { _Indices_... };
//public:
//	template <
//		arch::CpuFeature	_Feature_,
//		typename			_DesiredType_>
//	static simd_stl_always_inline auto unwrap() noexcept {
//		if constexpr (arch::__is_xmm_v<_Feature_>) {
//			if constexpr (sizeof...(_Indices_) == 2)
//				return (indices[0] << 1) | indices[1];
//			else if constexpr (sizeof...(_Indices_) == 4)
//				return (indices[0] << 6) | (indices[1] << 4) | (indices[2] << 2) | indices[3];
//			else if constexpr (sizeof...(_Indices_) == 8)
//				return indices.data();
//			else if constexpr (sizeof....(_Indices_) == 16 && static_cast<uint8>(_Feature_) >= static_cast<uint8>(arch::CpuFeature::SSSE3))
//				return _mm_lddqu_si128(reinterpret_cast<__m128i*>(indices.data()));
//			else if constexpr (sizeof...(_Indices_) == 16 && static_cast<uint8>(_Feature_) < static_cast<uint8>(arch::CpuFeature::SSSE3))
//				return indices.data();
//		}
//	}
};



__SIMD_STL_NUMERIC_NAMESPACE_END

