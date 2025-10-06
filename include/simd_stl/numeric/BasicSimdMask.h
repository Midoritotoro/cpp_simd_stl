#pragma once 

#include <simd_stl/numeric/BasicSimdMaskImplementation.h>
#include <simd_stl/math/BitMath.h>


__SIMD_STL_NUMERIC_NAMESPACE_BEGIN

template <
	arch::CpuFeature	_SimdGeneration_,
	typename			_Element_>
class basic_simd_mask {
	static_assert(type_traits::__is_generation_supported_v<_SimdGeneration_>);
	static_assert(type_traits::__is_vector_type_supported_v<_Element_>);

	using implementation = BasicSimdMaskImplementation<_SimdGeneration_, _Element_>;
public:
	using mask_type = typename implementation::mask_type;
	using size_type = typename implementation::size_type;

	basic_simd_mask(const mask_type mask) noexcept:
		_mask(mask)
	{}

	/**
	   * @return true, если все биты маски установлены.
    */
	constexpr simd_stl_always_inline bool allOf() const noexcept {
		return implementation::allOf(_mask);
	}

	/**
		* @return true, если хотя бы один бит маски установлен.
	*/
	constexpr simd_stl_always_inline bool anyOf() const noexcept {
		return implementation::anyOf(_mask);
	}

	/**
		* @return true, если ни один бит маски не установлен.
	*/
	constexpr simd_stl_always_inline bool noneOf() const noexcept {
		return implementation::noneOf(_mask);
	}

	/**
		* @return Количество установленных битов маски.
	*/
	constexpr simd_stl_always_inline size_type countSet() const noexcept {
		return implementation::countSet(_mask);
	}

	/**
		* @return Количество конечных нулевых битов в маске.
	*/
	constexpr simd_stl_always_inline size_type countTrailingZeroBits() const noexcept {
		return implementation::countTrailingZeroBits(_mask);
	}

	/**
		* @return Количество ведущих нулевых битов маски.
	*/	
	constexpr simd_stl_always_inline size_type countLeadingZeroBits() const noexcept {
		return implementation::countLeadingZeroBits(_mask);
	}

	/**
		* @return Числовое значение маски.
	*/	
	constexpr simd_stl_always_inline mask_type unwrap() const noexcept {
		return _mask;
	}
private:
	mask_type _mask = 0;
};

//basic_simd_mask& operator>>(const int8 shift) noexcept {

//}

__SIMD_STL_NUMERIC_NAMESPACE_END

