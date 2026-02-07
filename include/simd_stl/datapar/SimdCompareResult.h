#pragma once 

#include <src/simd_stl/datapar/SimdCompare.h>
#include <src/simd_stl/datapar/SimdCompareAdapters.h>

#include <src/simd_stl/type_traits/TypeTraits.h>


__SIMD_STL_DATAPAR_NAMESPACE_BEGIN

template <
	arch::CpuFeature	_SimdGeneration_,
	class				_Element_,
	class				_RegisterPolicy_,
	__simd_comparison	_Comparison_>
class simd_compare_result {
	friend __as_index_mask_t;
	friend __as_mask_t;
	friend __as_native_t;
	friend __as_simd_t;

	using __native_type = __simd_native_compare_return_type<simd<_SimdGeneration_, _Element_, _RegisterPolicy_>, _Element_, _Comparison_>;
public:
	static inline constexpr auto __generation = _SimdGeneration_;
	static inline constexpr auto __comparison = _Comparison_;

	using element_type		= _Element_;
	using register_policy	= _RegisterPolicy_;

	using native_type		= std::conditional_t<std::is_integral_v<__native_type>, simd_mask<_SimdGeneration_, element_type, _RegisterPolicy_>, __native_type>;

	simd_compare_result(const native_type& __result) noexcept;

	simd_compare_result(const simd_compare_result&) = delete;
	simd_compare_result& operator=(const simd_compare_result&) = delete;
	
	simd_compare_result(simd_compare_result&&) noexcept = default; 
	simd_compare_result& operator=(simd_compare_result&&) noexcept = default;

	template <class _Type_>
	simd_stl_always_inline friend simd_compare_result operator& <>(simd_compare_result __compare_result, _Type_ __other) noexcept
		requires std::is_same_v<_Type_, native_type> || std::is_integral_v<_Type_> || __is_simd_mask_v<_Type_>
	{
		return __compare_result._compare_result & __other;
	}

	template <class _Type_>
	simd_stl_always_inline friend simd_compare_result operator| <>(simd_compare_result __compare_result, _Type_ __other) noexcept requires std::is_same_v<_Type_, native_type> || std::is_integral_v<_Type_> {
		return __compare_result._compare_result | __other;
	}
	
	template <class _Type_>
	simd_stl_always_inline friend simd_compare_result operator^ <>(simd_compare_result __compare_result, _Type_ __other) noexcept requires std::is_same_v<_Type_, native_type> || std::is_integral_v<_Type_> {
		return __compare_result._compare_result ^ __other;
	}

	simd_stl_always_inline simd_compare_result operator~() noexcept;
	simd_stl_always_inline operator bool() const noexcept;
private:
	native_type _compare_result;
};

template <
	class _SimdCompareResult_,
	class = void>
struct __is_valid_simd_compare_result :
	std::false_type
{};

template <class _SimdCompareResult_>
struct __is_valid_simd_compare_result<
	_SimdCompareResult_,
    std::void_t<simd_compare_result<
        _SimdCompareResult_::__generation,
        typename _SimdCompareResult_::value_type,
        typename _SimdCompareResult_::policy_type,
		_SimdCompareResult_::__comparison>>>
    : std::bool_constant<
        type_traits::is_virtual_base_of_v<
			simd_compare_result<_SimdCompareResult_::__generation,
                typename _SimdCompareResult_::value_type,
                typename _SimdCompareResult_::policy_type,
				_SimdCompareResult_::__comparison>,
            _SimdCompareResult_> ||
        std::is_same_v<
            simd_compare_result<_SimdCompareResult_::__generation,
				typename _SimdCompareResult_::value_type,
				typename _SimdCompareResult_::policy_type,
				_SimdCompareResult_::__comparison>,
            _SimdCompareResult_>> 
{};

template <class _SimdCompareResult_>
constexpr bool __is_valid_simd_compare_result_v = __is_valid_simd_compare_result<_SimdCompareResult_>::value;

__SIMD_STL_DATAPAR_NAMESPACE_END

#include <src/simd_stl/datapar/SimdCompareResult.inl>
