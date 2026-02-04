#pragma once 

#include <simd_stl/datapar/Simd.h>


__SIMD_STL_DATAPAR_NAMESPACE_BEGIN

template <
	class _DataparType_,
	class _ReduceBinaryFunction_ = type_traits::plus<>>
__simd_nodiscard_inline auto reduce(
	const _DataparType_&		__datapar,
	_ReduceBinaryFunction_&&	__reduce = _ReduceBinaryFunction_{}) noexcept requires (
		(__is_valid_basic_simd_v<std::remove_cvref_t<_DataparType_>> ||
			__is_intrin_type_v<std::remove_cvref_t<_DataparType_>>) &&
		std::is_invocable_r_v<std::remove_cvref_t<_DataparType_>, _ReduceBinaryFunction_,
			std::remove_cvref_t<_DataparType_>, std::remove_cvref_t<_DataparType_>>
		)
{
	using _RawDataparType	= std::remove_cvref_t<_DataparType_>;
	using _RawReductionType = std::remove_cvref_t<_ReduceBinaryFunction_>;

	if constexpr (type_traits::is_any_of_v<_RawReductionType, std::plus<>, type_traits::plus<>>)
		return __simd_reduce<_RawDataparType::__generation, typename _RawDataparType::policy_type, 
			typename _RawDataparType::value_type>(__simd_unwrap(__datapar));
	else
		return __simd_horizontal_fold<_RawDataparType::__generation, typename _RawDataparType::policy_type,
			typename _RawDataparType::value_type>(__simd_unwrap(__datapar), type_traits::__pass_function(__reduce));
}

template <
	class _DataparType_,
	class _ReduceBinaryFunction_ = type_traits::plus<>>
__simd_nodiscard_inline auto reduce(
	const _DataparType_& __datapar,
	_ReduceBinaryFunction_&& __unused = _ReduceBinaryFunction_{}) noexcept requires (
		__is_simd_mask_v<std::remove_cvref_t<_DataparType_>> ||
		__is_simd_index_mask_v<std::remove_cvref_t<_DataparType_>>
	)
{
	return __datapar.count_set();
}

template <
    class _DataparType_,
    class _ReduceBinaryFunction_ = type_traits::plus<>>
__simd_nodiscard_inline auto reduce(
    const _DataparType_& __datapar,
	_ReduceBinaryFunction_&& __unused = _ReduceBinaryFunction_{}) noexcept requires (
        std::is_integral_v<std::remove_cvref_t<_DataparType_>> &&
        std::is_unsigned_v<std::remove_cvref_t<_DataparType_>>
    )
{
    return math::__population_count(__datapar);
}



__SIMD_STL_DATAPAR_NAMESPACE_END
