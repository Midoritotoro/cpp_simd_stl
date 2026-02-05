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

template <class _DataparType_> 
__simd_nodiscard_inline auto make_tail_mask(uint32 __bytes) noexcept -> __make_tail_mask_return_type<_DataparType_>
	requires(__is_valid_basic_simd_v<std::remove_cvref_t<_DataparType_>>)
{
	using _RawDataparType = std::remove_cvref_t<_DataparType_>;
	return __simd_make_tail_mask<_RawDataparType::__generation, typename _RawDataparType::policy_type,
			typename _RawDataparType::value_type>(__bytes);
}


template <class _DataparType_>
__simd_nodiscard_inline auto abs(const _DataparType_& __datapar) noexcept -> _DataparType_
	requires(__is_valid_basic_simd_v<std::remove_cvref_t<_DataparType_>>)
{
	using _RawDataparType = std::remove_cvref_t<_DataparType_>;
	return __simd_abs<_RawDataparType::__generation, typename _RawDataparType::policy_type,
		typename _RawDataparType::value_type>(__simd_unwrap(__datapar));
}


template <class _DataparType_>
__simd_nodiscard_inline auto horizontal_min(const _DataparType_& __datapar) noexcept -> typename _DataparType_::value_type
	requires(__is_valid_basic_simd_v<std::remove_cvref_t<_DataparType_>>) 
{
	using _RawDataparType = std::remove_cvref_t<_DataparType_>;
	return __simd_horizontal_min<_RawDataparType::__generation, typename _RawDataparType::policy_type,
		typename _RawDataparType::value_type>(__simd_unwrap(__datapar));
}

template <class _DataparType_>
__simd_nodiscard_inline auto horizontal_max(const _DataparType_& __datapar) noexcept -> typename _DataparType_::value_type
	requires(__is_valid_basic_simd_v<std::remove_cvref_t<_DataparType_>>) 
{
	using _RawDataparType = std::remove_cvref_t<_DataparType_>;
	return __simd_horizontal_max<_RawDataparType::__generation, typename _RawDataparType::policy_type,
		typename _RawDataparType::value_type>(__simd_unwrap(__datapar));
}

template <class _DataparType_>
__simd_nodiscard_inline auto vertical_min(
	const _DataparType_& __first, 
	const _DataparType_& __second) noexcept -> _DataparType_
		requires(__is_valid_basic_simd_v<std::remove_cvref_t<_DataparType_>>) 
{
	using _RawDataparType = std::remove_cvref_t<_DataparType_>;
	return __simd_vertical_min<_RawDataparType::__generation, typename _RawDataparType::policy_type,
		typename _RawDataparType::value_type>(__simd_unwrap(__first), __simd_unwrap(__second));
}

template <class _DataparType_>
__simd_nodiscard_inline auto vertical_max(
	const _DataparType_& __first,
	const _DataparType_& __second) noexcept -> _DataparType_
		requires(__is_valid_basic_simd_v<std::remove_cvref_t<_DataparType_>>) 
{
	using _RawDataparType = std::remove_cvref_t<_DataparType_>;
	return __simd_vertical_max<_RawDataparType::__generation, typename _RawDataparType::policy_type,
		typename _RawDataparType::value_type>(__simd_unwrap(__first), __simd_unwrap(__second));
}

template <arch::CpuFeature _SimdGeneration_>
struct __zero_upper_at_exit_guard {
    __zero_upper_at_exit_guard(const __zero_upper_at_exit_guard&) noexcept = delete;
    __zero_upper_at_exit_guard(__zero_upper_at_exit_guard&&) noexcept = delete;

	__zero_upper_at_exit_guard() noexcept
	{}

	~__zero_upper_at_exit_guard() noexcept {
		if constexpr (type_traits::__is_zeroupper_required_v<_SimdGeneration_>)
			_mm256_zeroupper();
	}
};

template <arch::CpuFeature _SimdGeneration_>
simd_stl_always_inline auto make_guard() noexcept -> __zero_upper_at_exit_guard<_SimdGeneration_> {
	return __zero_upper_at_exit_guard<_SimdGeneration_>();
}

template <class _DataparType_>
simd_stl_always_inline auto make_guard() noexcept -> __zero_upper_at_exit_guard<std::remove_cvref_t<_DataparType_>::__generation>
	requires(__is_valid_basic_simd_v<std::remove_cvref_t<_DataparType_>>) 
{
	return __zero_upper_at_exit_guard<std::remove_cvref_t<_DataparType_>::__generation>();
}

template <
	class _DataparType_,
	class _MaskType_>
__simd_nodiscard_inline auto blend(
	const _DataparType_&	__first,
	const _DataparType_&	__second,
	const _MaskType_&		__mask) noexcept -> _DataparType_
		requires(__is_valid_basic_simd_v<std::remove_cvref_t<_DataparType_>> &&
			(__is_valid_basic_simd_v<std::remove_cvref_t<_MaskType_>> || __is_simd_mask_v<std::remove_cvref_t<_MaskType_>>))
{
	using _RawDataparType = std::remove_cvref_t<_DataparType_>;
	return __simd_blend<_RawDataparType::__generation, typename _RawDataparType::policy_type,
		typename _RawDataparType::value_type>(__simd_unwrap(__first), __simd_unwrap(__second), __simd_unwrap_mask(__mask));
}

template <class _DataparType_>
__simd_nodiscard_inline auto reverse(const _DataparType_& __datapar) noexcept -> _DataparType_
	requires(__is_valid_basic_simd_v<std::remove_cvref_t<_DataparType_>>)
{
	using _RawDataparType = std::remove_cvref_t<_DataparType_>;
	return __simd_reverse<_RawDataparType::__generation, typename _RawDataparType::policy_type,
		typename _RawDataparType::value_type>(__simd_unwrap(__datapar));
}

template <class _DataparType_>
__simd_nodiscard_inline void streaming_fence() noexcept
	requires(__is_valid_basic_simd_v<std::remove_cvref_t<_DataparType_>>)
{
	return __simd_streaming_fence<std::remove_cvref_t<_DataparType_>::__generation>();
}

template <arch::CpuFeature _SimdGeneration_>
__simd_nodiscard_inline void streaming_fence() noexcept {
	return __simd_streaming_fence<_SimdGeneration_>();
}

template <
	class _DataparType_,
	class _MaskType_>
__simd_nodiscard_inline auto compress(
	const _DataparType_&	__datapar, 
	const _MaskType_&		__mask) noexcept -> std::pair<uint32, _DataparType_>
		requires(__is_valid_basic_simd_v<std::remove_cvref_t<_DataparType_>> &&
			(__is_valid_basic_simd_v<std::remove_cvref_t<_MaskType_>> || __is_simd_mask_v<std::remove_cvref_t<_MaskType_>>))
{
	using _RawDataparType = std::remove_cvref_t<_DataparType_>;
	return __simd_compress<_RawDataparType::__generation, typename _RawDataparType::policy_type,
		typename _RawDataparType::value_type>(__simd_unwrap(__datapar), __simd_unwrap_mask(__mask));
}

template <
	class _DataparType_,
    class _MaskType_,
    class _AlignmentPolicy_ = unaligned_policy>
simd_stl_always_inline auto compress_store(
	void*					__address,
	const _DataparType_&	__datapar,
	const _MaskType_&		__mask,
	_AlignmentPolicy_&&		__policy = _AlignmentPolicy_{}) noexcept -> typename _DataparType_::value_type*
		requires(__is_valid_basic_simd_v<std::remove_cvref_t<_DataparType_>> &&
			(__is_valid_basic_simd_v<std::remove_cvref_t<_MaskType_>> || __is_simd_mask_v<std::remove_cvref_t<_MaskType_>>))
{
	using _RawDataparType = std::remove_cvref_t<_DataparType_>;
	return __simd_compress_store<_RawDataparType::__generation, typename _RawDataparType::policy_type, typename _RawDataparType::value_type>(
		reinterpret_cast<typename _RawDataparType::value_type*>(__address),
		__simd_to_mask<_RawDataparType::__generation, typename _RawDataparType::policy_type, 
			typename _RawDataparType::value_type>(__simd_unwrap_mask(__mask)), __simd_unwrap(__datapar), __policy);
}

template <class _DataparType_>
__simd_nodiscard_inline auto non_temporal_load(const void* __address) noexcept -> _DataparType_
	requires(__is_valid_basic_simd_v<std::remove_cvref_t<_DataparType_>>)
{
	using _RawDataparType = std::remove_cvref_t<_DataparType_>;
	return __simd_non_temporal_load<_RawDataparType::__generation, typename _RawDataparType::policy_type, typename _RawDataparType::vector_type>(__address);
}

template <class _DataparType_>
simd_stl_always_inline auto non_temporal_store(
	void*					__address,
	const _DataparType_&	__datapar) noexcept -> void
		requires(__is_valid_basic_simd_v<std::remove_cvref_t<_DataparType_>>)
{
	using _RawDataparType = std::remove_cvref_t<_DataparType_>;
	__simd_non_temporal_store<_RawDataparType::__generation, typename _RawDataparType::policy_type>(__address, __simd_unwrap(__datapar));
}

template <
	class _DataparType_, 
	class _AlignmentPolicy_ = unaligned_policy>
simd_stl_always_inline auto load(
	const void*			__address,
	_AlignmentPolicy_&& __policy = _AlignmentPolicy_{}) noexcept -> _DataparType_
		requires(__is_valid_basic_simd_v<std::remove_cvref_t<_DataparType_>>)
{
	using _RawDataparType = std::remove_cvref_t<_DataparType_>;
	return __simd_load<_RawDataparType::__generation, typename _RawDataparType::policy_type, typename _RawDataparType::vector_type>(__address, __policy);
}

template <
	class _DataparType_, 
	class _AlignmentPolicy_ = unaligned_policy>
simd_stl_always_inline auto store(
	void*					__address,
	const _DataparType_&	__datapar,
	_AlignmentPolicy_&&		__policy = _AlignmentPolicy_{}) noexcept -> void
		requires(__is_valid_basic_simd_v<std::remove_cvref_t<_DataparType_>>)
{
	using _RawDataparType = std::remove_cvref_t<_DataparType_>;
	return __simd_store<_RawDataparType::__generation, typename _RawDataparType::policy_type, typename _RawDataparType::vector_type>(__address, __simd_unwrap(__datapar), __policy);
}

template <
	class _DataparType_,
	class _MaskType_,
	class _AlignmentPolicy_ = unaligned_policy>
__simd_nodiscard_inline auto mask_load(
	const void*				__address,
	const _DataparType_&	__datapar,
	const _MaskType_&		__mask,
	_AlignmentPolicy_&&		__policy = _AlignmentPolicy_{}) noexcept -> _DataparType_
		requires(__is_valid_basic_simd_v<std::remove_cvref_t<_DataparType_>> &&
			(__is_valid_basic_simd_v<std::remove_cvref_t<_MaskType_>> || __is_simd_mask_v<std::remove_cvref_t<_MaskType_>>))
{
	using _RawDataparType = std::remove_cvref_t<_DataparType_>;
	return __simd_mask_load<_RawDataparType::__generation, typename _RawDataparType::policy_type, typename _RawDataparType::value_type, typename _RawDataparType::vector_type>(
		reinterpret_cast<const _DesiredType_*>(__address), __simd_unwrap_mask(__mask), __policy);
}

template <
	class _DataparType_,			
	class _MaskType_,
	class _AlignmentPolicy_ = unaligned_policy>
__simd_nodiscard_inline auto mask_load(
	const void*				__address,
	const _MaskType_&		__mask,
	const _DataparType_&	__additional_source,
	_AlignmentPolicy_&&		__policy = _AlignmentPolicy_{}) noexcept -> _DataparType_
		requires(__is_valid_basic_simd_v<std::remove_cvref_t<_DataparType_>> &&
			(__is_valid_basic_simd_v<std::remove_cvref_t<_MaskType_>> || __is_simd_mask_v<std::remove_cvref_t<_MaskType_>>))
{
	using _RawDataparType = std::remove_cvref_t<_DataparType_>;
	return __simd_mask_load<_RawDataparType::__generation, typename _RawDataparType::policy_type, typename _RawDataparType::value_type>(
		reinterpret_cast<const typename _RawDataparType::value_type*>(__address), __simd_unwrap_mask(__mask), __simd_unwrap(__additional_source), __policy);
}

template <
	class _DataparType_,	
	class _MaskType_,
	class _AlignmentPolicy_ = unaligned_policy>
simd_stl_always_inline auto mask_store(
	void*					__address,
	const _DataparType_&	__datapar,
	const _MaskType_&		__mask,
	_AlignmentPolicy_&&		__policy = _AlignmentPolicy_{}) noexcept -> void
		requires(__is_valid_basic_simd_v<std::remove_cvref_t<_DataparType_>> &&
			(__is_valid_basic_simd_v<std::remove_cvref_t<_MaskType_>> || __is_simd_mask_v<std::remove_cvref_t<_MaskType_>>))
{ 
	using _RawDataparType = std::remove_cvref_t<_DataparType_>;
	__simd_mask_store<_RawDataparType::__generation, typename _RawDataparType::policy_type, typename _RawDataparType::value_type>(
		reinterpret_cast<typename _RawDataparType::value_type*>(__address), __simd_unwrap_mask(__mask), _vector, __policy);
}

__SIMD_STL_DATAPAR_NAMESPACE_END
