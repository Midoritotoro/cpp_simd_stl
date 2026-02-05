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
__simd_nodiscard_inline __make_tail_mask_return_type<_DataparType_> make_tail_mask(uint32 __bytes) noexcept
	requires(__is_valid_basic_simd_v<std::remove_cvref_t<_DataparType_>>)
{
	using _RawDataparType = std::remove_cvref_t<_DataparType_>;
	return __simd_make_tail_mask<_RawDataparType::__generation, typename _RawDataparType::policy_type,
			typename _RawDataparType::value_type>(__bytes);
}


template <class _DataparType_>
__simd_nodiscard_inline _DataparType_ abs(const _DataparType_& __datapar) noexcept
	requires(__is_valid_basic_simd_v<std::remove_cvref_t<_DataparType_>>)
{
	using _RawDataparType = std::remove_cvref_t<_DataparType_>;
	return __simd_abs<_RawDataparType::__generation, typename _RawDataparType::policy_type,
		typename _RawDataparType::value_type>(__simd_unwrap(__datapar));
}


template <class _DataparType_>
__simd_nodiscard_inline typename _DataparType_::value_type horizontal_min(const _DataparType_& __datapar) noexcept
	requires(__is_valid_basic_simd_v<std::remove_cvref_t<_DataparType_>>) 
{
	using _RawDataparType = std::remove_cvref_t<_DataparType_>;
	return __simd_horizontal_min<_RawDataparType::__generation, typename _RawDataparType::policy_type,
		typename _RawDataparType::value_type>(__simd_unwrap(__datapar));
}

template <class _DataparType_>
__simd_nodiscard_inline typename _DataparType_::value_type horizontal_max(const _DataparType_& __datapar) noexcept
	requires(__is_valid_basic_simd_v<std::remove_cvref_t<_DataparType_>>) 
{
	using _RawDataparType = std::remove_cvref_t<_DataparType_>;
	return __simd_horizontal_max<_RawDataparType::__generation, typename _RawDataparType::policy_type,
		typename _RawDataparType::value_type>(__simd_unwrap(__datapar));
}

template <class _DataparType_>
__simd_nodiscard_inline _DataparType_ vertical_min(
	const _DataparType_& __first, 
	const _DataparType_& __second) noexcept
		requires(__is_valid_basic_simd_v<std::remove_cvref_t<_DataparType_>>) 
{
	using _RawDataparType = std::remove_cvref_t<_DataparType_>;
	return __simd_vertical_min<_RawDataparType::__generation, typename _RawDataparType::policy_type,
		typename _RawDataparType::value_type>(__simd_unwrap(__first), __simd_unwrap(__second));
}

template <class _DataparType_>
__simd_nodiscard_inline _DataparType_ vertical_max(
	const _DataparType_& __first,
	const _DataparType_& __second) noexcept
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
simd_stl_always_inline __zero_upper_at_exit_guard<_SimdGeneration_> make_guard() noexcept {
	return __zero_upper_at_exit_guard<_SimdGeneration_>();
}

template <class _DataparType_>
simd_stl_always_inline __zero_upper_at_exit_guard<std::remove_cvref_t<_DataparType_>::__generation> make_guard() noexcept
	requires(__is_valid_basic_simd_v<std::remove_cvref_t<_DataparType_>>) 
{
	return __zero_upper_at_exit_guard<std::remove_cvref_t<_DataparType_>::__generation>();
}

template <
	class _DataparType_,
	class _MaskType_>
__simd_nodiscard_inline _DataparType_ blend(
	const _DataparType_&	__first,
	const _DataparType_&	__second,
	const _MaskType_&		__mask) noexcept
		requires(__is_valid_basic_simd_v<std::remove_cvref_t<_DataparType_>> &&
			(__is_valid_basic_simd_v<std::remove_cvref_t<_MaskType_>> || __is_simd_mask_v<std::remove_cvref_t<_MaskType_>>))
{
	using _RawDataparType = std::remove_cvref_t<_DataparType_>;
	return __simd_blend<_RawDataparType::__generation, typename _RawDataparType::policy_type,
		typename _RawDataparType::value_type>(__simd_unwrap(__first), __simd_unwrap(__second), __simd_unwrap_mask(__mask));
}

template <class _DataparType_>
__simd_nodiscard_inline _DataparType_ reverse(const _DataparType_& __datapar) noexcept
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
__simd_nodiscard_inline std::pair<uint32, _DataparType_> compress(
	const _DataparType_&	__datapar, 
	const _MaskType_&		__mask) noexcept
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
simd_stl_always_inline typename _DataparType_::value_type* compress_store(
	void*					__address,
	const _DataparType_&	__datapar,
	const _MaskType_&		__mask,
	_AlignmentPolicy_&&		__policy = _AlignmentPolicy_{}) noexcept
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
__simd_nodiscard_inline _DataparType_ non_temporal_load(const void* __address) noexcept
	requires(__is_valid_basic_simd_v<std::remove_cvref_t<_DataparType_>>)
{
	using _RawDataparType = std::remove_cvref_t<_DataparType_>;
	return __simd_non_temporal_load<_RawDataparType::__generation, typename _RawDataparType::policy_type, typename _RawDataparType::vector_type>(__address);
}

template <class _DataparType_>
simd_stl_always_inline void non_temporal_store(
	void*					__address,
	const _DataparType_&	__datapar) noexcept
		requires(__is_valid_basic_simd_v<std::remove_cvref_t<_DataparType_>>)
{
	using _RawDataparType = std::remove_cvref_t<_DataparType_>;
	__simd_non_temporal_store<_RawDataparType::__generation, typename _RawDataparType::policy_type>(__address, __simd_unwrap(__datapar));
}

template <
	class _DataparType_, 
	class _AlignmentPolicy_ = unaligned_policy>
simd_stl_always_inline _DataparType_ load(
	const void*			__address,
	_AlignmentPolicy_&& __policy = _AlignmentPolicy_{}) noexcept
		requires(__is_valid_basic_simd_v<std::remove_cvref_t<_DataparType_>>)
{
	using _RawDataparType = std::remove_cvref_t<_DataparType_>;
	return __simd_load<_RawDataparType::__generation, typename _RawDataparType::policy_type, typename _RawDataparType::vector_type>(__address, __policy);
}

template <
	class _DataparType_, 
	class _AlignmentPolicy_ = unaligned_policy>
simd_stl_always_inline void store(
	void*					__address,
	const _DataparType_&	__datapar,
	_AlignmentPolicy_&&		__policy = _AlignmentPolicy_{}) noexcept
		requires(__is_valid_basic_simd_v<std::remove_cvref_t<_DataparType_>>)
{
	using _RawDataparType = std::remove_cvref_t<_DataparType_>;
	return __simd_store<_RawDataparType::__generation, typename _RawDataparType::policy_type, typename _RawDataparType::vector_type>(__address, __simd_unwrap(__datapar), __policy);
}

template <
	class _DataparType_,
	class _MaskType_,
	class _AlignmentPolicy_ = unaligned_policy>
__simd_nodiscard_inline _DataparType_ maskz_load(
	const void*				__address,
	const _MaskType_&		__mask,
	_AlignmentPolicy_&&		__policy = _AlignmentPolicy_{}) noexcept
		requires(__is_valid_basic_simd_v<std::remove_cvref_t<_DataparType_>> &&
			(__is_valid_basic_simd_v<std::remove_cvref_t<_MaskType_>> || __is_simd_mask_v<std::remove_cvref_t<_MaskType_>>))
{
	using _RawDataparType = std::remove_cvref_t<_DataparType_>;
	return __simd_mask_load<_RawDataparType::__generation, typename _RawDataparType::policy_type, typename _RawDataparType::value_type, typename _RawDataparType::vector_type>(
		reinterpret_cast<const typename _RawDataparType::value_type*>(__address), __simd_unwrap_mask(__mask), __policy);
}

template <
	class _DataparType_,
	class _MaskType_,
	class _AlignmentPolicy_ = unaligned_policy>
__simd_nodiscard_inline _DataparType_ mask_load(
	const void*					__address,
	const _MaskType_&			__mask,
	const _DataparType_&		__additional_source,
	_AlignmentPolicy_&&			__policy = _AlignmentPolicy_{}) noexcept
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
simd_stl_always_inline void mask_store(
	void*					__address,
	const _DataparType_&	__datapar,
	const _MaskType_&		__mask,
	_AlignmentPolicy_&&		__policy = _AlignmentPolicy_{}) noexcept
		requires(__is_valid_basic_simd_v<std::remove_cvref_t<_DataparType_>> &&
			(__is_valid_basic_simd_v<std::remove_cvref_t<_MaskType_>> || __is_simd_mask_v<std::remove_cvref_t<_MaskType_>>))
{ 
	using _RawDataparType = std::remove_cvref_t<_DataparType_>;
	__simd_mask_store<_RawDataparType::__generation, typename _RawDataparType::policy_type, typename _RawDataparType::value_type>(
		reinterpret_cast<typename _RawDataparType::value_type*>(__address), __simd_unwrap_mask(__mask), __simd_unwrap(__datapar), __policy);
}

template <class _DataparType_>
__simd_nodiscard_inline auto reduce_equal(
	const _DataparType_& __first,
	const _DataparType_& __second) noexcept
{
	constexpr auto __is_native_compare_return_number = std::is_integral_v<datapar::__simd_native_compare_return_type<_DataparType_,
		typename _DataparType_::value_type, datapar::simd_comparison::equal>>;

	if constexpr (!__is_native_compare_return_number) {
		auto __zeros = _DataparType_();
		__zeros.clear();

		const auto __compared = (__first == __second) | as_simd;
		const auto __count_vector = __zeros - __compared;

		return reduce(__count_vector, type_traits::plus<>{});
	}
	else {
		const auto __mask = (__first == __second) | as_index_mask;
		return reduce(__mask, type_traits::plus<>{});
	}
}

template <
	class _DataparType_,
	class _MaskType_>
__simd_nodiscard_inline auto reduce_equal(
	const _DataparType_&	__first,
	const _DataparType_&	__second,
	const _MaskType_&		__tail_mask) noexcept
{
	constexpr auto __is_native_compare_return_number = std::is_integral_v<datapar::__simd_native_compare_return_type<_DataparType_,
		typename _DataparType_::value_type, datapar::simd_comparison::equal>>;

	if constexpr (!__is_native_compare_return_number) {
		auto __zeros = _DataparType_();
		__zeros.clear();

		const auto __compared = ((__first == __second) & __tail_mask) | as_simd;
		return reduce(__zeros - __compared, type_traits::plus<>{});
	}
	else {
		const auto __mask = ((__first == __second) & __tail_mask) | as_index_mask;
		return reduce(__mask, type_traits::plus<>{});
	}
}

__SIMD_STL_DATAPAR_NAMESPACE_END
