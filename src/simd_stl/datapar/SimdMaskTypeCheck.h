#pragma once 

#include <src/simd_stl/type_traits/TypeTraits.h>

__SIMD_STL_DATAPAR_NAMESPACE_BEGIN

template <
	arch::CpuFeature	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_ = datapar::__default_register_policy<_SimdGeneration_>>
class simd_index_mask;

template <
	class _SimdMask_, 
	class = void>
struct __is_simd_index_mask :
	std::false_type
{};

template <class _SimdMask_>
struct __is_simd_index_mask<
	_SimdMask_,
    std::void_t<simd_index_mask<
        _SimdMask_::__generation,
        typename _SimdMask_::element_type,
        typename _SimdMask_::policy_type>>>
    : std::bool_constant<
        type_traits::is_virtual_base_of_v<
            simd_index_mask<
				_SimdMask_::__generation,
                typename _SimdMask_::element_type,
                typename _SimdMask_::policy_type>,
            _SimdMask_> ||
        std::is_same_v<
            simd_index_mask<
				_SimdMask_::__generation,
				typename _SimdMask_::element_type,
				typename _SimdMask_::policy_type>,
            _SimdMask_>> 
{};

template <class _SimdMask_>
constexpr bool __is_simd_index_mask_v = __is_simd_index_mask<_SimdMask_>::value;

template <
	arch::CpuFeature	_SimdGeneration_,
	typename			_Element_,
	class				_RegisterPolicy_ = datapar::__default_register_policy<_SimdGeneration_>>
class simd_mask;

template <
	class _SimdMask_, 
	class = void>
struct __is_simd_mask :
	std::false_type
{};

template <class _SimdMask_>
struct __is_simd_mask<
	_SimdMask_,
    std::void_t<simd_mask<_SimdMask_::__generation,
                typename _SimdMask_::element_type,
                typename _SimdMask_::policy_type>>>
    : std::bool_constant<
        type_traits::is_virtual_base_of_v<
            simd_mask<_SimdMask_::__generation,
                typename _SimdMask_::element_type,
                typename _SimdMask_::policy_type>,
            _SimdMask_> ||
        std::is_same_v<
            simd_mask<_SimdMask_::__generation,
				typename _SimdMask_::element_type,
				typename _SimdMask_::policy_type>,
            _SimdMask_>> 
{};

template <class _SimdMask_>
constexpr bool __is_simd_mask_v = __is_simd_mask<_SimdMask_>::value;

__SIMD_STL_DATAPAR_NAMESPACE_END