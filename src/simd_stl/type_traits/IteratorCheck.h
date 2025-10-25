#pragma once 

#include <simd_stl/SimdStlNamespace.h>
#include <simd_stl/compatibility/CxxVersionDetection.h>

#include <type_traits>
#include <xutility>


__SIMD_STL_TYPE_TRAITS_NAMESPACE_BEGIN


#if simd_stl_has_cxx20
	template <class _Iterator_>
	using IteratorReferenceType		= std::iter_reference_t<_Iterator_>;

	template <class _Iterator_>
	using IteratorValueType			= std::iter_value_t<_Iterator_>;

	template <class _Iterator_>
	using IteratorDifferenceType	= std::iter_difference_t<_Iterator_>;
#else
	template <class _Iterator_>
	using IteratorReferenceType		= typename std::iterator_traits<_Iterator_>::reference;

	template <class _Iterator_>
	using IteratorValueType			= typename std::iterator_traits<_Iterator_>::value_type;

	template <class _Iterator_>
	using IteratorDifferenceType	= typename std::iterator_traits<_Iterator_>::difference_type;
#endif // simd_stl_has_cxx20


#if simd_stl_has_cxx20
	template <class _Iterator_>
	constexpr bool is_iterator_contiguous_v = std::contiguous_iterator<_Iterator_>;
#else
	template <class _Iterator_>
	constexpr bool is_iterator_contiguous_v = std::is_pointer_v<_Iterator_>;
#endif

template <
	class _FirstIterator_,
	class _SecondIterator_>
constexpr bool are_iterators_contiguous = is_iterator_contiguous_v<_FirstIterator_> 
	&& is_iterator_contiguous_v<_SecondIterator_>;

template <class _Iterator_>
constexpr inline bool is_iterator_volatile_v = std::is_volatile_v<std::remove_reference_t<std::iter_reference_t<_Iterator_>>>;

template <
	class _Type_,
	class = void>
constexpr inline bool is_iterator_v = false;

template <class _Type_>
constexpr inline bool is_iterator_v<_Type_, std::void_t<
	typename std::iterator_traits<_Type_>::iterator_category>> = true;


template <class _Iterator_>
constexpr inline bool is_iterator_input_cxx17_v = std::is_convertible_v<
	typename std::iterator_traits<_Iterator_>::iterator_category, std::input_iterator_tag>;

template <class _Iterator_>
constexpr inline bool is_iterator_input_ranges_v =
#if simd_stl_has_cxx20
    (std::input_iterator<_Iterator_> && std::sentinel_for<_Iterator_, _Iterator_>) ||
#endif // simd_stl_has_cxx20
	is_iterator_input_cxx17_v<_Iterator_>;


template <class _Iterator_>
constexpr inline bool is_iterator_bidirectional_cxx17_v = std::is_convertible_v<
	typename std::iterator_traits<_Iterator_>::iterator_category, std::bidirectional_iterator_tag>;

template <class _Iterator_>
constexpr inline bool is_iterator_bidirectional_ranges_v =
#if simd_stl_has_cxx20
    std::bidirectional_iterator<_Iterator_> ||
#endif // simd_stl_has_cxx20
	is_iterator_bidirectional_cxx17_v<_Iterator_>;


template <class _Iterator_>
constexpr inline bool is_iterator_random_cxx17_v = std::is_convertible_v<
	typename std::iterator_traits<_Iterator_>::iterator_category, std::random_access_iterator_tag>;

template <class _Iterator_>
constexpr inline bool is_iterator_random_ranges_v =
#if simd_stl_has_cxx20
    std::random_access_iterator<_Iterator_> ||
#endif // simd_stl_has_cxx20
	is_iterator_random_cxx17_v<_Iterator_>;


template <class _Iterator_>
constexpr inline bool is_iterator_forward_cxx17_v = std::is_convertible_v<
    typename std::iterator_traits<_Iterator_>::iterator_category, std::forward_iterator_tag>;

template <class _Iterator_>
constexpr inline bool is_iterator_forward_ranges_v =
#if simd_stl_has_cxx20
    std::forward_iterator<_Iterator_> ||
#endif // simd_stl_has_cxx20
	is_iterator_forward_cxx17_v<_Iterator_>;


template <class _Iterator_>
constexpr inline bool is_iterator_parallel_v = is_iterator_forward_ranges_v<_Iterator_>;

#if !defined(__simd_stl_require_parallel_iterator)
#define __simd_stl_require_parallel_iterator(_Iterator_) \
    static_assert(simd_stl::type_traits::is_iterator_parallel_v<_Iterator_>, "Parallel algorithms require forward iterators or stronger.")
#endif // !defined(__simd_stl_require_parallel_iterator)

template <
    class _Iterator_,
    class = void>
constexpr bool _Allow_inheriting_unwrap_v = true;

template <class _Iterator_>
constexpr bool _Allow_inheriting_unwrap_v<_Iterator_, std::void_t<typename _Iterator_::_Prevent_inheriting_unwrap>> =
    std::is_same_v<_Iterator_, typename _Iterator_::_Prevent_inheriting_unwrap>;

template <
    class _Iterator_,
    class _Sentinel_ = _Iterator_,
    class = void>
constexpr bool is_range_verifiable_v = false;

template <
    class _Iterator_, 
    class _Sentinel_>
constexpr bool is_range_verifiable_v<
    _Iterator_, _Sentinel_,
    std::void_t<decltype(_VerifyRange(
        std::declval<const _Iterator_&>(), std::declval<const _Sentinel_&>()))>> = _Allow_inheriting_unwrap_v<_Iterator_>;

template <
    class _Iterator_, 
    class = void>
constexpr bool is_iterator_unwrappable_v = false;

template <class _Iterator_>
constexpr bool is_iterator_unwrappable_v<_Iterator_,
    std::void_t<decltype(std::declval<std::remove_cvref_t<_Iterator_>&>()._Seek_to(std::declval<_Iterator_>()._Unwrapped()))>> =
    _Allow_inheriting_unwrap_v<std::remove_cvref_t<_Iterator_>>;

template <
    class _Iterator_, 
    class = void>
constexpr bool is_nothrow_unwrappable_v = false;

template <class _Iterator_>
constexpr bool is_nothrow_unwrappable_v<_Iterator_, std::void_t<decltype(std::declval<_Iterator_>()._Unwrapped())>> =
    noexcept(std::declval<_Iterator_>()._Unwrapped());

template <
    class _Iterator_,
    class = bool>
constexpr bool can_unwrap_when_unverified_v = false;

template <class _Iterator_>
constexpr bool can_unwrap_when_unverified_v<_Iterator_, decltype(static_cast<bool>(_Iterator_::_Unwrap_when_unverified))> =
    static_cast<bool>(_Iterator_::_Unwrap_when_unverified);

template <class _Iterator_>
constexpr bool is_possibly_unverified_iterator_unwrappable_v =
    type_traits::is_iterator_unwrappable_v<_Iterator_> && can_unwrap_when_unverified_v<std::remove_cvref_t<_Iterator_>>;

template <
    class _Iterator_,
    class = void>
constexpr bool is_offset_verifiable_v = false;

template <class _Iterator_>
constexpr bool is_offset_verifiable_v
    <_Iterator_, std::void_t<decltype(std::declval<const _Iterator_&>()._Verify_offset(type_traits::IteratorDifferenceType<_Iterator_>{}))>> = true;

template <
    class _Iterator_,
    class = void>
constexpr bool is_offset_nothrow_verifiable_v = false;

template <class _Iterator_>
constexpr bool is_offset_nothrow_verifiable_v
    <_Iterator_, std::void_t<decltype(std::declval<const _Iterator_&>()._Verify_offset(type_traits::IteratorDifferenceType<_Iterator_>{}))>> = 
        noexcept(std::declval<const _Iterator_&>()._Verify_offset(type_traits::IteratorDifferenceType<_Iterator_>{}));

template <class _Iterator_>
constexpr bool is_iterator_unwrappable_for_offset_v = 
    type_traits::is_iterator_unwrappable_v<_Iterator_> && is_offset_verifiable_v<std::remove_cvref_t<_Iterator_>>;

template <class _Iterator_>
constexpr bool is_iterator_nothrow_unwrappable_for_offset_v = 
    type_traits::is_nothrow_unwrappable_v<_Iterator_> && is_offset_nothrow_verifiable_v<std::remove_cvref_t<_Iterator_>>;

template <
    class _Iterator_, 
    class _UnwrappedIterator_,
    class = void>
constexpr bool is_wrapped_iterator_seekable_v = false;

template <
    class _Iterator_, 
    class _UnwrappedIterator_>
constexpr bool is_wrapped_iterator_seekable_v
    <_Iterator_, _UnwrappedIterator_, std::void_t<decltype(std::declval<_Iterator_&>()._Seek_to(std::declval<_UnwrappedIterator_>()))>> = true;

template <
    class _Iterator_, 
    class _UnwrappedIterator_,
    class = void>
constexpr bool is_wrapped_iterator_nothrow_seekable_v = false;

template <
    class _Iterator_, 
    class _UnwrappedIterator_>
constexpr bool is_wrapped_iterator_nothrow_seekable_v
    <_Iterator_, _UnwrappedIterator_, std::void_t<decltype(std::declval<_Iterator_&>()._Seek_to(std::declval<_UnwrappedIterator_>()))>> = 
        noexcept(std::declval<_Iterator_&>()._Seek_to(std::declval<_UnwrappedIterator_>()));


__SIMD_STL_TYPE_TRAITS_NAMESPACE_END
