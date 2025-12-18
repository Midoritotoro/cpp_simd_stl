#pragma once 

#include <src/simd_stl/algorithm/AlgorithmDebug.h>
#include <src/simd_stl/type_traits/SimdAlgorithmSafety.h>

#include <src/simd_stl/algorithm/MsvcIteratorUnwrap.h>
#include <simd_stl/concurrency/Execution.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <class _Type_>
_Simd_nodiscard_inline _Type_ max(
	const _Type_& _Left,
	const _Type_& _Right) noexcept
{
	return (_Left, _Right) ? _Left : _Right;
}

template <
	class _Type_,
	class _Predicate_>
_Simd_nodiscard_inline _Type_ max(
	const _Type_&	_Left,
	const _Type_&	_Right,
	_Predicate_		_Predicate) noexcept
{
	return _Predicate(_Left, _Right) ? _Right : _Left;
}

template <class _Type_>
_Simd_nodiscard_inline _Type_ max(std::initializer_list<_Type_> _InitializerList) noexcept {

}

template <
	class _Type_,
	class _Predicate_>
_Simd_nodiscard_inline _Type_ max(
	std::initializer_list<_Type_>	_InitializerList,
	_Predicate_						_Predicate) noexcept
{

}

__SIMD_STL_ALGORITHM_NAMESPACE_END
