#pragma once 

#include <src/simd_stl/algorithm/vectorized/minmax/MaxVectorized.h>
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
	return _MaxVectorized<_Type_>(_InitializerList.begin(), _InitializerList.end());
}

template <
	class _Type_,
	class _Predicate_>
_Simd_nodiscard_inline _Type_ max(
	std::initializer_list<_Type_>	_InitializerList,
	_Predicate_						_Predicate) noexcept
{
	if (_InitializerList.size() == 0)
        return *_InitializerList.begin();
 
	auto _Current = _InitializerList.begin();
    auto _Maximum = _Current;
 
	const auto _Last = _InitializerList.end();

    while (++_Current != _Last)
        if (_Predicate(*_Current, *_Maximum))
			_Maximum = _Current;
 
    return _Maximum;
}

__SIMD_STL_ALGORITHM_NAMESPACE_END
