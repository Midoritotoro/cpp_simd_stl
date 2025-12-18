#pragma once 

#include <src/simd_stl/algorithm/vectorized/minmax/MaxElementVectorized.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <class _UnwrappedInputIterator_>
_Simd_nodiscard_inline_constexpr _UnwrappedInputIterator_ _MaxElementUnchecked(
	_UnwrappedInputIterator_ _FirstUnwrapped,
	_UnwrappedInputIterator_ _LastUnwrapped) noexcept
{
	using _ValueType = type_traits::IteratorValueType<_UnwrappedInputIterator_>;

	if constexpr (type_traits::is_vectorized_find_algorithm_safe_v<_UnwrappedInputIterator_, _ValueType>)
	{
#if simd_stl_has_cxx20
		if (type_traits::is_constant_evaluated() == false)
#endif // simd_stl_has_cxx20
		{ 
			const auto _FirstAdress = std::to_address(_FirstUnwrapped);
			const auto _Position = _MaxElementVectorized<_ValueType>(_FirstAdress, std::to_address(_LastUnwrapped));

			if constexpr (std::is_pointer_v<_UnwrappedInputIterator_>)
				return _Position;
			else
				return _FirstUnwrapped + (_Position - _FirstAdress);
		}
	}

	if (_FirstUnwrapped == _LastUnwrapped)
		return _LastUnwrapped;

	auto _Max = _FirstUnwrapped;

	for (; ++_FirstUnwrapped != _LastUnwrapped; )
		if (*_FirstUnwrapped > *_Max)
			_Max = _FirstUnwrapped;

	return _Max;
}

template <
	class _UnwrappedInputIterator_,
	class _Predicate_>
_Simd_nodiscard_inline_constexpr _UnwrappedInputIterator_ _MaxElementUnchecked(
	_UnwrappedInputIterator_	_FirstUnwrapped,
	_UnwrappedInputIterator_	_LastUnwrapped,
	_Predicate_					_Predicate) noexcept
{
	using _ValueType = type_traits::IteratorValueType<_UnwrappedInputIterator_>;

	if constexpr (type_traits::is_vectorized_find_algorithm_safe_v<_UnwrappedInputIterator_, _ValueType>)
	{
#if simd_stl_has_cxx20
		if (type_traits::is_constant_evaluated() == false)
#endif // simd_stl_has_cxx20
		{ 
			const auto _FirstAdress = std::to_address(_FirstUnwrapped);
			const auto _Position = _MaxElementVectorized<_ValueType>(_FirstAdress, std::to_address(_LastUnwrapped));

			if constexpr (std::is_pointer_v<_UnwrappedInputIterator_>)
				return _Position;
			else
				return _FirstUnwrapped + (_Position - _FirstAdress);
		}
	}

	if (_FirstUnwrapped == _LastUnwrapped)
		return _LastUnwrapped;

	auto _Max = _FirstUnwrapped;

	for (; ++_FirstUnwrapped != _LastUnwrapped; )
		if (_Predicate(_Max, _FirstUnwrapped))
			_Max = _FirstUnwrapped;

	return _Max;
}


__SIMD_STL_ALGORITHM_NAMESPACE_END
