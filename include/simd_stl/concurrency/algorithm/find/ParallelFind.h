#pragma once 

#include <simd_stl/algorithm/find/Find.h>
#include <simd_stl/concurrency/ThreadPool.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <
	class _ExecutionPolicy_,
	class _Iterator_,
	class _FindFunction_>
simd_stl_nodiscard _Iterator_ _ParallelFind(
	_ExecutionPolicy_&&,
	_Iterator_			first,
	const _Iterator_	last,
	_FindFunction_&&	function) noexcept
{
	if constexpr (_ExecutionPolicy_::parallelize) {
		const auto hardwareThreads = arch::ProcessorInformation::hardwareConcurrency();
		if (hardwareThreads > 1) {

		}
	}

	
}

template <
	class _ExecutionPolicy_,
	class _Iterator_,
	class _Type_,
	concurrency::enable_if_execution_policy<_ExecutionPolicy_>>
simd_stl_nodiscard _Iterator_ find(
	_ExecutionPolicy_&&,
	_Iterator_			first,
	const _Iterator_	last,
	const _Type_&		value) noexcept
{

}


__SIMD_STL_ALGORITHM_NAMESPACE_END
