#pragma once 

#include <simd_stl/algorithm/find/Find.h>
#include <simd_stl/concurrency/ThreadPool.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

constexpr inline auto _OversubmissionMultiplier = 4;
constexpr inline auto _OversubscriptionMultiplier = 32;


enum class _CancellationToken : uint8 {
	_Running,
	_Cancelling
};

template <
	class _UnwrappedIterator_, 
	class _FindFunction_>
class _ParallelFindTask {
	_UnwrappedIterator_ _first;
	_UnwrappedIterator_ _last;
	
	_FindFunction_		_function;
	const type_traits::IteratorValueType<_UnwrappedIterator_>& _value;

	std::atomic<_UnwrappedIterator_*> _outputIterator;
public:
	template <class _Type_ = type_traits::IteratorValueType<_UnwrappedIterator_>>
	_ParallelFindTask(
		_UnwrappedIterator_		first,
		_UnwrappedIterator_		last,
		_FindFunction_&&		function,
		const _Type_&			value,
		_UnwrappedIterator_&	iterator
	) noexcept:
		_first(first),
		_last(last),
		_function(function),
		_value(value),
		_outputIterator(iterator)
	{}

	void apply() noexcept {
		_outputIterator.store(_function(_first, _last, _value), std::memory_order_relaxed);
	}
};

template <
	class _UnwrappedIterator_,
	class _FindFunction_>
class _ParallelFindChunked {
	std::vector<_ParallelFindTask<_UnwrappedIterator_, _FindFunction_>> _tasks;
	std::atomic<_CancellationToken> _cancellationToken;

	uint32 _current = 0;

	_UnwrappedIterator_ _result;

	sizetype _chunkSize = 0;
	uint32 _length = 0;
public:
	template <class _Type_ = type_traits::IteratorValueType<_UnwrappedIterator_>>
	_ParallelFindChunked(
		uint32				hardwareThreads,
		_UnwrappedIterator_ firstUnwrapped,
		_UnwrappedIterator_	lastUnwrapped,
		_FindFunction_&&	function,
		const _Type_&		value) noexcept:
			_result(lastUnwrapped)
	{
		_length = (lastUnwrapped - firstUnwrapped);

		if (_length > hardwareThreads) {
			_chunkSize = _length / hardwareThreads;
			_tasks.reserve(_length / _chunkSize);
		}
		else {
			_chunkSize = 1;
			_tasks.reserve(_length);
		}

		for (uint32 current = 0; current < _tasks.size(); ++current)
			_tasks[current] = _ParallelFindTask(
				firstUnwrapped + _chunkSize * current,
				firstUnwrapped + _chunkSize * current + _chunkSize, function, value, _result);
	}
	

	void __stdcall processChunk() noexcept {
		if (_cancellationToken.load(std::memory_order_relaxed) == _CancellationToken::_Cancelling)
			return;
		
		_task[_current].apply();
		++_current;
	}

	static void __stdcall threadPoolCallback(
		PTP_CALLBACK_INSTANCE, PVOID args, PTP_WORK) noexcept
	{
		reinterpret_cast<_ParallelFindChunked*>(args)->processChunk();
	}

	simd_stl_always_inline uint32 chunks() const noexcept {
		return _tasks.size();
	}

	simd_stl_always_inline bool end() const noexcept {
		return _result != _lastUnwrapped;
	}
};

template <
	class _ExecutionPolicy_,
	class _UnwrappedIterator_,
	class _FindFunction_,
	class _Type_>
simd_stl_nodiscard _UnwrappedIterator_ _ParallelFind(
	_ExecutionPolicy_&&,
	_UnwrappedIterator_			firstUnwrapped,
	const _UnwrappedIterator_	lastUnwrapped,
	_FindFunction_&&			function,
	const _Type_&				value) noexcept
{
	if constexpr (_ExecutionPolicy_::parallelize) {
		const auto hardwareThreads = arch::ProcessorInformation::hardwareConcurrency();
		if (hardwareThreads > 1) {
			_ParallelFindChunked<_UnwrappedIterator_, _FindFunction_> work { hardwareThreads, firstUnwrapped, lastUnwrapped, type_traits::passFunction(function), value };
			
			concurrency::_ThreadPoolExecutor executor(work);
			
			const auto submissions = (std::min)(hardwareThreads * _OversubmissionMultiplier, work.chunks());

			for (uint32 current = 0; current < submissions; ++current)
				executor.submit();

			executor.join();
		}
	}

	return function(firstUnwrapped, lastUnwrapped);
}

template <
	class _ExecutionPolicy_,
	class _Iterator_,
	class _Type_,
	concurrency::enable_if_execution_policy<_ExecutionPolicy_>>
simd_stl_nodiscard _Iterator_ find(
	_ExecutionPolicy_&&	policy,
	_Iterator_			first,
	const _Iterator_	last,
	const _Type_&		value) noexcept
{
	__verifyRange(first, last); 

	auto firstUnwrapped			= _UnwrapIterator(first);
	const auto lastUnwrapped	= _UnwrapIterator(last);
	
	_SeekPossiblyWrappedIterator(first, _ParallelFind(std::forward<_ExecutionPolicy_>(policy),
		firstUnwrapped, lastUnwrapped, &_FindUnchecked<decltype(firstUnwrapped)>, value));

	return first;
}


__SIMD_STL_ALGORITHM_NAMESPACE_END
