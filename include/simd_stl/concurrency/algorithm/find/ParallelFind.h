#pragma once 

#include <simd_stl/algorithm/find/Find.h>
#include <simd_stl/concurrency/ThreadPool.h>

#include <mutex>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

constexpr inline auto _OversubmissionMultiplier = 4;
constexpr inline auto _OversubscriptionMultiplier = 32;

enum class _CancellationToken : uint8 {
	_Running,
	_Cancelling
};

template <class _Iterator_> 
struct _Range {
	_Iterator_ first;
	_Iterator_ last;
};

//template <
//	class _UncheckedIterator_,
//	class _FindFunction_>
//class _ParallelFindTask {
//	_VerifyUnchecked(_UncheckedIterator_);
//
//	_Range<_UncheckedIterator_> _range;
//	_FindFunction_				_function;
//public:
//	_ParallelFindTask(
//		_Range<_UncheckedIterator_> range,
//		_FindFunction_&&			function
//	) noexcept:
//		_range(range),
//		_function(function)
//	{}
//
//	template <class _Type_ = type_traits::IteratorValueType<_UncheckedIterator_>>
//	auto apply(const _Type_& value) noexcept -> decltype(type_traits::invoke(_function, _range.first, _range.last, _value)) {
//		return type_traits::invoke(_function, _range.first, _range.last, _value);
//	}
//};

template <class _Work>
void _Run_available_chunked_work(_Work& _Operation) {
	while (_Operation.processChunk() == _CancellationToken::_Running) { // process while there are chunks remaining
	}
}

template <class _Work>
void _Run_chunked_parallel_work(_Work& _Operation) {
	concurrency::_ThreadPoolExecutor _Work_op{ _Operation };

	_Work_op.submitForChunks(_Operation.partitions());
	_Run_available_chunked_work(_Operation);
}

template <
	class _UncheckedIterator_,
	class _FindFunction_,
	class _Type_ = type_traits::iterator_value_type<_UncheckedIterator_>>
class _ParallelFindChunked {
	_VerifyUnchecked(_UncheckedIterator_);

	std::atomic<uint32> _processedPartitions = 0;

	std::vector<_Range<_UncheckedIterator_>> _partitions;
	std::atomic<_CancellationToken> _cancellationToken;

	std::pair<std::atomic<uint32>, std::atomic<_UncheckedIterator_>> _result;
	std::mutex _mutex;

	_UncheckedIterator_ _last;

	sizetype _partitionSize = 0;
	sizetype _remaining = 0;

	const _Type_& _value;
	_FindFunction_ _function;
public:
	_ParallelFindChunked(
		uint32				hardwareThreads,
		_UncheckedIterator_ firstUnwrapped,
		_UncheckedIterator_	lastUnwrapped,
		_FindFunction_&&	function,
		const _Type_&		value) noexcept:
			_last(lastUnwrapped),
			_result(-1, lastUnwrapped),
			_value(value),
			_function(std::move(function))
	{
		const auto length	= distance(firstUnwrapped, lastUnwrapped);
		const auto division = std::div(length, hardwareThreads);

		if (length > hardwareThreads) {
			_partitionSize = division.quot;
			_remaining = division.rem;
			_partitions.reserve(hardwareThreads);
		}
		else {
			_partitionSize = 1;
			_partitions.reserve(length);
		}

		for (uint32 current = 0; current < _partitions.capacity(); ++current)
			_partitions.emplace_back(
				firstUnwrapped + _partitionSize * current,
				firstUnwrapped + _partitionSize * current + _partitionSize
			);
	}
	

	_CancellationToken processChunk() noexcept {
		const auto currentPartitionNumber	= _processedPartitions.load(std::memory_order_relaxed);
		
		const auto resultIterator			= _result.second.load(std::memory_order_relaxed);
		const auto resultPartitionNumber	= _result.first.load(std::memory_order_relaxed);

		if (resultIterator != _last && resultPartitionNumber < currentPartitionNumber)
			return _CancellationToken::_Cancelling;

		const auto partition	= _partitions[currentPartitionNumber];
		const auto findResult	= type_traits::invoke(_function, partition.first, partition.last, _value);

		if (findResult != partition.last) {
			std::lock_guard guard(_mutex);

			_result.first.store(currentPartitionNumber, std::memory_order_release);
			_result.second.store(findResult, std::memory_order_release);

			return _CancellationToken::_Running;
		}

		_processedPartitions.fetch_add(1, std::memory_order_acq_rel);
		
		if (_processedPartitions == _partitions.size())
			return _CancellationToken::_Cancelling;

		return _CancellationToken::_Running;
	}

	static void __stdcall threadPoolCallback(
		PTP_CALLBACK_INSTANCE, PVOID args, PTP_WORK) noexcept
	{
		_Run_available_chunked_work(*static_cast<_ParallelFindChunked*>(args));
	}

	simd_stl_always_inline uint32 partitions() const noexcept {
		return _partitions.size();
	}

	simd_stl_always_inline _UncheckedIterator_ result() const noexcept {
		return _result.second.load(std::memory_order_relaxed);
	}
};

template <
	class _ExecutionPolicy_,
	class _UncheckedIterator_,
	class _FindFunction_,
	class _Type_>
simd_stl_nodiscard _UncheckedIterator_ _ParallelFind(
	_ExecutionPolicy_&&,
	_UncheckedIterator_			firstUnwrapped,
	const _UncheckedIterator_	lastUnwrapped,
	_FindFunction_&&			function,
	const _Type_&				value) noexcept
{
	_VerifyUnchecked(_UncheckedIterator_);

	if constexpr (std::remove_reference_t<_ExecutionPolicy_>::parallelize) {
		const auto hardwareThreads = arch::ProcessorInformation::hardwareConcurrency();

		if (hardwareThreads > 1) {
			_ParallelFindChunked<_UncheckedIterator_, _FindFunction_> work {
				hardwareThreads, firstUnwrapped, lastUnwrapped,
				type_traits::passFunction(function), value
			};

			// auto submissions	= (std::min)(hardwareThreads * _OversubmissionMultiplier, work.partitions());
			_Run_chunked_parallel_work(work);



			return work.result();
		}
	}

	return function(firstUnwrapped, lastUnwrapped, value);
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
	
	__seek_possibly_wrapped_iterator(first, _ParallelFind(std::forward<_ExecutionPolicy_>(policy),
		firstUnwrapped, lastUnwrapped, &_FindUnchecked<decltype(firstUnwrapped)>, value));

	return first;
}


__SIMD_STL_ALGORITHM_NAMESPACE_END
