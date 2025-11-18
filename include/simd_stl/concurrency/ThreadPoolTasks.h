#pragma once 

#include <src/simd_stl/concurrency/WindowsThreadPoolTasks.h>


__SIMD_STL_CONCURRENCY_NAMESPACE_BEGIN

class _ParallelForEach  {
	sizetype _chunksCount;
	sizetype _chunkSize = 0;
public:
	void __stdcall runChuncked() noexcept {

	}

	static void __stdcall threadPoolCallback(
		PTP_CALLBACK_INSTANCE	instance,
		PVOID					args,
		PTP_WORK				work) noexcept
	{
		reinterpret_cast<_ParallelForEach*>(args)->runChuncked();
	}

	sizetype chunksCount() const noexcept {
		return _chunksCount;
	}

	sizetype chunkSize() const noexcept {
		return _chunkSize;
	}
private:

};

class _ParallelFind {
public:

};

namespace task_type {
	struct for_each {
		using type = _ParallelForEach;
	};

	struct find {
		using type = _ParallelFind;
	};
}

template <class _Type_>
constexpr inline bool is_task_type_v = false;

template <>
constexpr inline bool is_task_type_v<task_type::find> = true;

template <>
constexpr inline bool is_task_type_v<task_type::for_each> = true;

__SIMD_STL_CONCURRENCY_NAMESPACE_END
