#pragma once

#include <simd_stl/Types.h>

#include <simd_stl/compatibility/Compatibility.h>
#include <src/simd_stl/utility/Assert.h>

#if defined(simd_stl_os_windows)
#  include <Windows.h>
#  include <threadpoolapiset.h>


__SIMD_STL_CONCURRENCY_NAMESPACE_BEGIN

using _ThreadPool		= TP_POOL;
using _ThreadPoolWork	= TP_WORK;
using _WorkCallback_	= void (__stdcall *)(PTP_CALLBACK_INSTANCE, PVOID, PTP_WORK);


class WindowsThreadPool {
public:
	static void setThreadsCount(
		_ThreadPool*	pool, 
		dword_t			count) noexcept
	{
		setMinimumThreadsCount(pool, count);
		setMaximumThreadsCount(pool, count);
	}

	static void setMinimumThreadsCount(
		_ThreadPool*	pool, 
		dword_t			minimum) noexcept
	{
		DebugAssert(pool != nullptr);
		SetThreadpoolThreadMinimum(pool, minimum);
	}

	static void setMaximumThreadsCount(
		_ThreadPool*	pool,
		dword_t			maximum) noexcept
	{
		DebugAssert(pool != nullptr);
		SetThreadpoolThreadMaximum(pool, maximum);
	}

	static _ThreadPool* create() noexcept {
		return CreateThreadpool(nullptr);
	}
	
	template <
		class		_Task_,
		class ...	_Args_>
	static _ThreadPoolWork* createWork(
		_Task_&&		task,
		_Args_&& ...	args) noexcept 
	{

	}

	static void close(_ThreadPool* pool) noexcept {
		CloseThreadpool(pool);
	}
};

__SIMD_STL_CONCURRENCY_NAMESPACE_END

#endif // defined(simd_stl_os_windows)
