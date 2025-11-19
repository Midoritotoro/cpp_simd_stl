#pragma once 

#include <simd_stl/arch/ProcessorInformation.h>

#if defined(simd_stl_os_windows)
#  include <src/simd_stl/concurrency/WindowsThreadPool.h>
#endif // defined(simd_stl_os_windows)

#include <simd_stl/concurrency/ThreadPoolTasks.h>


__SIMD_STL_CONCURRENCY_NAMESPACE_BEGIN

class _ThreadPoolExecutor {
#if defined(simd_stl_os_windows)
	using implementation = WindowsThreadPool;
#endif // defined(simd_stl_os_windows)
public:
	class thread_pool_executor;
	using executor_type = thread_pool_executor;

	~_ThreadPoolExecutor() noexcept;

	template <class _Task_>
	_ThreadPoolExecutor(_Task_&& task) noexcept;

	void submit() noexcept;

	template <class _Task_> 
	void post(_Task_&& task) noexcept;

	simd_stl_nodiscard bool join() noexcept;
private:
	_ThreadPoolWork* _work = nullptr;
};

template <class _Task_>
_ThreadPoolExecutor::_ThreadPoolExecutor(_Task_&& task) noexcept {
	post(std::move(task));
}

template <class _Task_>
void _ThreadPoolExecutor::post(_Task_&& task) noexcept {
	_work = implementation::createWork(std::move(task));
}

void _ThreadPoolExecutor::submit() noexcept {
	implementation::submit(_work);
}

_ThreadPoolExecutor::~_ThreadPoolExecutor() noexcept {
	implementation::closeWork(_work);
}

bool _ThreadPoolExecutor::join() noexcept {
	implementation::waitFor(_work);
}

__SIMD_STL_CONCURRENCY_NAMESPACE_END
