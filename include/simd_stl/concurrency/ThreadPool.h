#pragma once 

#include <simd_stl/arch/ProcessorInformation.h>

#if defined(simd_stl_os_windows)
#  include <src/simd_stl/concurrency/WindowsThreadPool.h>
#endif // defined(simd_stl_os_windows)


__SIMD_STL_CONCURRENCY_NAMESPACE_BEGIN

class thread_pool {
#if defined(simd_stl_os_windows)
	using implementation = WindowsThreadPool;
#endif // defined(simd_stl_os_windows)
public:
	thread_pool(sizetype threads = arch::ProcessorInformation::hardwareConcurrency()) noexcept;
	~thread_pool() noexcept;

	template <class _Task_>
	void submit(_Task_&& task) noexcept;

	template <
		class		_Task_,
		class		_FirstArgument_,
		class ...	_Args_>
	void submit(
		_Task_&&			task,
		_FirstArgument_&&	firstArgument,
		_Args_&& ...		args);

	template <
		class		_TaskType_,
		class		_Task_,
		class		_FirstArgument_,
		class ...	_Args_>
	void parallelize(
		_Task_&&			task,
		_FirstArgument_&&	firstArgument,
		_Args_&& ...		args);

	simd_stl_nodiscard bool wait_all() noexcept;
private:
	_ThreadPool*		_pool = nullptr;
	_ThreadPoolWork*	_work = nullptr;
};

thread_pool::thread_pool(sizetype threads) noexcept {
	_pool = implementation::create();
	implementation::setThreadsCount(_pool, threads);
}

thread_pool::~thread_pool() noexcept {
	implementation::close(_pool);
}

template <class _Task_>
void thread_pool::submit(_Task_&& task) noexcept {
	
}

template <
	class		_Task_,
	class		_FirstArgument_,
	class ...	_Args_>
void thread_pool::submit(
	_Task_&&			task,
	_FirstArgument_&&	firstArgument,
	_Args_&& ...		args) 
{

}

template <
	class		_TaskType_,
	class		_Task_,
	class		_FirstArgument_,
	class ...	_Args_>
void thread_pool::parallelize(
	_Task_&&			task,
	_FirstArgument_&&	firstArgument,
	_Args_&& ...		args)
{

}

bool thread_pool::wait_all() noexcept {

}

__SIMD_STL_CONCURRENCY_NAMESPACE_END
