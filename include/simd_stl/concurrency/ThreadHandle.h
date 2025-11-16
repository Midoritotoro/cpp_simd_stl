#pragma once 

#include <simd_stl/system/Handle.h>
#include <simd_stl/algorithm/swap/Swap.h>


__SIMD_STL_CONCURRENCY_NAMESPACE_BEGIN

#if defined(simd_stl_os_windows)
	enum Priority: int32 {
		IdlePriority = THREAD_PRIORITY_IDLE,

		LowestPriority = THREAD_PRIORITY_LOWEST,
		LowPriority = THREAD_PRIORITY_BELOW_NORMAL,

		NormalPriority = THREAD_PRIORITY_NORMAL,
		HighPriority = THREAD_PRIORITY_ABOVE_NORMAL,

		HighestPriority = THREAD_PRIORITY_HIGHEST,
		TimeCriticalPriority = THREAD_PRIORITY_TIME_CRITICAL,
	};
#endif // defined(simd_stl_os_windows)

class thread_handle: public system::handle {
public:
	thread_handle() noexcept {}
	~thread_handle() noexcept {}

	thread_handle(
		native_handle_type	handle,
		bool				autoDelete,
		deleter_type		deleter = CloseHandle) noexcept:	
			system::handle(handle, autoDelete, deleter)
	{}

	simd_stl_always_inline thread_handle& operator=(const thread_handle& other) noexcept {
		_nativeHandle = other._nativeHandle;
		return *this;
	}

	simd_stl_always_inline thread_handle& operator=(const native_handle_type other) noexcept {
		_nativeHandle = other;
		return *this;
	}
};

__SIMD_STL_CONCURRENCY_NAMESPACE_END

__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <>
void swap<concurrency::thread_handle>(
	concurrency::thread_handle& first,
	concurrency::thread_handle& second) noexcept
{
	concurrency::thread_handle temp = first;
	temp.setAutoDelete(false);

	first = std::move(second);
	second = std::move(temp);
}

__SIMD_STL_ALGORITHM_NAMESPACE_END
