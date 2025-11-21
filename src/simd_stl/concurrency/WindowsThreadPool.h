#pragma once

#include <simd_stl/Types.h>

#include <simd_stl/compatibility/Compatibility.h>
#include <src/simd_stl/utility/Assert.h>

#if defined(simd_stl_os_windows)
#  include <Windows.h>
#  include <threadpoolapiset.h>


__SIMD_STL_CONCURRENCY_NAMESPACE_BEGIN

using _ThreadPoolWork	= TP_WORK;

class WindowsThreadPool {
public:
	template <class _Task_> 
	static _ThreadPoolWork* createWork(_Task_&& task) noexcept {
		return CreateThreadpoolWork(&_Task_::threadPoolCallback, reinterpret_cast<PVOID>(&task), nullptr);
	}

	static void closeWork(_ThreadPoolWork* work) noexcept {
		CloseThreadpoolWork(work);
	}

	static void submit(_ThreadPoolWork* work) noexcept {
		SubmitThreadpoolWork(work);
	}

	static void waitFor(_ThreadPoolWork* work) noexcept {
		WaitForThreadpoolWorkCallbacks(work, true);
	}
};

__SIMD_STL_CONCURRENCY_NAMESPACE_END

#endif // defined(simd_stl_os_windows)
