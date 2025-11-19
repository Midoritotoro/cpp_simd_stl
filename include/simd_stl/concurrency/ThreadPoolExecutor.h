#pragma once 

#include <simd_stl/concurrency/ThreadPool.h>

__SIMD_STL_CONCURRENCY_NAMESPACE_BEGIN

class thread_pool::thread_pool_executor {
public:
	thread_pool_executor(thread_pool* pool) noexcept;
	~thread_pool_executor() noexcept;
private:
	thread_pool* _pool = nullptr;
};



__SIMD_STL_CONCURRENCY_NAMESPACE_END
