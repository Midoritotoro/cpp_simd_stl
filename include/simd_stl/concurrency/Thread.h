#pragma once 

#include <simd_stl/arch/ProcessorInformation.h>

#if defined(simd_stl_os_win)
#  include <src/simd_stl/concurrency/WindowsThread.h>
#endif // defined(simd_stl_os_win)

#include <src/simd_stl/concurrency/ThreadYield.h>
#include <simd_stl/algorithm/swap/Swap.h>

#include <chrono>


__SIMD_STL_CONCURRENCY_NAMESPACE_BEGIN

class thread {
#if defined(simd_stl_os_win)
	using implementation = WindowsThread;
#endif // defined(simd_stl_os_win)
public:
	using native_handle_type	= typename implementation::native_handle_type;
	using id					= typename implementation::thread_id_wrapper_type;

	simd_stl_nodiscard_constructor thread() noexcept;
	simd_stl_nodiscard_constructor thread(native_handle_type handle) noexcept;
    simd_stl_nodiscard_constructor thread(thread&& other) noexcept;

    template <
        class       _Task_, 
        class...    _Args_>
    simd_stl_nodiscard_constructor thread(
        _Task_&&    task, 
        _Args_&&... args);

    thread& operator=(thread&& other) noexcept;

    ~thread() noexcept;

    void swap(thread& other) noexcept;

    void join();
    void detach();

    simd_stl_nodiscard simd_stl_always_inline bool joinable() const noexcept;

    simd_stl_nodiscard simd_stl_always_inline static uint32 hardware_concurrency() noexcept;
    simd_stl_nodiscard simd_stl_always_inline native_handle_type native_handle() noexcept;
private:
	native_handle_type _handle = nullptr;
	id _id = 0;
};

thread::thread() noexcept {}

thread::thread(thread&& other) noexcept:
    _handle(std::exchange(other._handle, {})),
    _id(std::exchange(other._id, {}))
{}

thread::thread(native_handle_type handle) noexcept :
    _handle(handle)
{
    _id = _ThreadId(handle);
}

void thread::swap(thread& other) noexcept {
    algorithm::swap(_handle, other._handle);
    algorithm::swap(_id, other._id);
}

thread::native_handle_type thread::native_handle() noexcept {
    return _handle;
}

template <
    class       _Task_,
    class...    _Args_>
thread::thread(
    _Task_&&    task,
    _Args_&&... args)
{

}

thread::~thread() noexcept {
    if (joinable())
        std::terminate();
}

uint32 thread::hardware_concurrency() noexcept {
    return arch::ProcessorInformation::hardwareConcurrency();
}

bool thread::joinable() const noexcept {
    return (_id.id() != 0);
}

thread& thread::operator=(thread&& other) noexcept {
    if (joinable())
        std::terminate();

    _id     = std::exchange(other._id, {});
    _handle = std::exchange(other._handle, {});

    return *this;
}

void thread::join() {

}

void thread::detach() {

}

namespace this_thread {
    thread::id get_id() noexcept {
        return _CurrentThreadId();
    }

    simd_stl_always_inline void yield() noexcept {
        _Yield();
    }

    template <
        class _Clock_,
        class _Duration_>
    void sleep_until(const std::chrono::time_point<_Clock_, _Duration_>& absoluteTime) {
        constexpr auto maximumSleepMs = std::chrono::milliseconds(std::chrono::hours(24));

        while (true) {
            const auto now = _Clock_::now();

            if (absoluteTime <= now)
                return;

            uint32 ms = 0;
            const uint32 remainingTime = (absoluteTime - now);

            if (remainingTime < maximumSleepMs)
                ms = maximumSleepMs;
            else
                ms = std::chrono::ceil<std::chrono::milliseconds>(remainingTime).count();

            _CurrentThreadSleep(ms);
        }
    }

    template <
        class _TickCountType_,
        class _Period_>
    void sleep_for(const std::chrono::duration<_TickCountType_, _Period_>& relativeTime) {
        sleep_until(_ToAbsoluteTime(relativeTime));
    }
} // namespace this_thread




__SIMD_STL_CONCURRENCY_NAMESPACE_END
