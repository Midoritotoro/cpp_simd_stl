#pragma once 

#include <simd_stl/arch/ProcessorInformation.h>

#if defined(simd_stl_os_win)
#  include <src/simd_stl/concurrency/WindowsThread.h>
#endif // defined(simd_stl_os_win)

#include <simd_stl/concurrency/ThreadId.h>
#include <simd_stl/concurrency/ThreadHandle.h>

#include <src/simd_stl/concurrency/ThreadYield.h>
#include <simd_stl/algorithm/swap/Swap.h>

#include <chrono>


__SIMD_STL_CONCURRENCY_NAMESPACE_BEGIN


class thread final {
public:
    static constexpr auto defaultStackSize = 1024 * 1024; // Mb

	using handle_type	= thread_handle;
	using id			= thread_id;

	simd_stl_nodiscard_constructor thread() noexcept;
	simd_stl_nodiscard_constructor thread(handle_type handle) noexcept;
    simd_stl_nodiscard_constructor thread(thread&& other) noexcept;

    template <
        class       _Task_,
        class...    _Args_>
    simd_stl_nodiscard_constructor thread(
        _Task_&& task,
        _Args_&&... args);

    ~thread() noexcept;

    void setTerminateOnDestroy(bool terminateOnDestroy) noexcept;
    simd_stl_nodiscard simd_stl_always_inline bool terminateOnDestroy() const noexcept;

    void setStackSize(sizetype bytes) noexcept;
    simd_stl_nodiscard simd_stl_always_inline sizetype stackSize() const noexcept;

    void setPriority(Priority priority) noexcept;
    simd_stl_nodiscard simd_stl_always_inline Priority priority() const noexcept;

    thread& operator=(thread&& other) noexcept;


    void swap(thread& other) noexcept;

    void join();
    void detach();
    void terminate();
    void start();

    simd_stl_nodiscard simd_stl_always_inline bool joinable() const noexcept;

    simd_stl_nodiscard simd_stl_always_inline static uint32 hardwareConcurrency() noexcept;
    simd_stl_nodiscard simd_stl_always_inline handle_type handle() noexcept;

    simd_stl_nodiscard simd_stl_always_inline bool isCurrentThread() const noexcept;
private:
    handle_type _handle;
	id _id = 0;

    bool _terminateOnDestroy = false;
    sizetype _stackSize = defaultStackSize;

    Priority _priority = Priority::NormalPriority;
};

thread::thread() noexcept {}

thread::thread(thread&& other) noexcept:
    _handle(std::exchange(other._handle, {})),
    _id(std::exchange(other._id, {}))
{}

thread::thread(handle_type handle) noexcept :
    _handle(handle)
{
    _id = _ThreadId(handle.nativeHandle());
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

void thread::setTerminateOnDestroy(bool terminateOnDestroy) noexcept {
    _terminateOnDestroy = terminateOnDestroy;
}

bool thread::terminateOnDestroy() const noexcept {
    return _terminateOnDestroy;
}

void thread::setStackSize(sizetype bytes) noexcept {
    _stackSize = bytes;
}

sizetype thread::stackSize() const noexcept {
    return _stackSize;
}

void thread::setPriority(Priority priority) noexcept {
    _priority = priority;

}

Priority thread::priority() const noexcept {
    return _priority;
}

void thread::swap(thread& other) noexcept {
    algorithm::swap(_handle, other._handle);
    algorithm::swap(_id, other._id);
}

void thread::join() {
    if (joinable() == false)
        std::terminate();
}

void thread::detach() {

}

void thread::terminate() {
    _TerminateThread(_handle.nativeHandle());
}

void thread::start() {

}

thread::handle_type thread::handle() noexcept {
    return _handle;
}

bool thread::isCurrentThread() const noexcept {
    return (this_thread::get_id().id() == _CurrentThreadId());
}

uint32 thread::hardwareConcurrency() noexcept {
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

namespace this_thread {
    Priority get_priority() noexcept {
        return static_cast<Priority>(_ThreadPriority(_CurrentThread()));
    }

    thread_id get_id() noexcept {
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
