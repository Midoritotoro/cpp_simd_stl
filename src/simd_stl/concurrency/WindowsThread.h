#pragma once 

#include <simd_stl/compatibility/Compatibility.h>
#include <chrono>
#include <type_traits>

#if defined(simd_stl_os_win) 

#if !defined(_DLL)
#  include <process.h> // _beginthreadex && _endthreadex
#else
   simd_stl_disable_warning_msvc(6258)
#endif // !defined(_DLL)

#include <Windows.h>

__SIMD_STL_CONCURRENCY_NAMESPACE_BEGIN

struct _ThreadType {
    void* handle = nullptr;
    dword_t id = 0;
};

enum class _ThreadResult: uint8 {
    _Error,
    _Success
};

enum _ThreadCreationFlags : dword_t {
    _RunAfterCreation = 0,
    _SuspendAfterCreation = CREATE_SUSPENDED
};

simd_stl_nodiscard dword_t _CurrentThreadId() noexcept {
	return GetCurrentThreadId();
}

simd_stl_nodiscard dword_t _ThreadId(void* handle) noexcept {
	return GetThreadId(handle);
}

simd_stl_nodiscard void* _CurrentThread() noexcept {
	return GetCurrentThread();
}

_ThreadResult _WaitForThread(void* handle) noexcept {
    if (WaitForSingleObjectEx(handle, INFINITE, FALSE) == WAIT_FAILED)
        return _ThreadResult::_Error;

    return _ThreadResult::_Success;
}

int _ThreadPriority(void* handle) noexcept {
    return GetThreadPriority(handle);
}

void _CurrentThreadSleep(dword_t milliseconds) noexcept {
	Sleep(milliseconds);
}

template <
    class           _Tuple_,
    sizetype ...    _Indices_>
uint32 simd_stl_stdcall _ThreadTaskInvoke(void* raw) noexcept {
    const std::unique_ptr<_Tuple_> args(static_cast<_Tuple_*>(raw));

    _Tuple_& tuple = *args.get();
    std::invoke(std::move(std::get<_Indices_>(tuple))...);

    return 0;
}

template <
    class       _Tuple_,
    size_t...   _Indices_>
static constexpr auto _GetThreadTaskInvoker(std::index_sequence<_Indices_...>) noexcept {
    return &_ThreadTaskInvoke<_Tuple_, _Indices_...>;
}

template <
    class       _Task_,
    class...    _Args_>
_ThreadType simd_stl_stdcall _CreateThread(
    _ThreadCreationFlags    creation,
    dword_t                 stackSize,
    _Task_&&                task,
    _Args_&& ...            args) noexcept
{
    _ThreadType result;
    using _Tuple_ = std::tuple<std::decay_t<_Task_>, std::decay_t<_Args_>...>;

    auto decayCopied = std::make_unique<_Tuple_>(std::forward<_Task_>(task),  std::forward<_Args_>(args)...);
    constexpr auto invoker = _GetThreadTaskInvoker<_Tuple_>(std::make_index_sequence<1 + sizeof...(_Args_)>{});

    auto threadId = dword_t(0);

#if defined(simd_stl_cpp_msvc) && !defined(_DLL)
    // -MT || -MTd 

    result.handle = reinterpret_cast<HANDLE>(
        _beginthreadex(
            nullptr, stackSize, invoker, decayCopied.get(), creation,
            reinterpret_cast<uint32*>(&threadId)
        )
    );
#else
    // -MD || -MDd

    result.handle = CreateThread(
        nullptr, stackSize, reinterpret_cast<LPTHREAD_START_ROUTINE>(invoker),
        reinterpret_cast<LPVOID>(decayCopied.get()),
        creation, reinterpret_cast<LPDWORD>(&threadId));

#endif // defined(simd_stl_cpp_msvc) && !defined(_DLL)

    if (simd_stl_likely(result.handle != nullptr)) {
        result.id = threadId;
        simd_stl_unused(decayCopied.release());
    }
    else {
        result.id = 0;
    }

    return result;
}

bool _ResumeSuspendedThread(void* handle) noexcept {
    return ResumeThread(handle) != -1;
}

dword_t _ThreadExitCode(void* handle) {
    dword_t exitCode = 0;
    GetExitCodeThread(handle, &exitCode);

    return exitCode;
}

void simd_stl_stdcall _DetachThread(void* handle) noexcept {
    CloseHandle(handle);
}   

dword_t simd_stl_stdcall _TerminateThread(void* handle) noexcept {
    return TerminateThread(handle, _ThreadExitCode(handle));
}

void simd_stl_stdcall _SetThreadPriority(
    void*   handle,
    int     priority) noexcept 
{
    if (!SetThreadPriority(handle, priority))
        printf("simd_stl::concurrency::_SetThreadPriority: Failed to set thread priority.");
}

template <
    class _TickCountType_, 
    class _Period_>
simd_stl_always_inline auto _ToAbsoluteTime(const std::chrono::duration<_TickCountType_, _Period_>& relativeTime) noexcept {
    constexpr auto zero = std::chrono::duration<_TickCountType_, _Period_>::zero();
    const auto now      = std::chrono::steady_clock::now();

    decltype(now + relativeTime) absoluteTime = now; 

    if (relativeTime > zero) {
        constexpr auto _Forever = (decltype(absoluteTime)::max)();

        if (absoluteTime < _Forever - relativeTime)
            absoluteTime += relativeTime;
        else
            absoluteTime = _Forever;
    }

    return absoluteTime;
}

__SIMD_STL_CONCURRENCY_NAMESPACE_END

#endif // defined(simd_stl_os_win)