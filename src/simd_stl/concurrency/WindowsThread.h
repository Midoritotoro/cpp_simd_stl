#pragma once 

#include <simd_stl/compatibility/Compatibility.h>
#include <chrono>
#include <src/simd_stl/type_traits/Invoke.h>

#if defined(simd_stl_os_win) 

#if !defined(_DLL)
#  include <process.h> // _beginthreadex && _endthreadex
#else
   simd_stl_disable_warning_msvc(6258)
#endif // !defined(_DLL)

#include <Windows.h>

__SIMD_STL_CONCURRENCY_NAMESPACE_BEGIN

struct __thread_type {
    void* handle = nullptr;
    dword_t id = 0;
};

enum class __thread_result: uint8 {
    Error,
    Success
};

enum __thread_creation_flags : dword_t {
    __run_after_creation        = 0,
    __suspend_after_creation    = CREATE_SUSPENDED
};

simd_stl_nodiscard dword_t __current_thread_id() noexcept {
	return GetCurrentThreadId();
}

simd_stl_nodiscard dword_t __thread_id(void* __handle) noexcept {
	return GetThreadId(__handle);
}

simd_stl_nodiscard void* __current_thread() noexcept {
	return GetCurrentThread();
}

__thread_result __wair_for_thread(void* __handle) noexcept {
    if (WaitForSingleObjectEx(__handle, INFINITE, FALSE) == WAIT_FAILED)
        return __thread_result::Error;

    return __thread_result::Success;
}

int __thread_priority(void* __handle) noexcept {
    return GetThreadPriority(__handle);
}

void __current_thread_sleep(dword_t __milliseconds) noexcept {
	Sleep(__milliseconds);
}

template <
    class           _Tuple_,
    sizetype ...    _Indices_>
uint32 simd_stl_stdcall __thread_task_invoke(void* __raw) noexcept {
    const std::unique_ptr<_Tuple_> __args(static_cast<_Tuple_*>(__raw));

    _Tuple_& __tuple = *__args.get();
    type_traits::invoke(std::move(std::get<_Indices_>(__tuple))...);

    return 0;
}

template <
    class       _Tuple_,
    size_t...   _Indices_>
static constexpr auto __get_thread_task_invoker(std::index_sequence<_Indices_...>) noexcept {
    return &__thread_task_invoke<_Tuple_, _Indices_...>;
}

template <
    class       _Task_,
    class...    _Args_>
__thread_type simd_stl_stdcall __create_thread(
    __thread_creation_flags     __creation,
    dword_t                     __stack_size,
    _Task_&&                    __task,
    _Args_&& ...                __args) noexcept
{
    __thread_type __result;
    using _Tuple = std::tuple<std::decay_t<_Task_>, std::decay_t<_Args_>...>;

    auto __decay_copied         = std::make_unique<_Tuple>(std::forward<_Task_>(__task),  std::forward<_Args_>(__args)...);
    constexpr auto __invoker    = __get_thread_task_invoker<_Tuple>(std::make_index_sequence<1 + sizeof...(_Args_)>{});

    auto __thread_id = dword_t(0);

#if defined(simd_stl_cpp_msvc) && !defined(_DLL)
    // -MT || -MTd 

    __result.handle = reinterpret_cast<HANDLE>(
        _beginthreadex(
            nullptr, __stack_size, __invoker, __decay_copied.get(), __creation,
            reinterpret_cast<uint32*>(&__thread_id)
        )
    );
#else
    // -MD || -MDd

    __result.handle = CreateThread(
        nullptr, __stack_size, reinterpret_cast<LPTHREAD_START_ROUTINE>(__invoker),
        reinterpret_cast<LPVOID>(__decay_copied.get()),
        __creation, reinterpret_cast<LPDWORD>(&__thread_id));

#endif // defined(simd_stl_cpp_msvc) && !defined(_DLL)

    if (simd_stl_likely(__result.handle != nullptr)) {
        __result.id = __thread_id;
        simd_stl_unused(__decay_copied.release());
    }
    else {
        __result.id = 0;
    }

    return __result;
}

bool __resume_suspended_thread(void* __handle) noexcept {
    return ResumeThread(__handle) != -1;
}

dword_t __thread_exit_code(void* __handle) {
    dword_t __exit_code = 0;
    GetExitCodeThread(__handle, &__exit_code);

    return __exit_code;
}

void simd_stl_stdcall __detach_thread(void* __handle) noexcept {
    CloseHandle(__handle);
}   

dword_t simd_stl_stdcall __terminate_thread(void* __handle) noexcept {
    return TerminateThread(__handle, __thread_exit_code(__handle));
}

void simd_stl_stdcall __set_thread_priority(
    void*   __handle,
    int     __priority) noexcept 
{
    if (!SetThreadPriority(__handle, __priority))
        printf("simd_stl::concurrency::_SetThreadPriority: Failed to set thread priority.");
}

template <
    class _TickCountType_, 
    class _Period_>
simd_stl_always_inline auto __to_absolute_time(const std::chrono::duration<_TickCountType_, _Period_>& __relative_time) noexcept {
    constexpr auto __zero = std::chrono::duration<_TickCountType_, _Period_>::zero();
    const auto __now      = std::chrono::steady_clock::now();

    decltype(__now + __relative_time) __absoluteTime = __now; 

    if (__relativeTime > __zero) {
        constexpr auto __forever = (decltype(__absolute_time)::max)();

        if (__absoluteTime < __forever - __relative_time)
            __absoluteTime += __relative_time;
        else
            __absoluteTime = __forever;
    }

    return __absoluteTime;
}

__SIMD_STL_CONCURRENCY_NAMESPACE_END

#endif // defined(simd_stl_os_win)