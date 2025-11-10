#pragma once 

#include <simd_stl/compatibility/Compatibility.h>

#if defined(simd_stl_os_win) 

#include <Windows.h>

__SIMD_STL_CONCURRENCY_NAMESPACE_BEGIN

simd_stl_nodiscard dword_t _CurrentThreadId() noexcept {
	return GetCurrentThreadId();
}

simd_stl_nodiscard dword_t _ThreadId(void* handle) noexcept {
	return GetThreadId(handle);
}

simd_stl_nodiscard void* _CurrentThread() noexcept {
	return GetCurrentThread();
}

void _CurrentThreadSleep(dword_t milliseconds) noexcept {
	Sleep(milliseconds);
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

class WindowsThread {
	class WindowsThreadId;
public:
	using native_handle_type		= void*;
	using thread_id_wrapper_type	= WindowsThreadId;

	simd_stl_always_inline thread_id_wrapper_type get_id() noexcept;
};

class WindowsThread::WindowsThreadId {
public:
	using thread_id_type = dword_t;

	WindowsThreadId() noexcept;
	WindowsThreadId(thread_id_type id) noexcept;

#if simd_stl_has_cxx20
	simd_stl_always_inline friend std::strong_ordering operator<=>(const WindowsThreadId& left, const WindowsThreadId& right) noexcept;
#else
	simd_stl_always_inline friend bool operator==(const WindowsThreadId& left, const WindowsThreadId& right) noexcept;
	simd_stl_always_inline friend bool operator!=(const WindowsThreadId& left, const WindowsThreadId& right) noexcept;

	simd_stl_always_inline friend bool operator<(const WindowsThreadId& left, const WindowsThreadId& right) noexcept;
	simd_stl_always_inline friend bool operator>(const WindowsThreadId& left, const WindowsThreadId& right) noexcept;

	simd_stl_always_inline friend bool operator<=(const WindowsThreadId& left, const WindowsThreadId& right) noexcept;
	simd_stl_always_inline friend bool operator>=(const WindowsThreadId& left, const WindowsThreadId& right) noexcept;
#endif // simd_stl_has_cxx20

	template <
		class _Char_,
		class _Traits_>
	friend std::basic_ostream<_Char_, _Traits_>& operator<<(
		std::basic_ostream<_Char_, _Traits_>&	stream,
		concurrency::thread::id					id);

	simd_stl_nodiscard simd_stl_always_inline thread_id_type id() const noexcept;
private:
	thread_id_type _id = 0;
};

WindowsThread::WindowsThreadId::WindowsThreadId() noexcept {}

WindowsThread::WindowsThreadId::WindowsThreadId(thread_id_type id) noexcept:
	_id(id)
{}

#if simd_stl_has_cxx20

std::strong_ordering operator<=>(
	const WindowsThread::thread_id_wrapper_type& left,
	const WindowsThread::thread_id_wrapper_type& right) noexcept
{
	return left._id <=> right._id;
}

#else

bool operator==(
	const WindowsThread::thread_id_wrapper_type& left,
	const WindowsThread::thread_id_wrapper_type& right) noexcept 
{
	return left._id == right._id;
}

bool operator!=(
	const WindowsThread::thread_id_wrapper_type& left,
	const WindowsThread::thread_id_wrapper_type& right) noexcept
{
	return left._id != right._id;
}

bool operator<(
	const WindowsThread::thread_id_wrapper_type& left,
	const WindowsThread::thread_id_wrapper_type& right) noexcept
{
	return left._id < right._id;
}

bool operator>(
	const WindowsThread::thread_id_wrapper_type& left,
	const WindowsThread::thread_id_wrapper_type& right) noexcept
{
	return left._id > right._id;
}

bool operator<=(
	const WindowsThread::thread_id_wrapper_type& left,
	const WindowsThread::thread_id_wrapper_type& right) noexcept 
{
	return left._id <= right._id;
}

bool operator>=(
	const WindowsThread::thread_id_wrapper_type& left,
	const WindowsThread::thread_id_wrapper_type& right) noexcept
{
	return left._id >= right._id;
}

#endif // simd_stl_has_cxx20

simd_stl_nodiscard simd_stl_always_inline WindowsThread::thread_id_wrapper_type::thread_id_type
	WindowsThread::thread_id_wrapper_type::id() const noexcept
{
	return _id;
}

template <
	class _Char_, 
	class _Traits_>
std::basic_ostream<_Char_, _Traits_>& operator<<(
	std::basic_ostream<_Char_, _Traits_>&	stream, 
	concurrency::thread::id					id)
{
	static_assert(sizeof(concurrency::thread::id) == 4);
	_Char_ buffer[11];

	_Char_* end = std::end(buffer);
	*--end = static_cast<_Char_>('\0');

	end = algorithm::UnsignedIntegralToBuffer(end, id._id);
	return stream << end;
}

__SIMD_STL_CONCURRENCY_NAMESPACE_END

#endif // defined(simd_stl_os_win)