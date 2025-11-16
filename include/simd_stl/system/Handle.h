#pragma once 

#include <simd_stl/compatibility/Compatibility.h>
#include <simd_stl/Types.h>

__SIMD_STL_SYSTEM_NAMESPACE_BEGIN

class handle {
public:
	using native_handle_type	= system_handle_t;
	using deleter_type			= system_bool_t(*)(native_handle_type);

	handle() noexcept;
	~handle() noexcept;

	handle(native_handle_type handle) noexcept;

	simd_stl_always_inline handle& operator=(const handle& other) noexcept;
	simd_stl_always_inline handle& operator=(const native_handle_type other) noexcept;

	handle(
		native_handle_type	handle,
		bool				autoDelete,
		deleter_type		deleter) noexcept;

	simd_stl_always_inline void setDeleter(deleter_type deleter) noexcept;
	simd_stl_nodiscard simd_stl_always_inline deleter_type deleter() const noexcept;

	simd_stl_always_inline void setAutoDelete(bool autoDelete) noexcept;
	simd_stl_nodiscard simd_stl_always_inline bool autoDelete() const noexcept;

	simd_stl_always_inline void setNativeHandle(
		native_handle_type	handle, 
		bool				deletePrevious = true) noexcept;
	simd_stl_nodiscard simd_stl_always_inline native_handle_type nativeHandle() noexcept;

	simd_stl_always_inline bool destroy() noexcept;
	simd_stl_nodiscard simd_stl_always_inline bool available() const noexcept;
	
	simd_stl_always_inline friend bool operator==(
		const handle& left,
		const handle& right) noexcept;

	simd_stl_always_inline friend bool operator!=(
		const handle& left,
		const handle& right) noexcept;
protected:
	native_handle_type	_nativeHandle	= nullptr;
	deleter_type		_deleter		= nullptr;

	bool _autoDelete = true;
};

handle::handle() noexcept {}

handle::~handle() noexcept {
	if (_autoDelete)
		destroy();
}

handle::handle(native_handle_type handle) noexcept:
	_nativeHandle(handle)
{}

handle::handle(
	native_handle_type	handle,
	bool				autoDelete,
	deleter_type		deleter
) noexcept:
	_nativeHandle(handle),
	_autoDelete(autoDelete),
	_deleter(std::move(deleter))
{}

handle& handle::operator=(const native_handle_type other) noexcept {
	_nativeHandle = other;
	return *this;
}

void handle::setDeleter(deleter_type deleter) noexcept {
	_deleter = std::move(deleter);
}

handle::deleter_type handle::deleter() const noexcept {
	return _deleter;
}

void handle::setAutoDelete(bool autoDelete) noexcept {
	_autoDelete = autoDelete;
}

bool handle::autoDelete() const noexcept {
	return _autoDelete;
}

void handle::setNativeHandle(
	native_handle_type	handle,
	bool				deletePrevious) noexcept 
{
	if (handle == _nativeHandle)
		return;

	if (deletePrevious)
		destroy();

	_nativeHandle = handle;
}

handle::native_handle_type handle::nativeHandle() noexcept {
	return _nativeHandle;
}

bool handle::destroy() noexcept {
	if (_nativeHandle != nullptr) {
		_deleter(_nativeHandle);
		_nativeHandle = nullptr;

		return true;
	}

	return false;
}

bool handle::available() const noexcept {
	return (_nativeHandle != nullptr);
}

handle& handle::operator=(const handle& other) noexcept {
	_nativeHandle = other._nativeHandle;
	return *this;
}

simd_stl_nodiscard bool operator==(
	const handle& left,
	const handle& right) noexcept
{
	return left._nativeHandle == right._nativeHandle;
}

simd_stl_nodiscard bool operator!=(
	const handle& left,
	const handle& right) noexcept
{
	return left._nativeHandle != right._nativeHandle;
}

__SIMD_STL_SYSTEM_NAMESPACE_END
