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
    static constexpr auto defaultStackSize      = 1024 * 1024; // 1 Mb
    static constexpr auto defaultStackPriority  = Priority::NormalPriority;

	using handle_type	= thread_handle;
	using id			= thread_id;

	simd_stl_nodiscard_constructor thread() noexcept;
	simd_stl_nodiscard_constructor thread(handle_type handle) noexcept;
    simd_stl_nodiscard_constructor thread(thread&& other) noexcept;

    ~thread() noexcept;

    /**
     * @brief Управляет поведением деструктора объекта потока.
     *
     * Если флаг установлен в true, то при уничтожении объекта поток
     * завершается немедленно, без ожидания выполнения задачи.
     * Если флаг равен false, то деструктор вызывает join(), дожидаясь
     * завершения работы потока, и только затем освобождает ресурсы.
     *
     * @param terminateOnDestroy Логический флаг, определяющий поведение деструктора.
     */
    void setTerminateOnDestroy(bool terminateOnDestroy) noexcept;
    simd_stl_nodiscard simd_stl_always_inline bool terminateOnDestroy() const noexcept;


    /**
     * @brief Устанавливает размер стека для нового потока.
     *
     * По умолчанию используется значение defaultStackSize.
     *
     * @param bytes Размер стека в байтах.
     */
    void setStackSize(sizetype bytes) noexcept;
    simd_stl_nodiscard simd_stl_always_inline sizetype stackSize() const noexcept;


    /**
     * @brief Устанавливает приоритет выполнения потока.
     *
     * По умолчанию используется defaultStackPriority (нормальный приоритет).
     *
     * @param priority Значение приоритета.
     */
    void setPriority(Priority priority) noexcept;
    simd_stl_nodiscard simd_stl_always_inline Priority priority() const noexcept;

    thread& operator=(thread&& other) noexcept;
    void swap(thread& other) noexcept;

    /**
     * @brief Блокирует текущий поток до завершения работы потока,
     *        принадлежащего данному объекту.
     *
     * После завершения освобождаются связанные ресурсы.
     */
    void join();

    /**
     * @brief Отделяет поток выполнения от объекта.
     *
     * Поток продолжает работу независимо, а объект освобождает свои ресурсы.
     * Завершение потока произойдёт автоматически, без участия объекта.
     */
    void detach();

    /**
     * @brief Принудительно уничтожает поток выполнения.
     *
     * Завершает поток немедленно, без ожидания его нормального окончания.
     */
    void terminate();


    /*
     * @brief Запускает новый поток выполнения, связанный с данным объектом.
     *
     * Вызов создаёт поток, который начинает выполнение указанной задачи.
     *
     * @tparam _Task_   Тип вызываемого объекта (функция, лямбда, функциональный объект).
     * @tparam _Args_   Типы аргументов, передаваемых в задачу.
     *
     * @param task      Вызываемый объект, определяющий задачу для нового потока.
     * @param args      Аргументы, которые будут переданы в вызываемый объект.
    */
    template <
        class       _Task_,
        class...    _Args_>
    void start(
        _Task_&&        task,
        _Args_&&...     args);


    /*
     * @brief Запускает новый поток выполнения, связанный с данным объектом.
     *
     * Вызов создаёт поток, который начинает выполнение указанного метода
     * заданного объекта.
     *
     * @tparam _Owner_  Тип класса, которому принадлежит метод.
     * @tparam _Method_ Тип вызываемого метода (указатель на член‑функцию).
     * @tparam _Args_   Типы аргументов, передаваемых в метод.
     *
     * @param owner     Указатель на объект, для которого будет вызван метод.
     * @param routine   Указатель на член‑функцию класса _Owner_, выполняемую в новом потоке.
     * @param args      Аргументы, которые будут переданы в метод.
    */
    template <
        class       _Owner_,
        class       _Method_,
        class ...   _Args_>
    void start(
        _Owner_*        owner,
        _Method_&&      routine,
        _Args_&& ...    args);


    simd_stl_nodiscard simd_stl_always_inline bool joinable() const noexcept;

    simd_stl_nodiscard simd_stl_always_inline static uint32 hardwareConcurrency() noexcept;
    simd_stl_nodiscard simd_stl_always_inline handle_type handle() noexcept;

    simd_stl_nodiscard simd_stl_always_inline bool isCurrentThread() const noexcept;
private:
    handle_type _handle;
	id _id = 0;

    bool _terminateOnDestroy = false;
    bool _joinable = false;

    sizetype _stackSize = defaultStackSize;
    Priority _priority = defaultStackPriority;
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
    DebugAssert(_handle.available());
    _priority = priority;

    if (joinable() == false)
        _SetThreadPriority(_handle.nativeHandle(), static_cast<int>(priority));
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
    return _joinable;
}

thread& thread::operator=(thread&& other) noexcept {
    if (joinable())
        std::terminate();

    _id     = std::exchange(other._id, {});
    _handle = std::exchange(other._handle, {});

    return *this;
}


template <
    class       _Task_,
    class...    _Args_>
void thread::start(
    _Task_&&    task,
    _Args_&&... args)
{
    if (_handle.available()) {
        _ResumeSuspendedThread(_handle.nativeHandle());
    }

    const auto result = _CreateThread(
        std::forward<_Task_>(task), std::forward<_Args_>(args)...,
        _ThreadCreationFlags::_RunAfterCreation, _stackSize);

    _handle = result.handle;
    _id     = result.id;
}

template <
    class       _Owner_,
    class       _Method_,
    class ...   _Args_>
void thread::start(
    _Owner_*        owner,
    _Method_&&      routine,
    _Args_&& ...    args)
{
    const auto result = _CreateThread(
        std::bind(
            std::forward<_Method_>(routine),
            owner,
            std::forward<_Args_>(args)...),
        _ThreadCreationFlags::_RunAfterCreation, defaultStackSize);

    _handle = result.handle;
    _id = result.id;
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
