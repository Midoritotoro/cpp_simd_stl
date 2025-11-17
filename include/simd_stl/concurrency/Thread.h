#pragma once 

#include <simd_stl/arch/ProcessorInformation.h>

#if defined(simd_stl_os_win)
#  include <src/simd_stl/concurrency/WindowsThread.h>
#endif // defined(simd_stl_os_win)

#include <simd_stl/concurrency/ThreadId.h>
#include <simd_stl/concurrency/ThreadHandle.h>

#include <src/simd_stl/concurrency/ThreadYield.h>
#include <chrono>
#include <thread>

__SIMD_STL_CONCURRENCY_NAMESPACE_BEGIN

class thread final {
public:
    static constexpr auto defaultStackSize      = 1024 * 1024; // 1 Mb
    static constexpr auto defaultStackPriority  = Priority::NormalPriority;

	using handle_type	= thread_handle;
	using id			= thread_id;

	simd_stl_nodiscard_constructor thread() noexcept;
    simd_stl_nodiscard_constructor thread(thread&& other) noexcept;

    thread(const thread&) = delete;
    thread& operator=(const thread&) = delete;

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

    /**
     * @brief Запускает новый поток выполнения, исполняющий переданную задачу.
     *
     * Принимает вызываемый объект
     * и создаёт поток, который будет выполнять его без дополнительных аргументов.
     *
     * @tparam _Task_ Тип вызываемого объекта.
     * @param task    Вызываемый объект, который будет выполнен в новом потоке.
     */
    template <class _Task_>
    inline void start(_Task_&& task);

    /**
     * @brief Запускает новый поток выполнения, исполняющий переданную задачу
     *        с аргументами, аналогично std::invoke.
     *
     * Данный метод принимает вызываемый объект
     * и аргументы, которые будут переданы при вызове. Первый аргумент используется
     * как объект для вызова указателя на метод или как первый параметр функции,
     * остальные аргументы передаются далее.
     *
     * @tparam _Task_          Тип вызываемого объекта.
     * @tparam _FirstArgument_ Тип первого аргумента.
     * @tparam _Args_          Типы остальных аргументов.
     *
     * @param task          Вызываемый объект, который будет выполнен в новом потоке.
     * @param firstArgument Первый аргумент, используемый для вызова (например, объект для метода).
     * @param args          Дополнительные аргументы, передаваемые вызываемому объекту.
     */
    template <
        class       _Task_,
        class       _FirstArgument_,
        class...    _Args_>
    inline void start(
        _Task_&&            task,
        _FirstArgument_&&   firstArgument,
        _Args_&& ...        args);


    simd_stl_nodiscard simd_stl_always_inline bool joinable() const noexcept;

    /**
     * @brief Возвращает количество аппаратных потоков выполнения,
     *        доступных на текущей системе.
     * @return Количество аппаратных потоков.
     */
    simd_stl_nodiscard simd_stl_always_inline static uint32 hardwareConcurrency() noexcept;

    /**
     * @brief Возвращает системный дескриптор потока.
     * @return Дескриптор потока.
     */
    simd_stl_nodiscard simd_stl_always_inline handle_type handle() noexcept;


    /**
    * @brief Проверяет, соответствует ли данный объект текущему потоку выполнения.
    *
    * Возвращает true, если объект управляет текущим выполняющимся потоком.
    *
    * @return Значение, указывающее на совпадение с текущим потоком.
    */
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

thread::thread(thread&& other) noexcept :
    _handle(std::exchange(other._handle, {})),
    _id(std::exchange(other._id, {})),
    _terminateOnDestroy(std::exchange(other._terminateOnDestroy, {})),
    _stackSize(std::exchange(other._stackSize, {})),
    _priority(std::exchange(other._priority, {}))
{
    DebugAssert(!other.joinable());
    _joinable = std::exchange(other._joinable, {});
}

thread::~thread() noexcept {
    if (joinable() && _terminateOnDestroy)
        _TerminateThread(_handle.nativeHandle());
    else
        _handle.destroy();
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
    algorithm::swap(_stackSize, other._stackSize);
    algorithm::swap(_joinable, other._joinable);
    algorithm::swap(_priority, other._priority);
    algorithm::swap(_terminateOnDestroy, other._terminateOnDestroy);
}

void thread::join() {
    if (_handle.available() == false)
        return;

    const auto result = _WaitForThread(_handle.nativeHandle());
    DebugAssert(result != _ThreadResult::_Error);

    _handle.destroy();
    _joinable = false;
    _id = 0;
}

void thread::detach() {
    _handle.setAutoDelete(false);
    concurrency::_DetachThread(_handle.nativeHandle());

    _id = 0;
    _handle.setNativeHandle(nullptr, false);
}

void thread::terminate() {
    if (_handle.available()) {
        concurrency::_TerminateThread(_handle.nativeHandle());

        _handle.destroy();
        _id = 0;
        _joinable = false;
    }
}

thread::handle_type thread::handle() noexcept {
    return _handle;
}

bool thread::isCurrentThread() const noexcept {
    return (concurrency::_CurrentThreadId() == _id.id());
}

uint32 thread::hardwareConcurrency() noexcept {
    return arch::ProcessorInformation::hardwareConcurrency();
}

bool thread::joinable() const noexcept {
    return _joinable;
}

thread& thread::operator=(thread&& other) noexcept {
    if (joinable())
        terminate();

    _id     = std::exchange(other._id, {});
    _handle = std::exchange(other._handle, {});

    return *this;
}

template <class _Task_>
void thread::start(_Task_&& task)
{
    if (_handle.available())
        terminate();

    const auto result = concurrency::_CreateThread(
        _ThreadCreationFlags::_SuspendAfterCreation, _stackSize,
        std::forward<_Task_>(task));

    concurrency::_SetThreadPriority(result.handle, _priority);

    _joinable = true;

    _handle = result.handle;
    _id = result.id;

    _handle.setDeleter(CloseHandle);
    concurrency::_ResumeSuspendedThread(result.handle);
}

template <
    class       _Task_,
    class       _FirstArgument_,
    class...    _Args_>
void thread::start(
    _Task_&&            task,
    _FirstArgument_&&   firstArgument,
    _Args_&& ...        args)
{
    if (_handle.available())
        terminate();

    const auto result = concurrency::_CreateThread(
        _ThreadCreationFlags::_SuspendAfterCreation, _stackSize,
        std::forward<_Task_>(task), std::forward<_FirstArgument_>(firstArgument), 
        std::forward<_Args_>(args)...);

    concurrency::_SetThreadPriority(result.handle, _priority);

    _joinable = true;

    _handle = result.handle;
    _id = result.id;

    _handle.setDeleter(CloseHandle);
    concurrency::_ResumeSuspendedThread(result.handle);
}

namespace this_thread {
    Priority get_priority() noexcept {
        return static_cast<Priority>(concurrency::_ThreadPriority(concurrency::_CurrentThread()));
    }

    thread_id get_id() noexcept {
        return concurrency::_CurrentThreadId();
    }

    simd_stl_always_inline void yield() noexcept {
        concurrency::_Yield();
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

            std::chrono::milliseconds ms;
            const auto remainingTime = (absoluteTime - now);

            if (remainingTime > maximumSleepMs)
                ms = maximumSleepMs;
            else
                ms = std::chrono::ceil<std::chrono::milliseconds>(remainingTime);

            _CurrentThreadSleep(ms.count());
        }
    }

    template <
        class _TickCountType_,
        class _Period_>
    void sleep_for(const std::chrono::duration<_TickCountType_, _Period_>& relativeTime) {
        sleep_until(concurrency::_ToAbsoluteTime(relativeTime));
    }
} // namespace this_thread

__SIMD_STL_CONCURRENCY_NAMESPACE_END
