#pragma once 

#include <simd_stl/compatibility/Nodiscard.h>

// Warnings

#define simd_stl_do_pragma(text)                      _Pragma(#text)

#if defined(simd_stl_cpp_msvc) && !defined(simd_stl_cpp_clang)
#  undef simd_stl_do_pragma
#endif // defined(simd_stl_cpp_msvc) && !defined(simd_stl_cpp_clang)

#if !defined(simd_stl_warning_push)
#  if defined(simd_stl_cpp_msvc) && !defined(simd_stl_cpp_clang)
#    define simd_stl_warning_push __pragma(warning(push))
#  elif defined(simd_stl_cpp_clang)
#    define simd_stl_warning_push simd_stl_do_pragma(clang diagnostic push)
#  elif defined(simd_stl_cpp_gnu) && (__GNUC__ * 100 + __GNUC_MINOR__ >= 406)
#    define simd_stl_warning_push simd_stl_do_pragma(GCC diagnostic push)
#  else 
#    define simd_stl_warning_push
#  endif // (defined(simd_stl_cpp_msvc) && !defined(simd_stl_cpp_clang)) || defined(simd_stl_cpp_clang)
         // || defined(simd_stl_cpp_gnu) && (__GNUC__ * 100 + __GNUC_MINOR__ >= 406)
#endif // !defined(simd_stl_warning_push)


#if !defined(simd_stl_warning_pop)
#  if defined(simd_stl_cpp_msvc) && !defined(simd_stl_cpp_clang)
#    define simd_stl_warning_pop __pragma(warning(pop))
#  elif defined(simd_stl_cpp_clang)
#    define simd_stl_warning_pop simd_stl_do_pragma(clang diagnostic pop)
#  elif defined(simd_stl_cpp_gnu) && (__GNUC__ * 100 + __GNUC_MINOR__ >= 406)
#    define simd_stl_warning_pop simd_stl_do_pragma(GCC diagnostic pop)
#  else
#    define simd_stl_warning_pop
#  endif // (defined(simd_stl_cpp_msvc) && !defined(simd_stl_cpp_clang)) || defined(simd_stl_cpp_clang)
         // || defined(simd_stl_cpp_gnu) && (__GNUC__ * 100 + __GNUC_MINOR__ >= 406)
#endif // !defined(simd_stl_warning_pop)


#if !defined(simd_stl_disable_warning_msvc)
#  if defined(simd_stl_cpp_msvc) && !defined(simd_stl_cpp_clang)
#    define simd_stl_disable_warning_msvc(number) __pragma(warning(disable: number))
#  else
#    define simd_stl_disable_warning_msvc(number)
#  endif // defined(simd_stl_cpp_msvc) && !defined(simd_stl_cpp_clang)
#endif // !defined(simd_stl_disable_warning_msvc)


#if !defined(simd_stl_disable_warning_clang)
#  if defined(simd_stl_cpp_clang)
#    define simd_stl_disable_warning_clang(text) simd_stl_do_pragma(clang diagnostic ignored text)
#  else
#    define simd_stl_disable_warning_clang(text)
#  endif // defined(simd_stl_cpp_msvc) && !defined(simd_stl_cpp_clang)
#endif // !defined(simd_stl_disable_warning_clang)


#if !defined(simd_stl_disable_warning_gcc)
#  if defined(simd_stl_cpp_gnu) && (__GNUC__ * 100 + __GNUC_MINOR__ >= 406)
#    define simd_stl_disable_warning_gcc(text) simd_stl_do_pragma(GCC diagnostic ignored text)
#  else
#    define simd_stl_disable_warning_gcc(text) 
#  endif // defined(simd_stl_cpp_msvc) && !defined(simd_stl_cpp_clang)
#endif // !defined(simd_stl_disable_warning_gcc)


#if !defined(simd_stl_disable_warning_deprecated)
#  if defined(simd_stl_cpp_msvc) && !defined(simd_stl_cpp_clang)
#    define simd_stl_disable_warning_deprecated simd_stl_disable_warning_msvc(4996)
#  elif defined(simd_stl_cpp_clang)
#    define simd_stl_disable_warning_deprecated simd_stl_disable_warning_clang("-Wdeprecated-declarations")
#  elif defined(simd_stl_cpp_gnu) && (__GNUC__ * 100 + __GNUC_MINOR__ >= 406)
#    define simd_stl_disable_warning_deprecated simd_stl_warning_disable_gcc("-Wdeprecated-declarations")
#  else
#    define simd_stl_disable_warning_deprecated
#  endif // (defined(simd_stl_cpp_msvc) && !defined(simd_stl_cpp_clang)) || defined(simd_stl_cpp_clang)
         // || defined(simd_stl_cpp_gnu) && (__GNUC__ * 100 + __GNUC_MINOR__ >= 406)
#endif // !defined(simd_stl_disable_warning_deprecated)


#if !defined(simd_stl_cpp_warnings)
#  define simd_stl_no_warnings
#endif // !defined(simd_stl_cpp_warnings)


#if defined(simd_stl_no_warnings)
#  if defined(simd_stl_cpp_msvc)
     simd_stl_disable_warning_msvc(4828) /* Файл содержит знак, начинающийся со смещения 0xX, который является недопустимым в текущей исходной кодировке. */
     simd_stl_disable_warning_msvc(4251) /* класс 'type' должен иметь dll-интерфейс, который будет использоваться клиентами класса 'type2' */
     simd_stl_disable_warning_msvc(4244) /* преобразование из 'type1' в 'type2', возможная потеря данных */
     simd_stl_disable_warning_msvc(4275) /* идентификатор ключа класса, отличного от DLL-интерфейса, используемый в качестве базового для идентификатора ключа класса DLL-интерфейса */
     simd_stl_disable_warning_msvc(4514) /* удалена встроенная функция, на которую нет ссылок */
     simd_stl_disable_warning_msvc(4800) /* 'type' : принудительное присвоение значения bool 'true' или 'false' (предупреждение о производительности) */
     simd_stl_disable_warning_msvc(4097) /* typedef-имя 'identifier1' используется как синоним имени класса 'identifier2' */
     simd_stl_disable_warning_msvc(4706) /* присвоение в условном выражении */
     simd_stl_disable_warning_msvc(4355) /* 'this' : используется в списке инициализаторов базовых элементов */
     simd_stl_disable_warning_msvc(4710) /* функция не встроена */
     simd_stl_disable_warning_msvc(4530) /* Используется обработчик исключений C++, но семантика размотки не включена. Укажите /EHsc */
     simd_stl_disable_warning_msvc(4006)
     simd_stl_disable_warning_msvc(4715)
#  endif // defined(simd_stl_cpp_msvc)

#endif // defined(simd_stl_no_warnings)

#if !defined(simd_stl_nodiscard_return_raw_ptr)
#  define simd_stl_nodiscard_return_raw_ptr \
        simd_stl_nodiscard_with_warning("This function allocates memory and returns a raw pointer. " \
            "Discarding the return value will cause a memory leak.")
#endif // !defined(simd_stl_nodiscard_return_raw_ptr)


#if !defined(simd_stl_nodiscard_thread_constructor)
#  define simd_stl_nodiscard_thread_constructor \
    simd_stl_nodiscard_constructor_with_warning("Creating a thread object without assigning it to a variable " \
        "may lead to unexpected behavior and resource leaks. Ensure " \
        "the thread is properly managed.")
#endif // !defined(simd_stl_nodiscard_thread_constructor)


#if !defined(simd_stl_nodiscard_remove_algorithm)
#  define simd_stl_nodiscard_remove_algorithm \
        simd_stl_nodiscard_with_warning("The 'remove' and 'remove_if' algorithms return the iterator past the last element " \
            "that should be kept. You need to call container.erase(result, container.end()) afterwards. " \
            "In C++20, 'std::erase' and 'std::erase_if' are simpler replacements for these two steps.")
#endif // !defined(simd_stl_nodiscard_remove_algorithm)


#if !defined(simd_stl_deprecated_warning)
#  if defined(simd_stl_cpp_msvc)
#    define simd_stl_deprecated_warning(message)                                           \
       __pragma(warning(push))                                                         \
       __pragma(warning(disable: 4996))                                                \
       __pragma(message (__FILE__ "(" __LINE__ ") : warning C4996: " message))         \
       __pragma(warning(pop))
#  elif defined(simd_stl_cpp_clang)
#    define simd_stl_deprecated_warning(message)                                           \
       simd_stl_do_pragma("clang diagnostic push")                                         \
       simd_stl_do_pragma("clang diagnostic warning \"-Wdeprecated-declarations\"")        \
       simd_stl_do_pragma("clang diagnostic ignored \"-Wunused-but-set-variable\"")        \
       simd_stl_do_pragma("message \"" __FILE__ "(" __LINE__ ") : warning: " message "\"") \
       simd_stl_do_pragma("clang diagnostic pop")
#  elif defined(simd_stl_cpp_gnu)
#    define simd_stl_deprecated_warning(message)                                           \
       simd_stl_do_pragma("GCC diagnostic push")                                           \
       simd_stl_do_pragma("GCC diagnostic warning \"-Wdeprecated-declarations\"")          \
       simd_stl_do_pragma("message \"" __FILE__ "(" __LINE__ ") : warning: " message "\"") \ 
         simd_stl_do_pragma("GCC diagnostic pop")
#  else
#    define simd_stl_deprecated_warning(message)
#  endif // defined(simd_stl_cpp_msvc) || defined(simd_stl_cpp_clang) || defined(simd_stl_cpp_gnu)
#endif // !defined(simd_stl_deprecated_warning)