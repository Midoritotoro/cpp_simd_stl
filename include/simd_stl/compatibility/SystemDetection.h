#pragma once

#if defined(__APPLE__) && (defined(__GNUC__) || defined(__xlC__) || defined(__xlc__))
#  define simd_stl_os_apple
#  if defined(TARGET_OS_MAC) && TARGET_OS_MAC
#    define simd_stl_os_darwin
#    define simd_stl_os_bsd4
#    if defined(OS_IPHONE) && TARGET_OS_IPHONE
#    else
#      define simd_stl_os_mac
#    endif
#  endif
#elif defined(__CYGWIN__)
#  define simd_stl_os_cygwin
#elif !defined(SAG_COM) && (!defined(WINAPI_FAMILY) || WINAPI_FAMILY==WINAPI_FAMILY_DESKTOP_APP) && (defined(WIN64) || defined(_WIN64) || defined(__WIN64__))
#  define simd_stl_os_win32
#  define simd_stl_os_win64
#elif !defined(SAG_COM) && (defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__))
#  define simd_stl_os_win32
#elif defined(__linux__) || defined(__linux)
#  define simd_stl_os_linux

#elif defined(__Lynx__)
#  define simd_stl_os_lynx

#elif defined(__GNU__)
#  define simd_stl_os_hurd

#elif defined(__FreeBSD__)
#  define simd_stl_os_freebsd

#elif defined(__NetBSD__)
#  define simd_stl_os_netbsd

#elif defined(__OpenBSD__)
#  define simd_stl_os_openbsd

#elif defined(__DragonFly__)
#  define simd_stl_os_dragonfly

#elif defined(__linux__)
#  define simd_stl_os_linux

#elif defined(__native_client__)
#  define simd_stl_os_nacl

#elif defined(__EMSCRIPTEN__)
#  define simd_stl_os_emscripten

#elif defined(__rtems__)
#  define simd_stl_os_rtems

#elif defined(__Fuchsia__)
#  define simd_stl_os_fuchsia

#elif defined (__SVR4) && defined (__sun)
#  define simd_stl_os_solaris

#elif defined(__QNX__)
#  define simd_stl_os_qnx

#elif defined(__MVS__)
#  define simd_stl_os_zos

#elif defined(__hexagon__)
#  define simd_stl_os_qurt

#else
#  error ""
#endif

#if defined(simd_stl_os_win32) || defined(simd_stl_os_win64)
#  define simd_stl_os_windows
#  define simd_stl_os_win
#endif

#if defined(simd_stl_os_windows)
#  undef simd_stl_os_unix
#elif !defined(simd_stl_os_unix)
#  define simd_stl_os_unix
#endif

#if defined(simd_stl_os_win)
#  define NOMINMAX
#endif // defined(simd_stl_os_win)

#if defined(simd_stl_os_windows)
#  include <windows.h>
#endif // defined(simd_stl_os_windows)

#if defined(max) 
#  undef max
#endif

#if defined(min) 
#  undef min
#endif