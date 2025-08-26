#pragma once 

#include "CompilerDetection.h"
#include "../arch/ProcessorDetection.h"

#include "SimdCompatibility.h"

#if !defined(simd_stl_fastcall)
#  if defined(simd_stl_processor_x86_32)
#    if defined(simd_stl_cpp_gnu) || defined(simd_stl_cpp_clang)
#      define simd_stl_fastcall __attribute__((regparm(3)))
#    elif defined(simd_stl_cpp_msvc)
#      define simd_stl_fastcall __fastcall
#    else
#      define simd_stl_fastcall
#    endif // defined(simd_stl_cpp_gnu) || defined(simd_stl_cpp_msvc) || defined(simd_stl_cpp_clang)
#  else
#    define simd_stl_fastcall
#  endif // defined(simd_stl_processor_x86_32)
#endif // !defined(simd_stl_fastcall)


#if !defined(simd_stl_stdcall)
#  if defined(simd_stl_cpp_msvc) || defined(simd_stl_cpp_clang)
#    define simd_stl_stdcall            __stdcall
#  elif defined(simd_stl_cpp_gnu)
#    define simd_stl_stdcall            __attribute__((__stdcall__))
#  else
#    define simd_stl_stdcall        
#  endif // defined(simd_stl_cpp_msvc) || defined(simd_stl_cpp_clang) || defined(simd_stl_cpp_gnu)
#endif // !defined(simd_stl_stdcall)


#if !defined(simd_stl_cdecl)
#  if defined(simd_stl_cpp_msvc) || defined(simd_stl_cpp_clang)
#    define simd_stl_cdecl          __cdecl
#  elif defined(simd_stl_cpp_gnu)
#    define simd_stl_cdecl          __attribute__((__cdecl__))
#  else
#    define simd_stl_cdecl        
#  endif // defined(simd_stl_cpp_msvc) || defined(simd_stl_cpp_clang) || defined(simd_stl_cpp_gnu)
#endif // !defined(simd_stl_cdecl)


#if !defined(simd_stl_vectorcall)
#  if defined(simd_stl_cpp_msvc) && defined(simd_stl_processor_x86) && defined(__SSE2__)
#    define simd_stl_vectorcall __vectorcall
#  else
#    define simd_stl_vectorcall
#  endif
#endif // !defined(simd_stl_vectorcall)
