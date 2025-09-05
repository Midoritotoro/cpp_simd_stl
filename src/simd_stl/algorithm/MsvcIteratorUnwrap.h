#pragma once 

#include <simd_stl/compatibility/CompilerDetection.h>
#include <simd_stl/SimdStlNamespace.h>

#include <xutility>

#if !defined(__unwrapIterator) 
#  if defined(simd_stl_cpp_msvc)
#    define __unwrapIterator(iterator) ::std::_Get_unwrapped(iterator)
#  else
#    define __unwrapIterator(iterator) std::move(iterator)
#  endif // defined(simd_stl_cpp_msvc) 
#endif // !defined(__unwrapIterator)
