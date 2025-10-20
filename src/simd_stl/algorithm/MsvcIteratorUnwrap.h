#pragma once 

#include <simd_stl/compatibility/CompilerDetection.h>
#include <simd_stl/SimdStlNamespace.h>

#include <simd_stl/math/IntegralTypesConversions.h>
#include <xutility>

#if !defined(__unwrapIterator) 
#  if defined(simd_stl_cpp_msvc)
#    define __unwrapIterator(iterator) ::std::_Get_unwrapped(iterator)
#  else
#    define __unwrapIterator(iterator) ::std::move(iterator)
#  endif // defined(simd_stl_cpp_msvc) 
#endif // !defined(__unwrapIterator)

#if !defined(__unwrapSizedIterator)
#  if defined(simd_stl_cpp_msvc)
#    define __unwrapSizedIterator(iterator, length) ::std::_Get_unwrapped_n(iterator, length)
#  else
#    define __unwrapSizedIterator(iterator, length)	::std::move(iterator)
#  endif // defined(simd_stl_cpp_msvc)
#endif // !defined(__unwrapSizedIterator)

#if !defined(__seekWrappedIterator)
#  if defined(simd_stl_cpp_msvc)
#    define __seekWrappedIterator(from, to) ::std::_Seek_wrapped(from, to)
#  else
#    define __seekWrappedIterator(from, to)	from = ::std::move(to);
#  endif // defined(simd_stl_cpp_msvc)
#endif // !defined(__seekWrappedIterator)