#pragma once 

#include <numeric>

#include <simd_stl/compatibility/CompilerDetection.h>
#include <simd_stl/compatibility/CallingConventions.h>

#include <simd_stl/SimdStlNamespace.h>


__SIMD_STL_ARCH_NAMESPACE_BEGIN

#if !defined(simd_stl_cpp_msvc)

#if !defined(__simd_stl_cpuid)

#  define __simd_stl_cpuid(leaf, a, b, c, d)			\
     __asm volatile("cpuid\n\t"							\
	   : "=a" (a), "=b" (b), "=c" (c), "=d" (d)			\
	   : "0" (leaf), "1" (0), "2" (0))
#endif // !defined(__simd_stl_cpuid)

#if !defined(__simd_stl_cpuidex)
#  define __simd_stl_cpuidex(leaf, subleaf, a, b, c, d)	\
     __asm volatile("cpuid\n\t"							\
	   : "=a" (a), "=b" (b), "=c" (c), "=d" (d)			\
	   : "0" (leaf), "1" (0), "2" (subleaf))
#endif // !defined(__simd_stl_cpuidex)

int simd_stl_cdecl GetCpuIdMax(
	uint32_t	ext,
	uint32_t*	sig) noexcept
{
	uint32_t eax, ebx, ecx, edx;

#if !defined(simd_stl_processor_x86_64)
#if simd_stl_cpp_gnu >= 3
	__asm volatile("pushf{l|d}\n\t"
		"pushf{l|d}\n\t"
		"pop{l}\t%0\n\t"
		"mov{l}\t{%0, %1|%1, %0}\n\t"
		"xor{l}\t{%2, %0|%0, %2}\n\t"
		"push{l}\t%0\n\t"
		"popf{l|d}\n\t"
		"pushf{l|d}\n\t"
		"pop{l}\t%0\n\t"
		"popf{l|d}\n\t"
		: "=&r" (eax), "=&r" (ebx)
		: "i" (0x00200000));
#else
	__asm volatile(
		"pushfl\n\t"
		"pushfl\n\t"
		"popl\t%0\n\t"
		"movl\t%0, %1\n\t"
		"xorl\t%2, %0\n\t"
		"pushl\t%0\n\t"
		"popfl\n\t"
		"pushfl\n\t"
		"popl\t%0\n\t"
		"popfl\n\t"
		: "=&r" (eax), "=&r" (ebx)
		: "i" (0x00200000)
	);
#endif // simd_stl_cpp_gnu >= 3

	if (!((eax ^ ebx) & 0x00200000)) 
		return 0;
#endif // !defined(simd_stl_processor_x86_64)

	__simd_stl_cpuid(ext, eax, ebx, ecx, edx);

	if (sig) 
		*sig = ebx;

	return eax;
}

int simd_stl_cdecl CheckLeafInvalid(uint32_t leaf) noexcept {
	return ((GetCpuIdMax(leaf & 0x40000000, 0) < leaf) && (GetCpuIdMax(leaf & 0x80000000, 0) < leaf));
}

int simd_stl_cdecl GetCpuId(
	uint32_t	leaf,
	uint32_t*	eax, 
	uint32_t*	ebx,
	uint32_t*	ecx, 
	uint32_t*	edx) noexcept
{
	if (CheckLeafInvalid(leaf))
		return 0;

	__simd_stl_cpuid(__leaf, *eax, *ebx, *ecx, *edx);
	return 1;
}

int simd_stl_cdecl GetCpuIdEx(
	uint32_t	leaf, 
	uint32_t	subleaf,
	uint32_t*	eax, 
	uint32_t*	ebx,
	uint32_t*	ecx,
	uint32_t*	edx) noexcept
{
	if (CheckLeafInvalid(__leaf))
		return 0;

	__simd_stl_cpuidex(__leaf, __subleaf, *__eax, *__ebx, *__ecx, *__edx);
	return 1;
}

#endif // !defined(simd_stl_cpp_msvc)

void simd_stl_cdecl cpuid(
	uint32_t regs[],
	uint32_t leaf) noexcept
{
#if !defined(simd_stl_cpp_msvc)
	GetCpuId(leaf, regs, regs + 1, regs + 2, regs + 3);
#else
	__cpuid((int32_t*)regs, leaf);
#endif // !defined(simd_stl_cpp_msvc)
}

void simd_stl_cdecl cpuidex(
	uint32_t regs[],
	uint32_t leaf,
	uint32_t subleaf) noexcept
{
#if !defined(simd_stl_cpp_msvc)
	GetCpuIdEx(leaf, subleaf, regs, regs + 1, regs + 2, regs + 3);
#else
	__cpuidex((int32_t*)regs, leaf, subleaf);
#endif // !defined(simd_stl_cpp_msvc)
}

__SIMD_STL_ARCH_NAMESPACE_END
