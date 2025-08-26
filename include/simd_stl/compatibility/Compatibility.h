#pragma once 

// Clang attributes
// https://clang.llvm.org/docs/AttributeReference.html#always-inline-force-inline
// Clang builtins
// https://clang.llvm.org/docs/LanguageExtensions.html

// Msvc attributes
// https://learn.microsoft.com/en-us/cpp/cpp/declspec?view=msvc-170
// Msvc SAL
// https://learn.microsoft.com/en-us/cpp/code-quality/using-sal-annotations-to-reduce-c-cpp-code-defects?view=msvc-170

// Gcc attributes
// https://ohse.de/uwe/articles/gcc-attributes.html and https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html

#include <simd_stl/compatibility/CompilerDetection.h>
#include <simd_stl/compatibility/AlignmentMacros.h>

#include <simd_stl/compatibility/BranchPrediction.h>
#include <simd_stl/compatibility/CallingConventions.h>

#include <simd_stl/compatibility/Inline.h>
#include <simd_stl/compatibility/CxxVersionDetection.h>

#include <simd_stl/compatibility/FunctionAttributes.h>
#include <simd_stl/compatibility/LanguageFeatures.h>

#include <simd_stl/compatibility/MemoryMacros.h>
#include <simd_stl/compatibility/Nodiscard.h>

#include <simd_stl/compatibility/SimdCompatibility.h>
#include <simd_stl/compatibility/SystemDetection.h>

#include <simd_stl/compatibility/UnreachableCode.h>
#include <simd_stl/compatibility/Warnings.h>

#include <cstddef>

simd_stl_disable_warning_msvc(4067)
