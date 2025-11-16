cmake_minimum_required (VERSION 4.1  FATAL_ERROR)

project ("cpp_simd_stl"
    VERSION 1.0.0
    DESCRIPTION "..."
    LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_LINK_ $<IF:$<CONFIG:Debug>, /NODEFAULTLIB:LIBCMT, /DEBUG;/OPT:REF> /INCREMENTAL:NO)

if (NOT DEFINED CMAKE_INSTALL_LIBDIR)
  set(CMAKE_INSTALL_LIBDIR "lib")
endif()
if (NOT DEFINED CMAKE_INSTALL_BINDIR)
  set(CMAKE_INSTALL_BINDIR "bin")
endif()
if (NOT DEFINED CMAKE_INSTALL_INCLUDEDIR)
  set(CMAKE_INSTALL_INCLUDEDIR "include")
endif()

option(CPP_SIMD_STL_INSTALL "" ${PROJECT_IS_TOP_LEVEL})
set(CPP_SIMD_STL_INSTALL_CMAKE_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/cpp_simd_stl")

set(PRIVATE_SOURCES  
    src/simd_stl/algorithm/vectorized/FindVectorized.cpp
)

set(PRIVATE_HEADERS 
    src/simd_stl/algorithm/vectorized/FindVectorized.h
    src/simd_stl/algorithm/vectorized/AvxZeroUpperGuard.h
    src/simd_stl/algorithm/vectorized/ContainsRangeVectorized.h
    src/simd_stl/algorithm/vectorized/ContainsVectorized.h
    src/simd_stl/algorithm/vectorized/CountVectorized.h
    src/simd_stl/algorithm/vectorized/FindLastRangeVectorized.h
    src/simd_stl/algorithm/vectorized/FindLastVectorized.h
    src/simd_stl/algorithm/vectorized/FindRangeVectorized.h
    src/simd_stl/algorithm/vectorized/FindVectorized.h
    src/simd_stl/algorithm/vectorized/MismatchVectorized.h
    src/simd_stl/algorithm/vectorized/MinMaxVectorized.h
    src/simd_stl/algorithm/vectorized/ReplaceVectorized.h
    src/simd_stl/algorithm/vectorized/ReverseCopyVectorized.h
    src/simd_stl/algorithm/vectorized/ReverseVectorized.h
    src/simd_stl/algorithm/vectorized/SwapRangesVectorized.h
    src/simd_stl/algorithm/vectorized/EqualVectorized.h
    src/simd_stl/algorithm/vectorized/FindEndVectorized.h
    src/simd_stl/algorithm/vectorized/CopyVectorized.h
    src/simd_stl/algorithm/vectorized/MoveVectorized.h
    src/simd_stl/algorithm/vectorized/FillVectorized.h
    src/simd_stl/algorithm/vectorized/RemoveVectorized.h
    src/simd_stl/algorithm/vectorized/RemoveCopyVectorized.h

    src/simd_stl/algorithm/AlgorithmDebug.h
    src/simd_stl/algorithm/AdvanceBytes.h

    src/simd_stl/math/CountTrailingZeros.h
    src/simd_stl/math/CountLeadingZeros.h
    src/simd_stl/math/PopulationCount.h


    "src/simd_stl/type_traits/CanMemcmpElements.h"
    "src/simd_stl/type_traits/ContainerComparison.h"
    "src/simd_stl/type_traits/FillMemsetSafety.h"
    "src/simd_stl/type_traits/FunctionPass.h"
    "src/simd_stl/type_traits/IntegralProperties.h"
    "src/simd_stl/type_traits/IsConstantEvaluated.h"
    "src/simd_stl/type_traits/IteratorCategory.h"
    "src/simd_stl/type_traits/IteratorCheck.h" 
    "src/simd_stl/type_traits/LexicographicalCompareTraits.h"
    "src/simd_stl/type_traits/Ordering.h" 
    "src/simd_stl/type_traits/SimdAlgorithmSafety.h"
    "src/simd_stl/type_traits/SwapSafety.h" 
    "src/simd_stl/type_traits/TypeCheck.h" 
    "src/simd_stl/type_traits/TypeTraits.h"
    "src/simd_stl/type_traits/UnwrapEnum.h"
    "src/simd_stl/type_traits/SimdTypeCheck.h"
    "src/simd_stl/type_traits/IsVirtualBaseOf.h"
    src/simd_stl/type_traits/OperatorWrappers.h
    src/simd_stl/type_traits/Invoke.h

    "src/simd_stl/utility/Assert.h"

    src/simd_stl/numeric/SimdDivisors.h
    src/simd_stl/numeric/SimdArithmetic.h
    src/simd_stl/numeric/SimdCast.h
    src/simd_stl/numeric/SimdConvert.h
    src/simd_stl/numeric/SimdCompare.h
    src/simd_stl/numeric/SimdElementAccess.h
    src/simd_stl/numeric/SimdElementWise.h
    src/simd_stl/numeric/SimdIntegralTypesCheck.h
    src/simd_stl/numeric/SimdMemoryAccess.h

    src/simd_stl/concurrency/WindowsThread.h
    src/simd_stl/concurrency/ThreadYield.h
    
    src/simd_stl/algorithm/vectorized/SearchVectorized.h
    src/simd_stl/algorithm/vectorized/traits/SearchLoop.h
    src/simd_stl/algorithm/FixedMemcmp.h
    src/simd_stl/algorithm/MsvcIteratorUnwrap.h

   
)

set(PUBLIC_HEADERS
    include/simd_stl/math/BitMath.h
    include/simd_stl/math/IntegralTypesConversions.h

    include/simd_stl/arch/ProcessorFeatures.h
    include/simd_stl/arch/ProcessorDetection.h
    include/simd_stl/arch/CpuId.h
    include/simd_stl/arch/CpuFeature.h
    include/simd_stl/arch/ProcessorInformation.h

    include/simd_stl/concurrency/Thread.h
    include/simd_stl/concurrency/Mutex.h
    include/simd_stl/concurrency/ThreadHandle.h
    include/simd_stl/concurrency/ThreadId.h

    include/simd_stl/compatibility/AlignmentMacros.h
    include/simd_stl/compatibility/BranchPrediction.h
    include/simd_stl/compatibility/CallingConventions.h
    include/simd_stl/compatibility/Compatibility.h
    include/simd_stl/compatibility/CxxVersionDetection.h
    include/simd_stl/compatibility/CompilerDetection.h
    include/simd_stl/compatibility/FunctionAttributes.h
    include/simd_stl/compatibility/Inline.h
    include/simd_stl/compatibility/LanguageFeatures.h
    include/simd_stl/compatibility/MemoryMacros.h
    include/simd_stl/compatibility/Nodiscard.h
    include/simd_stl/compatibility/SystemDetection.h
    include/simd_stl/compatibility/UnreachableCode.h
    include/simd_stl/compatibility/Warnings.h
    include/simd_stl/compatibility/SimdCompatibility.h

    include/simd_stl/Types.h
    include/simd_stl/SimdStlNamespace.h

    include/simd_stl/algorithm/find/Find.h
    include/simd_stl/algorithm/find/Search.h
    include/simd_stl/algorithm/find/Contains.h
    include/simd_stl/algorithm/find/Count.h
    include/simd_stl/algorithm/find/Mismatch.h
    include/simd_stl/algorithm/find/FindFirstOf.h
    include/simd_stl/algorithm/find/StartsWith.h
    include/simd_stl/algorithm/find/EndsWith.h
    include/simd_stl/algorithm/find/AllOf.h
    include/simd_stl/algorithm/find/NoneOf.h
    include/simd_stl/algorithm/find/AnyOf.h
    include/simd_stl/algorithm/find/FindEnd.h
    include/simd_stl/algorithm/find/FindLast.h

    include/simd_stl/algorithm/copy/Copy.h
    include/simd_stl/algorithm/copy/CopyN.h
    include/simd_stl/algorithm/copy/CopyBackward.h
    include/simd_stl/algorithm/copy/Move.h
    include/simd_stl/algorithm/copy/MoveBackward.h

    include/simd_stl/algorithm/fill/Fill.h
    include/simd_stl/algorithm/fill/FillN.h

    include/simd_stl/algorithm/swap/Swap.h

    include/simd_stl/algorithm/remove/Remove.h
    include/simd_stl/algorithm/remove/RemoveCopy.h

    include/simd_stl/algorithm/replace/Replace.h

    include/simd_stl/algorithm/order/Reverse.h
    include/simd_stl/algorithm/order/ReverseCopy.h


    include/simd_stl/system/Handle.h

    include/simd_stl/memory/Intersects.h
    include/simd_stl/memory/PointerToIntegral.h
    include/simd_stl/memory/Alignment.h

    include/simd_stl/numeric/Simd.h
    include/simd_stl/numeric/BasicSimd.h
    include/simd_stl/numeric/BasicSimdElementReference.h
    include/simd_stl/numeric/BasicSimdMaskImplementation.h
    include/simd_stl/numeric/BasicSimdMask.h
    include/simd_stl/numeric/BasicSimdShuffleMask.h
)

include(CMakePackageConfigHelpers)
include(GenerateExportHeader)

add_library(cpp_simd_stl STATIC ${PRIVATE_SOURCES} ${PUBLIC_HEADERS})
add_library(cpp_simd_stl::cpp_simd_stl ALIAS cpp_simd_stl)

include(cmake/Util_ExportConfig.cmake)

target_sources(cpp_simd_stl PUBLIC 
    $<BUILD_INTERFACE:${PRIVATE_HEADERS}> 
    PRIVATE ${PRIVATE_SOURCES})


target_include_directories(cpp_simd_stl
    PUBLIC
         "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
)

target_include_directories(cpp_simd_stl
    PUBLIC
        "$<INSTALL_INTERFACE:include>"
)

target_include_directories(cpp_simd_stl
    PUBLIC
        "\$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
)

target_include_directories(cpp_simd_stl
    PUBLIC
        "\$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/export>"
)


set_target_properties(
    cpp_simd_stl PROPERTIES
    
    CXX_STANDARD_REQUIRED ON
    CXX_STANDARD 23
    CXX_EXTENSIONS OFF

    VERSION "${PROJECT_VERSION}"
    SOVERSION "${PROJECT_VERSION_MAJOR}"

    EXPORT_NAME cpp_simd_stl
    OUTPUT_NAME cpp_simd_stl

    LINKER_LANGUAGE CXX
)

if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_features(cpp_simd_stl PUBLIC cxx_std_23)
    set(CMAKE_DEBUG_TARGET_PROPERTIES "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

if(CMAKE_GENERATOR MATCHES "Visual Studio")
	if((CMAKE_VS_PLATFORM_NAME STREQUAL "x64") OR (CMAKE_VS_PLATFORM_NAME STREQUAL "x86"))
		include(cmake/x86_Simd.cmake)
    endif()
endif()


if((CMAKE_SYSTEM_PROCESSOR STREQUAL "i686") OR (CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64") OR (CMAKE_SYSTEM_PROCESSOR STREQUAL "AMD64"))
		include(cmake/x86_Simd.cmake)
endif()


if((CMAKE_CXX_COMPILER_ID STREQUAL "MSVC") OR (CMAKE_CXX_COMPILER_ID STREQUAL "Clang"))
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_options(-g) 
    elseif (CMAKE_BUILD_TYPE STREQUAL "Release")
        add_compile_options(-O2 -Ot -Ob1)
    endif()

    target_compile_options(
        cpp_simd_stl 
        PRIVATE
            /permissive-
            /W4

            /utf-8

            /MP
            /EHsc

            /guard:cf

            /w15038 
            /w14265 
            /wd4018
            /wd4100
            /wd4242
            /wd4244
            /wd4245
            /wd4267
            /wd4305
            /wd4324
            /wd4389
            /wd4456
            /wd4457
            /wd4458
            /wd4459
            /wd4611
            /wd4702
            /wd4828

            /Zi
            /Zc:__cplusplus
    )
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(
        cpp_simd_stl 
        PRIVATE
            /utf-8

            /O3
            /W4

            /EHsc   

            /Zi
            /Zc:__cplusplus
    )
endif()

if(PROJECT_IS_TOP_LEVEL)
    enable_testing()
    add_subdirectory(tests)
endif()

if(PROJECT_IS_TOP_LEVEL)
    enable_testing()
    add_subdirectory(benchmarks)
endif()

if(CPP_SIMD_STL_INSTALL AND NOT CMAKE_SKIP_INSTALL_RULES)
    include(cmake/Util_InstallRules.cmake)
endif()
