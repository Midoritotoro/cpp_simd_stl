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
    "src/simd_stl/algorithm/vectorized/find/FindVectorized.cpp"
)

set(PRIVATE_HEADERS 
    "src/simd_stl/algorithm/vectorized/find/FindVectorized.h"
    
    "src/simd_stl/algorithm/vectorized/find/ContainsVectorized.h"
    src/simd_stl/algorithm/vectorized/find/CountVectorized.h
    "src/simd_stl/algorithm/vectorized/find/FindLastVectorized.h"
    src/simd_stl/algorithm/vectorized/find/FindVectorized.h
    "src/simd_stl/algorithm/vectorized/find/MismatchVectorized.h"
    src/simd_stl/algorithm/vectorized/replace/ReplaceVectorized.h
    src/simd_stl/algorithm/vectorized/replace/ReplaceCopyVectorized.h
    src/simd_stl/algorithm/vectorized/order/ReverseCopyVectorized.h
    src/simd_stl/algorithm/vectorized/order/ReverseVectorized.h
    src/simd_stl/algorithm/vectorized/swap/SwapRangesVectorized.h
    "src/simd_stl/algorithm/vectorized/find/EqualVectorized.h"
    "src/simd_stl/algorithm/vectorized/find/FindEndVectorized.h"
    src/simd_stl/algorithm/vectorized/copy/CopyVectorized.h
    src/simd_stl/algorithm/vectorized/copy/MoveVectorized.h
    src/simd_stl/algorithm/vectorized/fill/FillVectorized.h
    src/simd_stl/algorithm/vectorized/remove/RemoveVectorized.h
    src/simd_stl/algorithm/vectorized/remove/RemoveCopyVectorized.h
    src/simd_stl/algorithm/vectorized/transform/TransformVectorized.h
    src/simd_stl/algorithm/vectorized/minmax/MinmaxElementVectorized.h
    src/simd_stl/algorithm/vectorized/minmax/MinElementVectorized.h
    src/simd_stl/algorithm/vectorized/minmax/MaxElementVectorized.h
    src/simd_stl/algorithm/vectorized/minmax/MinVectorized.h
    src/simd_stl/algorithm/vectorized/minmax/MaxVectorized.h
    src/simd_stl/algorithm/vectorized/minmax/MinmaxVectorized.h

    src/simd_stl/algorithm/AlgorithmDebug.h
    src/simd_stl/algorithm/AdvanceBytes.h

    "src/simd_stl/algorithm/unchecked/find/FindUnchecked.h"
    "src/simd_stl/algorithm/unchecked/find/FindIfUnchecked.h"
    "src/simd_stl/algorithm/unchecked/find/FindIfNotUnchecked.h"
    "src/simd_stl/algorithm/unchecked/find/ContainsUnchecked.h"
    "src/simd_stl/algorithm/unchecked/find/MismatchUnchecked.h"
    "src/simd_stl/algorithm/unchecked/find/FindEndUnchecked.h" 
    "src/simd_stl/algorithm/unchecked/find/SearchUnchecked.h"
    "src/simd_stl/algorithm/unchecked/find/FindLastUnchecked.h"
    "src/simd_stl/algorithm/unchecked/find/EqualUnchecked.h"
    "src/simd_stl/algorithm/unchecked/find/CountUnchecked.h"
    "src/simd_stl/algorithm/unchecked/find/CountIfUnchecked.h"
    "src/simd_stl/algorithm/unchecked/find/FindLastIfNotUnchecked.h"
    "src/simd_stl/algorithm/unchecked/find/FindLastIfUnchecked.h"

    src/simd_stl/algorithm/unchecked/copy/CopyUnchecked.h
    src/simd_stl/algorithm/unchecked/copy/CopyIfUnchecked.h
    src/simd_stl/algorithm/unchecked/copy/CopyBackwardUnchecked.h
    src/simd_stl/algorithm/unchecked/copy/MoveUnchecked.h
    src/simd_stl/algorithm/unchecked/copy/MoveBackwardUnchecked.h

    src/simd_stl/algorithm/unchecked/replace/ReplaceCopyIfUnchecked.h
    src/simd_stl/algorithm/unchecked/replace/ReplaceCopyUnchecked.h

    src/simd_stl/algorithm/unchecked/replace/ReplaceIfUnchecked.h
    src/simd_stl/algorithm/unchecked/replace/ReplaceUnchecked.h

    src/simd_stl/algorithm/unchecked/transform/TransformUnchecked.h

    src/simd_stl/algorithm/unchecked/batch/ForEachUnchecked.h
    src/simd_stl/algorithm/unchecked/batch/ForEachNUnchecked.h

    src/simd_stl/algorithm/unchecked/remove/RemoveUnchecked.h
    src/simd_stl/algorithm/unchecked/remove/RemoveIfUnchecked.h

    src/simd_stl/algorithm/unchecked/remove/RemoveCopyUnchecked.h
    src/simd_stl/algorithm/unchecked/remove/RemoveCopyIfUnchecked.h

    src/simd_stl/algorithm/unchecked/minmax/MinmaxElementUnchecked.h
    src/simd_stl/algorithm/unchecked/minmax/MinElementUnchecked.h
    src/simd_stl/algorithm/unchecked/minmax/MaxElementUnchecked.h

    src/simd_stl/algorithm/unchecked/minmax/MinUnchecked.h
    src/simd_stl/algorithm/unchecked/minmax/MaxUnchecked.h
    src/simd_stl/algorithm/unchecked/minmax/MinmaxUnchecked.h

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
    src/simd_stl/type_traits/FunctionInformation.h

    "src/simd_stl/utility/Assert.h"

    src/simd_stl/datapar/SimdBroadcast.h
    src/simd_stl/datapar/ShuffleTables.h
    src/simd_stl/datapar/SimdDivisors.h
    src/simd_stl/datapar/SimdArithmetic.h
    src/simd_stl/datapar/IntrinBitcast.h
    src/simd_stl/datapar/SimdConvert.h
    src/simd_stl/datapar/SimdCompare.h
    src/simd_stl/datapar/SimdElementAccess.h
    src/simd_stl/datapar/SimdElementWise.h
    src/simd_stl/datapar/SimdIntegralTypesCheck.h
    src/simd_stl/datapar/SimdMemoryAccess.h

    src/simd_stl/datapar/SimdBroadcast.inl
    src/simd_stl/datapar/SimdArithmetic.inl
    src/simd_stl/datapar/SimdConvert.inl
    src/simd_stl/datapar/SimdCompare.inl
    src/simd_stl/datapar/SimdElementAccess.inl
    src/simd_stl/datapar/SimdElementWise.inl
    src/simd_stl/datapar/SimdMemoryAccess.inl
    src/simd_stl/datapar/SimdIndexMask.inl
    src/simd_stl/datapar/SimdComparison.h
    src/simd_stl/datapar/SimdCompareResult.inl
    src/simd_stl/datapar/SimdCompareAdapters.h
    src/simd_stl/datapar/SimdCompareAdapters.inl
    src/simd_stl/datapar/SimdMaskCommonOperations.h
    src/simd_stl/datapar/SimdMaskCommonOperations.inl
    src/simd_stl/datapar/SimdElementMaskOperations.h
    src/simd_stl/datapar/SimdIndexMaskOperations.h
    src/simd_stl/datapar/SimdElementMaskOperations.inl
    src/simd_stl/datapar/SimdIndexMaskOperations.inl
    src/simd_stl/datapar/MaskTypeSelector.h
    src/simd_stl/datapar/SimdMaskTypeCheck.h


    src/simd_stl/datapar/SimdOperations.h
    src/simd_stl/datapar/ZmmThreshold.h
    src/simd_stl/datapar/CachePrefetcher.h
    src/simd_stl/datapar/MaskExpand.h

    src/simd_stl/datapar/BasicSimd.inl
    src/simd_stl/datapar/SimdCast.inl
    src/simd_stl/datapar/BasicSimdElementReference.inl
    src/simd_stl/datapar/IsComparable.h
    src/simd_stl/datapar/SimdDispatcher.h
    src/simd_stl/datapar/SizedSimdDispatcher.h
    src/simd_stl/concurrency/WindowsThread.h
    src/simd_stl/concurrency/ThreadYield.h
    src/simd_stl/concurrency/WindowsThreadPool.h
    src/simd_stl/concurrency/WindowsThreadPoolTasks.h
    
    "src/simd_stl/algorithm/vectorized/find/SearchVectorized.h"
    src/simd_stl/algorithm/vectorized/traits/SearchLoop.h
    src/simd_stl/algorithm/FixedMemcmp.h
    src/simd_stl/algorithm/MsvcIteratorUnwrap.h

    src/simd_stl/datapar/bitwise/BitAnd.h
    src/simd_stl/datapar/bitwise/BitOr.h
    src/simd_stl/datapar/bitwise/BitXor.h
    src/simd_stl/datapar/bitwise/BitNot.h
    src/simd_stl/datapar/bitwise/ToMask.h
    src/simd_stl/datapar/bitwise/ToVector.h
    src/simd_stl/datapar/bitwise/ToIndexMask.h
    src/simd_stl/datapar/bitwise/MaskConvert.h

    src/simd_stl/datapar/arithmetic/Sub.h
    src/simd_stl/datapar/arithmetic/Add.h
    src/simd_stl/datapar/arithmetic/Mul.h
    src/simd_stl/datapar/arithmetic/Div.h
    src/simd_stl/datapar/arithmetic/Negate.h
    src/simd_stl/datapar/arithmetic/VerticalMax.h
    src/simd_stl/datapar/arithmetic/VerticalMin.h
    src/simd_stl/datapar/arithmetic/Abs.h

    src/simd_stl/datapar/compare/Less.h
    src/simd_stl/datapar/compare/LessEqual.h
    src/simd_stl/datapar/compare/Greater.h
    src/simd_stl/datapar/compare/GreaterEqual.h
    src/simd_stl/datapar/compare/Equal.h
    src/simd_stl/datapar/compare/NotEqual.h

    src/simd_stl/datapar/memory/CompressStore.h
    src/simd_stl/datapar/memory/Load.h
    src/simd_stl/datapar/memory/Store.h
    src/simd_stl/datapar/memory/MaskLoad.h
    src/simd_stl/datapar/memory/MaskzLoad.h
    src/simd_stl/datapar/memory/MaskStore.h
    src/simd_stl/datapar/memory/NonTemporalLoad.h
    src/simd_stl/datapar/memory/NonTemporalStore.h
    src/simd_stl/datapar/memory/StreamingFence.h
    src/simd_stl/datapar/memory/AlignmentPolicy.h

    src/simd_stl/datapar/reduce/ReduceAdd.h
    src/simd_stl/datapar/reduce/ReduceAddType.h
    src/simd_stl/datapar/reduce/Fold.h
    src/simd_stl/datapar/reduce/HorizontalMax.h
    src/simd_stl/datapar/reduce/HorizontalMin.h

    src/simd_stl/datapar/shuffle/Insert.h
    src/simd_stl/datapar/shuffle/Extract.h
    src/simd_stl/datapar/shuffle/Compress.h
    src/simd_stl/datapar/shuffle/Broadcast.h
    src/simd_stl/datapar/shuffle/BroadcastZeros.h
    src/simd_stl/datapar/shuffle/Reverse.h
    src/simd_stl/datapar/shuffle/Blend.h

)

set(PUBLIC_HEADERS
    include/simd_stl/algorithm/Algorithm.h

    include/simd_stl/math/Math.h
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
    include/simd_stl/concurrency/ThreadPool.h
    include/simd_stl/concurrency/Execution.h

    include/simd_stl/concurrency/algorithm/find/ParallelFind.h
    include/simd_stl/concurrency/algorithm/find/ParallelFindIf.h
    include/simd_stl/concurrency/algorithm/find/ParallelFindIfNot.h

    include/simd_stl/concurrency/algorithm/find/ParallelAllOf.h
    include/simd_stl/concurrency/algorithm/find/ParallelAnyOf.h
    include/simd_stl/concurrency/algorithm/find/ParallelNoneOf.h

    include/simd_stl/algorithm/batch/ForEach.h
    include/simd_stl/algorithm/batch/ForEachN.h

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
    include/simd_stl/compatibility/StaticOperators.h

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
    include/simd_stl/algorithm/replace/ReplaceCopy.h

    include/simd_stl/algorithm/order/Reverse.h
    include/simd_stl/algorithm/order/ReverseCopy.h

    include/simd_stl/algorithm/minmax/Min.h
    include/simd_stl/algorithm/minmax/Max.h
    include/simd_stl/algorithm/minmax/Minmax.h
    include/simd_stl/algorithm/minmax/MaxElement.h
    include/simd_stl/algorithm/minmax/MinElement.h
    include/simd_stl/algorithm/minmax/MinmaxElement.h



    include/simd_stl/system/Handle.h

    include/simd_stl/memory/Intersects.h
    include/simd_stl/memory/PointerToIntegral.h
    include/simd_stl/memory/Alignment.h

    include/simd_stl/datapar/Simd.h
    include/simd_stl/datapar/BasicSimd.h
    include/simd_stl/datapar/BasicSimdElementReference.h
    include/simd_stl/datapar/BasicSimdMask.h
    include/simd_stl/datapar/BasicSimdShuffleMask.h
    include/simd_stl/datapar/SimdCast.h
    include/simd_stl/datapar/SimdIndexMask.h
    include/simd_stl/datapar/SimdCompareResult.h
    include/simd_stl/datapar/SimdDataparAlgorithms.h
    include/simd_stl/datapar/SimdConfig.h
)

include(CMakePackageConfigHelpers)
include(GenerateExportHeader)

message(${CMAKE_CXX_COMPILER})

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

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_options(-g -Ob3) 
    elseif (CMAKE_BUILD_TYPE STREQUAL "Release")
        add_compile_options(-O2 -Ot -Ob3 -GS- -Oy)
    endif()

    target_link_options(cpp_simd_stl PRIVATE "/LTCG;/opt:ref,icf;")

    target_compile_options(
        cpp_simd_stl 
        PRIVATE
            /bigobj
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
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(
        cpp_simd_stl 
        PRIVATE
            /permissive-
            /W4
            /utf-8
            /MP
            /EHsc
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

set(SIMD_STL_FORCE_LIST
    SIMD_STL_FORCE_SSE2
    SIMD_STL_FORCE_SSE3
    SIMD_STL_FORCE_SSSE3
    SIMD_STL_FORCE_SSE41
    SIMD_STL_FORCE_SSE42
    SIMD_STL_FORCE_AVX
    SIMD_STL_FORCE_AVX2
    SIMD_STL_FORCE_AVX512F
    SIMD_STL_FORCE_AVX512BW
    SIMD_STL_FORCE_AVX512CD
    SIMD_STL_FORCE_AVX512DQ
    SIMD_STL_FORCE_AVX512VL
    SIMD_STL_FORCE_AVX512VBMI
    SIMD_STL_FORCE_AVX512VBMI2
)

set(SIMD_STL_USER_FORCE_ENABLED OFF)

foreach(flag ${SIMD_STL_FORCE_LIST})
    if(DEFINED ${flag})
        set(SIMD_STL_USER_FORCE_ENABLED ON)
    endif()
endforeach()

function(simd_stl_test_msvc_arch flag result_var) 
    set(src "${CMAKE_BINARY_DIR}/simd_test_${flag}.cpp") 
    file(WRITE "${src}" "int main(){return 0;}") 

    try_compile(${result_var} 
        "${CMAKE_BINARY_DIR}" 
        "${src}" CMAKE_FLAGS 
        "-DCMAKE_CXX_FLAGS=/arch:${flag}" 
        OUTPUT_VARIABLE out) 
endfunction()

if(SIMD_STL_USER_FORCE_ENABLED)
    message(STATUS "SIMD_STL: ISA FORCE mode enabled")

    if(MSVC)

        message(STATUS "SIMD_STL: MSVC detected — validating /arch: flags")

        if(SIMD_STL_FORCE_SSE2)
            add_compile_definitions(SIMD_STL_FORCE_SSE2=1)
        endif()

        if(SIMD_STL_FORCE_AVX)
            simd_stl_test_msvc_arch("AVX" AVX_OK)
            if(NOT AVX_OK)
                message(FATAL_ERROR "MSVC does not support /arch:AVX")
            endif()
            add_compile_options(/arch:AVX)
            add_compile_definitions(SIMD_STL_FORCE_AVX=1)
        endif()

        if(SIMD_STL_FORCE_AVX2)
            simd_stl_test_msvc_arch("AVX2" AVX2_OK)
            if(NOT AVX2_OK)
                message(FATAL_ERROR "MSVC does not support /arch:AVX2")
            endif()
            add_compile_options(/arch:AVX2)
            add_compile_definitions(SIMD_STL_FORCE_AVX2=1)
        endif()


        if(SIMD_STL_FORCE_AVX512F OR SIMD_STL_FORCE_AVX512BW OR SIMD_STL_FORCE_AVX512CD
           OR SIMD_STL_FORCE_AVX512DQ OR SIMD_STL_FORCE_AVX512VL
           OR SIMD_STL_FORCE_AVX512VBMI OR SIMD_STL_FORCE_AVX512VBMI2)

            simd_stl_test_msvc_arch("AVX512" AVX512_OK)
            if(NOT AVX512_OK)
                message(FATAL_ERROR "MSVC does not support /arch:AVX512")
            endif()

            add_compile_options(/arch:AVX512)

            add_compile_definitions(
                SIMD_STL_FORCE_AVX512F=1
                SIMD_STL_FORCE_AVX512BW=1
                SIMD_STL_FORCE_AVX512CD=1
                SIMD_STL_FORCE_AVX512DQ=1
                SIMD_STL_FORCE_AVX512VL=1
                SIMD_STL_FORCE_AVX512VBMI=1
                SIMD_STL_FORCE_AVX512VBMI2=1
            )
        endif()

    else()
        include(CheckCXXCompilerFlag)

        macro(simd_stl_require flag compileflag)
            if(${flag})
                check_cxx_compiler_flag(${compileflag} ${flag}_SUPPORTED)
                if(NOT ${flag}_SUPPORTED)
                    message(FATAL_ERROR "${flag}=1 requires ${compileflag}, but compiler does not support it")
                endif()
                add_compile_options(${compileflag})
                add_compile_definitions(${flag}=1)
            endif()
        endmacro()

        simd_stl_require(SIMD_STL_FORCE_SSE2        "-msse2")
        simd_stl_require(SIMD_STL_FORCE_SSE3        "-msse3")
        simd_stl_require(SIMD_STL_FORCE_SSSE3       "-mssse3")
        simd_stl_require(SIMD_STL_FORCE_SSE41       "-msse4.1")
        simd_stl_require(SIMD_STL_FORCE_SSE42       "-msse4.2")
        simd_stl_require(SIMD_STL_FORCE_AVX         "-mavx")
        simd_stl_require(SIMD_STL_FORCE_AVX2        "-mavx2")
        simd_stl_require(SIMD_STL_FORCE_AVX512F     "-mavx512f")
        simd_stl_require(SIMD_STL_FORCE_AVX512BW    "-mavx512bw")
        simd_stl_require(SIMD_STL_FORCE_AVX512CD    "-mavx512cd")
        simd_stl_require(SIMD_STL_FORCE_AVX512DQ    "-mavx512dq")
        simd_stl_require(SIMD_STL_FORCE_AVX512VL    "-mavx512vl")
        simd_stl_require(SIMD_STL_FORCE_AVX512VBMI  "-mavx512vbmi")
        simd_stl_require(SIMD_STL_FORCE_AVX512VBMI2 "-mavx512vbmi2")

    endif()

    add_compile_definitions(SIMD_STL_ISA_FORCE_ENABLED=1)

else()

    message(STATUS "SIMD_STL: No FORCE ISA specified — using runtime dispatch")
    add_compile_definitions(SIMD_STL_ISA_FORCE_ENABLED=0)

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
