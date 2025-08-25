cmake_minimum_required (VERSION 3.29 FATAL_ERROR)

project ("cpp_simd_stl"
    VERSION 1.0.0
    DESCRIPTION "..."
    LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_LINK_ $<IF:$<CONFIG:Debug>, /NODEFAULTLIB:LIBCMT, /DEBUG;/OPT:REF>
       /INCREMENTAL:NO)

if (NOT DEFINED CMAKE_INSTALL_LIBDIR)
  set(CMAKE_INSTALL_LIBDIR "lib")
endif()
if (NOT DEFINED CMAKE_INSTALL_BINDIR)
  set(CMAKE_INSTALL_BINDIR "bin")
endif()
if (NOT DEFINED CMAKE_INSTALL_INCLUDEDIR)
  set(CMAKE_INSTALL_INCLUDEDIR "include")
endif()

set(libs_loc "D:/Libraries/win64")


option(CPP_SIMD_STL_INSTALL "Генерация цели для установки библиотеки" ${PROJECT_IS_TOP_LEVEL})
set(CPP_SIMD_STL_INSTALL_CMAKE_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/lib_base")


set(PRIVATE_SOURCES
)

set(PRIVATE_HEADERS
)

set(PUBLIC_HEADERS
)

include(CMakePackageConfigHelpers)
include(GenerateExportHeader)

add_library(cpp_simd_stl STATIC ${PRIVATE_SOURCES} ${PUBLIC_HEADERS})
add_library(cpp_simd_stl::cpp_simd_stl ALIAS cpp_simd_stl)

include(cmake/Util_ExportConfig.cmake)

include(cmake/Util_FindQt.cmake)

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
        add_compile_options(-O2 -Ot)
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