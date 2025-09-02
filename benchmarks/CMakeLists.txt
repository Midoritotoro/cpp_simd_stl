cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

project(cpp_simd_stlBenchmarks LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (PROJECT_IS_TOP_LEVEL)
    find_package(cpp_simd_stl REQUIRED)
endif()

enable_testing()
set(CMAKE_MODULE_PATH
    ${CMAKE_MODULE_PATH}
    ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks
)

set(benchmark_DIR "C:/Users/danya/source/repos/Midoritotoro/Libraries/win64/buil/d")
find_package(benchmark REQUIRED)

macro(cpp_simd_stlBenchmarks_add_benchmark filename)
    add_executable(${filename} ${filename}.cpp ${CMAKE_CURRENT_SOURCE_DIR}/tools/BenchmarkHelper.h)
    target_include_directories(${filename} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/tools)

    set(CMAKE_CXX_FLAGS "-march=native")
    target_link_libraries(
      ${filename} PUBLIC
      cpp_simd_stl::cpp_simd_stl
      benchmark::benchmark
    )

    add_test(${filename} ${CMAKE_CURRENT_BINARY_DIR}/${filename})
endmacro(cpp_simd_stlBenchmarks_add_benchmark)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

FILE(GLOB cpp_simd_stlBenchmarks *.cpp *.h)

FOREACH(benchsource ${cpp_simd_stlBenchmarks})
   GET_FILENAME_COMPONENT(filename ${benchsource} NAME_WE)
   cpp_simd_stlBenchmarks_add_benchmark(${filename})
ENDFOREACH()
