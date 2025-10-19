cmake_minimum_required(VERSION 4.1  FATAL_ERROR)

project(cpp_simd_stlTests LANGUAGES CXX)

set(cpp_simd_stl_DIR "C:/Users/danya/source/repos/Midoritotoro/cpp_simd_stl/out/build")

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(cpp_simd_stl)


add_subdirectory(algorithm)
add_subdirectory(math)
