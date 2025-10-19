cmake_minimum_required(VERSION 3.29 FATAL_ERROR)

project(cpp_simd_stlTestsMath LANGUAGES CXX)


set(cpp_simd_stl_DIR "C:/Users/danya/source/repos/Midoritotoro/cpp_simd_stl/out/build")

# if (PROJECT_IS_TOP_LEVEL)
    find_package(cpp_simd_stl REQUIRED COMPONENTS
        cpp_simd_stl 
   )
# endif()

enable_testing()
macro(cpp_simd_stl_math_add_test filename)
    add_executable(${filename} ${filename}.cpp)

    set(CMAKE_CXX_FLAGS "-march=native")
    target_link_libraries(
      ${filename} PRIVATE
      cpp_simd_stl::cpp_simd_stl
    )

    add_test(${filename} ${CMAKE_CURRENT_BINARY_DIR}/"algorithm"/${filename})
    target_compile_features(${filename} PUBLIC cxx_std_23)
endmacro(cpp_simd_stl_math_add_test)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

FILE(GLOB mathTests *.cpp)

FOREACH(testsource ${mathTests})
    GET_FILENAME_COMPONENT(filename ${testsource} NAME_WE)
    cpp_simd_stl_math_add_test(${filename})
ENDFOREACH()