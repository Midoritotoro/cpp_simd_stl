include(CMakeFindDependencyMacro)

if (EXISTS "${CMAKE_CURRENT_LIST_DIR}/cpp_simd_stlTargets.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/cpp_simd_stlTargets.cmake")
endif()