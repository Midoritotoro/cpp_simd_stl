#----------------------------------------------------------------
# Generated CMake target import file for configuration "RelWithDebInfo".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "cpp_simd_stl" for configuration "RelWithDebInfo"
set_property(TARGET cpp_simd_stl APPEND PROPERTY IMPORTED_CONFIGURATIONS RELWITHDEBINFO)
set_target_properties(cpp_simd_stl PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELWITHDEBINFO "CXX"
  IMPORTED_LOCATION_RELWITHDEBINFO "${_IMPORT_PREFIX}/lib/cpp_simd_stl.lib"
  )

list(APPEND _cmake_import_check_targets cpp_simd_stl )
list(APPEND _cmake_import_check_files_for_cpp_simd_stl "${_IMPORT_PREFIX}/lib/cpp_simd_stl.lib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
