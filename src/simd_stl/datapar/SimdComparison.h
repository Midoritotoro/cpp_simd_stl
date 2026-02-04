#pragma once 

#include <simd_stl/Types.h>


__SIMD_STL_DATAPAR_NAMESPACE_BEGIN

enum class __simd_comparison: int32 { 
	equal, 
	not_equal, 
	less,
	greater,
	less_equal,
	greater_equal
};

__SIMD_STL_DATAPAR_NAMESPACE_END
