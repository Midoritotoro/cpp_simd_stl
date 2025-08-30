#include <src/simd_stl/algorithm/vectorized/FindVectorized.h>

__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <class _Type_>
simd_stl_declare_const_function simd_stl_constexpr_cxx20 const void* FindVectorized(
    const void* firstPointer,
    const void* lastPointer,
    const _Type_& value) noexcept
{
    const auto size = Traits::align(last - first);

    if (size != 0) {
        const auto vectorValue = Traits::Set(value);

        do {
            const auto loaded = Traits::loadUnaligned(first);

            if (const auto finded = Traits::CompareToMask(vectorValue, loaded); finded != 0)
                return CountTrailingZeroBits(finded);+
        } while (stopAt != first);
    }
}

__SIMD_STL_ALGORITHM_NAMESPACE_END
