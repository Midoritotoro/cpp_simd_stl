#include <src/simd_stl/algorithm/vectorized/FindVectorized.h>
#include <simd_stl/arch/ProcessorFeatures.h>

#include <src/simd_stl/algorithm/vectorized/traits/FindVectorizedTraits.h>
#include <src/simd_stl/algorithm/AdvanceBytes.h>

#include <src/simd_stl/math/BitMath.h>
#include <simd_stl/compatibility/Inline.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <typename _VectorType_>
constexpr bool is_xmm_type_v = std::_Is_any_of_v<_VectorType_, __m128, __m128i, __m128d>;

template <typename _VectorType_>
constexpr bool is_ymm_type_v = std::_Is_any_of_v<_VectorType_, __m256, __m256i, __m256d>;

template <typename _VectorType_>
constexpr bool is_zmm_type_v = std::_Is_any_of_v<_VectorType_, __m512, __m512i, __m512d>;

template <typename _FindedType_>
using mask_type_t = std::conditional_t<
    std::is_integral_v<_FindedType_>, _FindedType_, std::conditional_t<
    is_xmm_type_v<_FindedType_>, uint16, std::conditional_t<
    is_ymm_type_v<_FindedType_>, uint32, std::conditional_t<
    is_zmm_type_v<_FindedType_>, uint64, void>>>>;


template <class _Type_>
simd_stl_declare_const_function simd_stl_always_inline simd_stl_constexpr_cxx20 const void* FindScalar(
    const void* firstPointer,
    const void* lastPointer,
    _Type_      value) noexcept 
{
    auto pointer = static_cast<const _Type_*>(firstPointer);

    while (pointer != lastPointer && *pointer != value)
        ++pointer;

    return pointer;
}

template <
    class _Traits_,
    class _Type_>
simd_stl_declare_const_function simd_stl_always_inline simd_stl_constexpr_cxx20 const void* FindVectorizedInternal(
    const void* firstPointer,
    const void* lastPointer,
    _Type_      value) noexcept
{
    const auto sizeInBytes  = lastPointer - firstPointer;
    const auto alignedSize  = sizeInBytes & (~_Traits_::portionSize);

    if (alignedSize != 0) {
        const auto comparand = _Traits_::Set(value);

        do {
            const auto loaded = _Traits_::LoadUnaligned(firstPointer);
            const auto finded = _Traits_::Compare(comparand, loaded);

            mask_type_t<decltype(finded)> mask = 0;

            if constexpr (std::is_integral_v<decltype(finded)>)
                mask = finded;
            else 
                mask = _Traits_::ToMask(finded);

            if (mask != 0)
                return math::CountTrailingZeroBits(finded);

            AdvanceBytes(firstPointer, _Traits_::portionSize);
        } while (firstPointer != lastPointer);
    }

    return FindScalar(firstPointer, lastPointer, value);
}

template <class _Type_>
simd_stl_declare_const_function simd_stl_constexpr_cxx20 const void* FindVectorized(
    const void* firstPointer,
    const void* lastPointer,
    _Type_      value) noexcept
{
    if (arch::ProcessorFeatures::AVX512F())
        return FindVectorizedInternal<FindTraits<arch::CpuFeature::AVX512F>, _Type_>(firstPointer, lastPointer, value);
    else if (arch::ProcessorFeatures::AVX2())
        return FindVectorizedInternal<FindTraits<arch::CpuFeature::AVX2>, _Type_>(firstPointer, lastPointer, value);
    else if (arch::ProcessorFeatures::SSE2())
        return FindVectorizedInternal<FindTraits<arch::CpuFeature::SSE2>, _Type_>(firstPointer, lastPointer, value);
}


__SIMD_STL_ALGORITHM_NAMESPACE_END
