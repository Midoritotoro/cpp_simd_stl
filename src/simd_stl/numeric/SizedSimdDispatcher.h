#pragma once 

#include <simd_stl/numeric/Simd.h>
#include <src/simd_stl/numeric/ZmmThreshold.h>


__SIMD_STL_NUMERIC_NAMESPACE_BEGIN

template <template <class> class _Function_>
struct __simd_sized_dispatcher {
    template <
        class       _Type_,
        class       _SizeType_,
        class       _FallbackFunction_,
        class ...   _Args_>
    simd_stl_always_inline static auto __apply(
        _SizeType_              __size,
        _FallbackFunction_&&    __fallback,
        _Args_&& ...            __args) noexcept 
    {
        if (__size >= __zmm_threshold<_Type_>) {
            if constexpr (sizeof(_Type_) <= 2) {
                using _Simd_                = simd512_avx512bw<_Type_>;

                const auto __aligned_size   = __size & (~(sizeof(_Simd_) - 1));
                const auto __tail_size      = __size & (sizeof(_Simd_) - sizeof(_Type_));

                if (__aligned_size != 0 && arch::ProcessorFeatures::AVX512BW())
                    return _Function_<_Simd_>()(__aligned_size, __tail_size, std::forward<_Args_>(__args)...);
            }
            else {
                using _Simd_                = simd512_avx512f<_Type_>;

                const auto __aligned_size   = __size & (~(sizeof(_Simd_) - 1));
                const auto __tail_size      = __size & (sizeof(_Simd_) - sizeof(_Type_));

                if (__aligned_size != 0 && arch::ProcessorFeatures::AVX512F())
                    return _Function_<_Simd_>()(__aligned_size, __tail_size, std::forward<_Args_>(__args)...);
            }
        }
        else {
            if constexpr (sizeof(_Type_) <= 2) {
                using _Simd_                = simd256_avx512vlbw<_Type_>;

                const auto __aligned_size   = __size & (~(sizeof(_Simd_) - 1));
                const auto __tail_size      = __size & (sizeof(_Simd_) - sizeof(_Type_));

                if (__aligned_size != 0 && arch::ProcessorFeatures::AVX512BW() && arch::ProcessorFeatures::AVX512VL())
                    return _Function_<_Simd_>()(__aligned_size, __tail_size, std::forward<_Args_>(__args)...);
            }
            else {
                using _Simd_                = simd256_avx512vlf<_Type_>;

                const auto __aligned_size   = __size & (~(sizeof(_Simd_) - 1));
                const auto __tail_size      = __size & (sizeof(_Simd_) - sizeof(_Type_));

                if (__aligned_size != 0 && arch::ProcessorFeatures::AVX512F() && arch::ProcessorFeatures::AVX512VL())
                    return _Function_<_Simd_>()(__aligned_size, __tail_size, std::forward<_Args_>(__args)...);
            }
        }

        if (const auto __aligned_size = __size & (~(sizeof(simd256_avx2<_Type_>) - 1)); __aligned_size != 0 && arch::ProcessorFeatures::AVX2()) {
            using _Simd_ = simd256_avx2<_Type_>;
            const auto __tail_size = __size & (sizeof(_Simd_) - sizeof(_Type_));

            return _Function_<_Simd_>()(__aligned_size, __tail_size, std::forward<_Args_>(__args)...);
        }

        if (const auto __aligned_size = __size & (~(sizeof(simd256_avx2<_Type_>) - 1)); __aligned_size != 0 && arch::ProcessorFeatures::SSE2()) {
            using _Simd_ = simd128_sse2<_Type_>;
            const auto __tail_size = __size & (sizeof(_Simd_) - sizeof(_Type_));

            return _Function_<_Simd_>()(__aligned_size, __tail_size, std::forward<_Args_>(__args)...);
        }

        return type_traits::invoke(type_traits::__pass_function(__fallback), std::forward<_Args_>(__args)...);
    }
};

__SIMD_STL_NUMERIC_NAMESPACE_END
