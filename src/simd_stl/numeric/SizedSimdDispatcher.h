#pragma once 

#include <simd_stl/numeric/Simd.h>
#include <src/simd_stl/numeric/ZmmThreshold.h>


__SIMD_STL_NUMERIC_NAMESPACE_BEGIN

template <template <class> class _Function_>
struct __simd_sized_dispatcher {
private:
    template <
        class _Simd_,
        class _SizeType_>
    simd_stl_always_inline static std::pair<_SizeType_, _SizeType_> __sizes(_SizeType_ __size) noexcept {
        return { 
            __size & (~(sizeof(_Simd_) - 1)),
            __size & (sizeof(_Simd_) - sizeof(typename _Simd_::value_type)) 
        };
    }

    template <
        class       _SpecializedFunction_,
        class       _SizeType_,
        class...    _VectorizedArgs_> 
    simd_stl_always_inline static auto __invoke_simd(
        _SizeType_                          __aligned_size, 
        _SizeType_                          __tail_size, 
        std::tuple<_VectorizedArgs_...>&&   __simd_args) noexcept 
    {
        return std::apply([&](auto&&... __args) {
            return _SpecializedFunction_()(__aligned_size, __tail_size, std::forward<decltype(__args)>(__args)...);
        }, std::move(__simd_args)); 
    }
public:
    template <
        class       _Type_,
        class       _SizeType_,
        class       _FallbackFunction_,
        class ...   _VectorizedArgs_,
        class ...   _FallbackArgs_>
    simd_stl_always_inline static auto __apply(
        _SizeType_                      __size,
        _FallbackFunction_&&            __fallback,
        std::tuple<_VectorizedArgs_...> __simd_args,
        std::tuple<_FallbackArgs_...>   __fallback_args) noexcept
    {
        if (__size >= __zmm_threshold<_Type_>) {
            if constexpr (sizeof(_Type_) <= 2) {
                using _Simd_                = simd512_avx512bw<_Type_>;
                const auto __calculated     = __sizes<_Simd_>(__size);

                if (__calculated.first != 0 && arch::ProcessorFeatures::AVX512BW())
                    return __invoke_simd<_Function_<_Simd_>>(__calculated.first, __calculated.second, std::move(__simd_args));
            }
            else {
                using _Simd_                = simd512_avx512f<_Type_>;
                const auto __calculated     = __sizes<_Simd_>(__size);

                if (__calculated.first != 0 && arch::ProcessorFeatures::AVX512F())
                    return __invoke_simd<_Function_<_Simd_>>(__calculated.first, __calculated.second, std::move(__simd_args));
            }
        }
        else {
            if constexpr (sizeof(_Type_) <= 2) {
                using _Simd_                = simd256_avx512vlbw<_Type_>;
                const auto __calculated     = __sizes<_Simd_>(__size);

                if (__calculated.first != 0 && arch::ProcessorFeatures::AVX512BW() && arch::ProcessorFeatures::AVX512VL())
                    return __invoke_simd<_Function_<_Simd_>>(__calculated.first, __calculated.second, std::move(__simd_args));
            }
            else {
                using _Simd_                = simd256_avx512vlf<_Type_>;
                const auto __calculated     = __sizes<_Simd_>(__size);

                if (__calculated.first != 0 && arch::ProcessorFeatures::AVX512F() && arch::ProcessorFeatures::AVX512VL())
                    return __invoke_simd<_Function_<_Simd_>>(__calculated.first, __calculated.second, std::move(__simd_args));
            }
        }

        if (const auto __calculated = __sizes<simd256_avx2<_Type_>>(__size); __calculated.first != 0 && arch::ProcessorFeatures::AVX2()) {
            using _Simd_ = simd256_avx2<_Type_>;
            return __invoke_simd<_Function_<_Simd_>>(__calculated.first, __calculated.second, std::move(__simd_args));
        }

        else if (const auto __calculated = __sizes<simd128_sse2<_Type_>>(__size); __calculated.first != 0 && arch::ProcessorFeatures::SSE2()) {
            using _Simd_ = simd128_sse2<_Type_>;
            return __invoke_simd<_Function_<_Simd_>>(__calculated.first, __calculated.second, std::move(__simd_args));
        }

        return std::apply(type_traits::__pass_function(__fallback), std::move(__fallback_args));
    }
};

__SIMD_STL_NUMERIC_NAMESPACE_END
