#pragma once 

#include <simd_stl/numeric/Simd.h>


__SIMD_STL_NUMERIC_NAMESPACE_BEGIN

template <template <class> class _Function_>
struct __simd_dispatcher {
    template <
        class       _Type_,
        class       _FallbackFunction_,
        class ...   _VectorizedArgs_,
        class ...   _FallbackArgs_>
    simd_stl_always_inline static auto __apply(
        _FallbackFunction_&&            __fallback,
        std::tuple<_VectorizedArgs_...> __simd_args,
        std::tuple<_FallbackArgs_...>   __fallback_args) noexcept
    {
        if constexpr (sizeof(_Type_)<= 2) {
            if (arch::ProcessorFeatures::AVX512BW())
                return _Function_<numeric::simd512_avx512bw<_Type_>>()(std::forward<_VectorizedArgs_>(__simd_args)...);
        }
        else {
            if (arch::ProcessorFeatures::AVX512F())
                return _Function_<numeric::simd512_avx512f<_Type_>>()(std::forward<_VectorizedArgs_>(__simd_args)...);
        }

        if (arch::ProcessorFeatures::AVX2())
            return _Function_<numeric::simd256_avx2<_Type_>>()(std::forward<_VectorizedArgs_>(__simd_args)...);

        else if (arch::ProcessorFeatures::SSE42())
            return _Function_<numeric::simd128_sse42<_Type_>>()(std::forward<_VectorizedArgs_>(__simd_args)...);

        else if (arch::ProcessorFeatures::SSE2())
            return _Function_<numeric::simd128_sse2<_Type_>>()(std::forward<_VectorizedArgs_>(__simd_args)...);

        return type_traits::invoke(type_traits::__pass_function(__fallback), std::forward<_FallbackArgs_>(__fallback_args)...);
    }

    template <
        class       _Type_,
        class       _FallbackFunction_,
        class ...   _Args_>
    simd_stl_always_inline static auto __apply(
        _FallbackFunction_&&    __fallback,
        _Args_&& ...            __args) noexcept
    {
        if constexpr (sizeof(_Type_) <= 2) {
            if (arch::ProcessorFeatures::AVX512BW())
                return _Function_<numeric::simd512_avx512bw<_Type_>>()(std::forward<_Args_>(__args)...);
        }
        else {
            if (arch::ProcessorFeatures::AVX512F())
                return _Function_<numeric::simd512_avx512f<_Type_>>()(std::forward<_Args_>(__args)...);
        }

        if (arch::ProcessorFeatures::AVX2())
            return _Function_<numeric::simd256_avx2<_Type_>>()(std::forward<_Args_>(__args)...);

        else if (arch::ProcessorFeatures::SSE42())
            return _Function_<numeric::simd128_sse42<_Type_>>()(std::forward<_Args_>(__args)...);

        else if (arch::ProcessorFeatures::SSE2())
            return _Function_<numeric::simd128_sse2<_Type_>>()(std::forward<_Args_>(__args)...);

        return type_traits::invoke(type_traits::__pass_function(__fallback), std::forward<_Args_>(__args)...);
    }
};

__SIMD_STL_NUMERIC_NAMESPACE_END
