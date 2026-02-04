#pragma once 

#include <simd_stl/datapar/Simd.h>


__SIMD_STL_DATAPAR_NAMESPACE_BEGIN

template <template <class> class _Function_>
struct __simd_dispatcher {
private:
    template <
        class       _SpecializedFunction_,
        class...    _Args_>
    simd_stl_always_inline static auto __invoke_simd_helper(_Args_&&... __args) noexcept {
        return _SpecializedFunction_()(std::forward<_Args_>(__args)...);
    }

    template <
        class       _SpecializedFunction_,
        class...    _VectorizedArgs_>
    simd_stl_always_inline static auto __invoke_simd(std::tuple<_VectorizedArgs_...>&& __simd_args) noexcept {
        return std::apply(&__invoke_simd_helper<_SpecializedFunction_, _VectorizedArgs_...>, std::move(__simd_args));
    }
public:
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
                return __invoke_simd<_Function_<datapar::simd512_avx512bw<_Type_>>>(std::move(__simd_args));
        }
        else {
            if (arch::ProcessorFeatures::AVX512F())
                return __invoke_simd<_Function_<datapar::simd512_avx512f<_Type_>>>(std::move(__simd_args));
        }

        if (arch::ProcessorFeatures::AVX2())
            return __invoke_simd<_Function_<datapar::simd256_avx2<_Type_>>>(std::move(__simd_args));

        else if (arch::ProcessorFeatures::SSE42())
            return __invoke_simd<_Function_<datapar::simd128_sse42<_Type_>>>(std::move(__simd_args));

        else if (arch::ProcessorFeatures::SSE2())
            return __invoke_simd<_Function_<datapar::simd128_sse2<_Type_>>>(std::move(__simd_args));

        return std::apply(type_traits::__pass_function(__fallback), std::move(__fallback_args));
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
                return _Function_<datapar::simd512_avx512bw<_Type_>>()(std::forward<_Args_>(__args)...);
        }
        else {
            if (arch::ProcessorFeatures::AVX512F())
                return _Function_<datapar::simd512_avx512f<_Type_>>()(std::forward<_Args_>(__args)...);
        }

        if (arch::ProcessorFeatures::AVX2())
            return _Function_<datapar::simd256_avx2<_Type_>>()(std::forward<_Args_>(__args)...);

        else if (arch::ProcessorFeatures::SSE42())
            return _Function_<datapar::simd128_sse42<_Type_>>()(std::forward<_Args_>(__args)...);

        else if (arch::ProcessorFeatures::SSE2())
            return _Function_<datapar::simd128_sse2<_Type_>>()(std::forward<_Args_>(__args)...);

        return type_traits::invoke(type_traits::__pass_function(__fallback), std::forward<_Args_>(__args)...);
    }
};

__SIMD_STL_DATAPAR_NAMESPACE_END
