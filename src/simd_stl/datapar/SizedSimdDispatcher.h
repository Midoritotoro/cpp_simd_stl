#pragma once 

#include <simd_stl/datapar/Simd.h>
#include <src/simd_stl/datapar/ZmmThreshold.h>


__SIMD_STL_DATAPAR_NAMESPACE_BEGIN

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
        class...    _Args>
    simd_stl_always_inline static auto __invoke_simd_helper(
        _SizeType_ __aligned_size,
        _SizeType_ __tail_size,
        _Args&&... __args) noexcept
    {
        return _SpecializedFunction_()(__aligned_size, __tail_size, std::forward<_Args>(__args)...);
    }

    template <
        class       _SpecializedFunction_,
        class       _SizeType_,
        class...    _VectorizedArgs_>
    simd_stl_always_inline static auto __invoke_simd(
        _SizeType_                          __aligned_size,
        _SizeType_                          __tail_size,
        std::tuple<_VectorizedArgs_...>&& __simd_args) noexcept
    {
        return std::apply(&__invoke_simd_helper<_SpecializedFunction_, _SizeType_, _VectorizedArgs_...>,
            std::tuple_cat(std::make_tuple(__aligned_size, __tail_size), std::move(__simd_args)));
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
//#if 0
//        if (__size >= __zmm_threshold<_Type_>) {
//            if constexpr (sizeof(_Type_) <= 2) {
//                if (const auto __aligned_size = __size & (~(sizeof(simd512_avx512bw<_Type_>) - 1)); __aligned_size != 0 && arch::ProcessorFeatures::AVX512BW())
//                    return __invoke_simd<_Function_<simd512_avx512bw<_Type_>>>(__aligned_size, __size & (sizeof(simd512_avx512bw<_Type_>) -
//                        sizeof(typename simd512_avx512bw<_Type_>::value_type)), std::move(__simd_args));
//            }
//            else {
//                if (const auto __aligned_size = __size & (~(sizeof(simd512_avx512f<_Type_>) - 1)); __aligned_size != 0 && arch::ProcessorFeatures::AVX512F())
//                    return __invoke_simd<_Function_<simd512_avx512f<_Type_>>>(__aligned_size, __size & (sizeof(simd512_avx512f<_Type_>) -
//                        sizeof(typename simd512_avx512f<_Type_>::value_type)), std::move(__simd_args));
//            }
//        }
//        else {
//            if constexpr (sizeof(_Type_) <= 2) {
//                if (const auto __aligned_size = __size & (~(sizeof(simd256_avx512vlbw<_Type_>) - 1)); __aligned_size != 0 && arch::ProcessorFeatures::AVX512BW() && arch::ProcessorFeatures::AVX512VL())
//                    return __invoke_simd<_Function_<simd256_avx512vlbw<_Type_>>>(__aligned_size, __size & (sizeof(simd256_avx512vlbw<_Type_>) -
//                        sizeof(typename simd256_avx512vlbw<_Type_>::value_type)), std::move(__simd_args));
//            }
//            else {
//                if (const auto __aligned_size = __size & (~(sizeof(simd256_avx512vlf<_Type_>) - 1)); __aligned_size != 0 && arch::ProcessorFeatures::AVX512F() && arch::ProcessorFeatures::AVX512VL())
//                    return __invoke_simd<_Function_<simd256_avx512vlf<_Type_>>>(__aligned_size, __size & (sizeof(simd256_avx512vlf<_Type_>) -
//                        sizeof(typename simd256_avx512vlf<_Type_>::value_type)), std::move(__simd_args));
//            }
//        }
//#else
//        if constexpr (sizeof(_Type_) <= 2) {
//            if (const auto __aligned_size = __size & (~(sizeof(simd512_avx512bw<_Type_>) - 1)); __aligned_size != 0 && arch::ProcessorFeatures::AVX512BW())
//                return __invoke_simd<_Function_<simd512_avx512bw<_Type_>>>(__aligned_size, __size & (sizeof(simd512_avx512bw<_Type_>) -
//                    sizeof(typename simd512_avx512bw<_Type_>::value_type)), std::move(__simd_args));
//        }
//        else {
//            if (const auto __aligned_size = __size & (~(sizeof(simd512_avx512f<_Type_>) - 1)); __aligned_size != 0 && arch::ProcessorFeatures::AVX512F())
//                return __invoke_simd<_Function_<simd512_avx512f<_Type_>>>(__aligned_size, __size & (sizeof(simd512_avx512f<_Type_>) -
//                    sizeof(typename simd512_avx512f<_Type_>::value_type)), std::move(__simd_args));
//        }
//#endif
//
//        if (const auto __aligned_size = __size & (~(sizeof(simd256_avx2<_Type_>) - 1)); __aligned_size != 0 && arch::ProcessorFeatures::AVX2()) {
//            return __invoke_simd<_Function_<simd256_avx2<_Type_>>>(__aligned_size, __size & (sizeof(simd256_avx2<_Type_>) -
//                sizeof(typename simd256_avx2<_Type_>::value_type)), std::move(__simd_args));
//        }
//
//        else if (const auto __aligned_size = __size & (~(sizeof(simd128_sse2<_Type_>) - 1)); __aligned_size != 0 && arch::ProcessorFeatures::SSE2()) {
//            return __invoke_simd<_Function_<simd128_sse2<_Type_>>>(__aligned_size, __size & (sizeof(simd128_sse2<_Type_>) -
//                sizeof(typename simd128_sse2<_Type_>::value_type)), std::move(__simd_args));
//        }

        const auto __aligned_size_64 = __size & (~(sizeof(simd512_avx512f<_Type_>) - 1));

        if (__aligned_size_64 != 0) {
            return __invoke_simd<_Function_<simd512_avx512bwdq<_Type_>>>(__aligned_size_64, __size & (sizeof(simd512_avx512bwdq<_Type_>) -
                sizeof(typename simd512_avx512bwdq<_Type_>::value_type)), std::move(__simd_args));
        }
        
        return std::apply(type_traits::__pass_function(__fallback), std::move(__fallback_args));
    }

    template <
        class       _Type_,
        class       _SizeType_,
        class       _FallbackFunction_,
        class ...   _Args_>
    simd_stl_always_inline static auto __apply(
        _SizeType_              __size,
        _FallbackFunction_&&    __fallback,
        _Args_...               __args) noexcept
    {

        //if (__size >= __zmm_threshold<_Type_>) {
        //    if constexpr (sizeof(_Type_) <= 2) {
        //        if (const auto __aligned_size = __size & (~(sizeof(simd512_avx512bw<_Type_>) - 1)); __aligned_size != 0 && arch::ProcessorFeatures::AVX512BW())
        //            return _Function_<simd512_avx512bw<_Type_>>()(__aligned_size, __size & (sizeof(simd512_avx512bw<_Type_>) -
        //                sizeof(typename simd512_avx512bw<_Type_>::value_type)), std::forward<_Args_>(__args)...);
        //    }
        //    else {
        //        if (const auto __aligned_size = __size & (~(sizeof(simd512_avx512f<_Type_>) - 1)); __aligned_size != 0 && arch::ProcessorFeatures::AVX512F())
        //            return _Function_<simd512_avx512f<_Type_>>()(__aligned_size, __size & (sizeof(simd512_avx512f<_Type_>) -
        //                sizeof(typename simd512_avx512f<_Type_>::value_type)), std::forward<_Args_>(__args)...);
        //    }
        //}
        //else {
        //    if constexpr (sizeof(_Type_) <= 2) {
        //        if (const auto __aligned_size = __size & (~(sizeof(simd256_avx512vlbw<_Type_>) - 1)); __aligned_size != 0 && arch::ProcessorFeatures::AVX512BW() && arch::ProcessorFeatures::AVX512VL())
        //            return _Function_<simd256_avx512vlbw<_Type_>>()(__aligned_size, __size & (sizeof(simd256_avx512vlbw<_Type_>) -
        //                sizeof(typename simd256_avx512vlbw<_Type_>::value_type)), std::forward<_Args_>(__args)...);
        //    }
        //    else {
        //        if (const auto __aligned_size = __size & (~(sizeof(simd256_avx512vlf<_Type_>) - 1)); __aligned_size != 0 && arch::ProcessorFeatures::AVX512F() && arch::ProcessorFeatures::AVX512VL())
        //            return _Function_<simd256_avx512vlf<_Type_>>()(__aligned_size, __size & (sizeof(simd256_avx512vlf<_Type_>) -
        //                sizeof(typename simd256_avx512vlf<_Type_>::value_type)), std::forward<_Args_>(__args)...);
        //    }
        //}

        const auto __aligned_size_64 = __size & (~(sizeof(simd512_avx512f<_Type_>) - 1));
        
        if (__aligned_size_64 != 0) {
           // if (arch::ProcessorFeatures::AVX512BW()) {
                //if (arch::ProcessorFeatures::AVX512DQ()) {
                    return _Function_<simd512_avx512bwdq<_Type_>>()(__aligned_size_64, __size & (sizeof(simd512_avx512bwdq<_Type_>) -
                        sizeof(typename simd512_avx512bwdq<_Type_>::value_type)), std::forward<_Args_>(__args)...);
               // }

               // return _Function_<simd512_avx512bw<_Type_>>()(__aligned_size_64, __size & (sizeof(simd512_avx512bw<_Type_>) -
                 //   sizeof(typename simd512_avx512bw<_Type_>::value_type)), std::forward<_Args_>(__args)...);
           // }


           // if (arch::ProcessorFeatures::AVX512F())
             //   return _Function_<simd512_avx512f<_Type_>>()(__aligned_size_64, __size & (sizeof(simd512_avx512f<_Type_>) -
             //       sizeof(typename simd512_avx512f<_Type_>::value_type)), std::forward<_Args_>(__args)...);
        }

       /* if (const auto __aligned_size = __size & (~(sizeof(simd256_avx2<_Type_>) - 1)); __aligned_size != 0 && arch::ProcessorFeatures::AVX2()) {
            return _Function_<simd256_avx2<_Type_>>()(__aligned_size, __size & (sizeof(simd256_avx2<_Type_>) -
                sizeof(typename simd256_avx2<_Type_>::value_type)), std::forward<_Args_>(__args)...);
        }

        else if (const auto __aligned_size = __size & (~(sizeof(simd128_sse2<_Type_>) - 1)); __aligned_size != 0 && arch::ProcessorFeatures::SSE2()) {
            return _Function_<simd128_sse2<_Type_>>()(__aligned_size, __size & (sizeof(simd128_sse2<_Type_>) - 
                sizeof(typename simd128_sse2<_Type_>::value_type)), std::forward<_Args_>(__args)...);
        }*/

        return type_traits::invoke(type_traits::__pass_function(__fallback), std::forward<_Args_>(__args)...);
    }
};

__SIMD_STL_DATAPAR_NAMESPACE_END
