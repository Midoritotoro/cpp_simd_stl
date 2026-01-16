#include <simd_stl/numeric/Simd.h>

#include <simd_stl/math/Math.h>

#include <string>


template <typename _Simd_>
void mask_compress_any(
    const typename _Simd_::value_type* a,
    const typename _Simd_::value_type* src,
    typename _Simd_::value_type* dst,
    typename _Simd_::mask_type mask)
{
    constexpr auto N = _Simd_::template size();

    int m = 0;

    for (int j = 0; j < N; ++j)
        if ((~(mask >> j)) & 1)
            dst[m++] = a[j];

    for (int i = m; i < N; ++i)
        dst[i] = src[i];
}

template <typename T, simd_stl::arch::CpuFeature Arch, typename _RegisterPolicy_>
void testMethods() {
    using Simd = simd_stl::numeric::simd<Arch, T, _RegisterPolicy_>;
    constexpr size_t N = Simd::size();

    {
        Simd v1;
        Simd v2(5);

        for (int i = 0; i < v2.size(); ++i) simd_stl_assert(v2.extract<T>(i) == 5);

        alignas(16) T arr[N];
        for (int i = 0; i < N; ++i)
            arr[i] = i + 1;

        Simd v3 = Simd::load(arr);
        for (int i = 0; i < v2.size(); ++i)
            simd_stl_assert(v3.extract<T>(i) == arr[i]);


        Simd v4(v3.unwrap());
        for (int i = 0; i < v2.size(); ++i) simd_stl_assert(v4.extract<T>(i) == arr[i]);

        Simd v5(v3);
        for (int i = 0; i < v2.size(); ++i) simd_stl_assert(v5.extract<T>(i) == arr[i]);
    }

    {
        Simd v;
        v.fill<T>(42);
        for (int i = 0; i < v.size(); ++i) simd_stl_assert(v.extract<T>(i) == 42);

        v.insert<T>(0, 99);
        simd_stl_assert(v.extract<T>(0) == 99);
    }

    {
        Simd v(7);
        auto ref = v.extract_wrapped<T>(0);
        ref = 123;
        simd_stl_assert(v.extract<T>(0) == 123);
    }

    {
        using simd_stl::numeric::simd_cast;
        Simd v(11);
        auto vOther = simd_cast<simd_stl::numeric::simd128_sse2<float>>(v);
        auto vOther2 = simd_cast<simd_stl::arch::CpuFeature::SSE2, float>(v);
        auto vOther3 = simd_cast<simd_stl::arch::CpuFeature::SSE2>(v);
        auto vOther4 = simd_cast<__m128i>(v);
        auto vOther5 = simd_cast<int>(v);

        static_assert(std::is_same_v<decltype(vOther), decltype(vOther2)>);
        static_assert(std::is_same_v<decltype(vOther2), simd_stl::numeric::simd128_sse2<float>>);
        static_assert(std::is_same_v<decltype(vOther3), simd_stl::numeric::simd<simd_stl::arch::CpuFeature::SSE2, typename Simd::value_type, simd_stl::numeric::xmm128>>);
        static_assert(std::is_same_v<decltype(vOther4), __m128i>);
        static_assert(std::is_same_v<decltype(vOther5), simd_stl::numeric::simd<Simd::__generation, int, typename Simd::policy_type>>);
    }

    {
        alignas(16) simd_stl::int32 arr[4] = { 10,20,30,40 };
        Simd v = Simd::load(arr, simd_stl::numeric::aligned_policy{});
        simd_stl::int32 out[4] = {};
        v.store(out, simd_stl::numeric::aligned_policy{});
        for (int i = 0; i < 4; ++i) simd_stl_assert(out[i] == arr[i]);

        Simd v2 = Simd::load(arr);
        simd_stl::int32 out2[4] = {};
        v2.store(out2);
        for (int i = 0; i < 4; ++i) simd_stl_assert(out2[i] == arr[i]);
    }

    {
        Simd v(42);
        auto raw = v.unwrap();
        (void)raw;
    }

    {
        alignas(64) T src[N];
        alignas(64) T dst[N];

        for (size_t i = 0; i < N; ++i) src[i] = static_cast<T>(i + 1);
        for (size_t i = 0; i < N; ++i) dst[i] = static_cast<T>(100 + i);

        typename Simd::mask_type mask = 0;
        for (size_t i = 0; i < N; ++i)
            if (i % 2 == 0)
                mask |= (typename Simd::mask_type(1) << i);

        Simd loaded_unaligned = Simd::mask_load(src, mask);
        for (size_t i = 0; i < N; ++i) {
            if ((mask >> i) & 1)
                simd_stl_assert(loaded_unaligned.extract<T>(i) == src[i]);
            else
                simd_stl_assert(loaded_unaligned.extract<T>(i) == T(0));
        }

        Simd loaded_aligned = Simd::mask_load(src, mask, simd_stl::numeric::aligned_policy{});
        for (size_t i = 0; i < N; ++i) {
            if ((mask >> i) & 1)
                simd_stl_assert(loaded_aligned.extract<T>(i) == src[i]);
            else
                simd_stl_assert(loaded_aligned.extract<T>(i) == T(0));
        }

        Simd v(77);
        v.mask_store(dst, mask);
        for (size_t i = 0; i < N; ++i) {
            if ((mask >> i) & 1)
                simd_stl_assert(dst[i] == T(77));
            else
                simd_stl_assert(dst[i] == T(100 + i));
        }

        for (size_t i = 0; i < N; ++i) dst[i] = static_cast<T>(200 + i);
        v.mask_store(dst, mask, simd_stl::numeric::aligned_policy{});
        for (size_t i = 0; i < N; ++i) {
            if (mask & (typename Simd::mask_type(1) << i))
                simd_stl_assert(dst[i] == T(77));
            else
                simd_stl_assert(dst[i] == T(200 + i));
        }
    }

    {
        alignas(64) T src[N];
        for (size_t i = 0; i < N; ++i) src[i] = static_cast<T>(i + 1);

        Simd v = Simd::load(src);


        typename Simd::mask_type mask = 0;
        for (size_t i = 0; i < N; i += 2)
            mask |= (typename Simd::mask_type(1) << i);

        {
            alignas(64) T dst[N] = {};
            v.compress_store(dst, mask);

            alignas(64) T expected[N];
            mask_compress_any<Simd>(src, src, expected, mask);

            simd_stl_assert(std::equal(expected, expected + N, dst));
        }

        {
            alignas(64) T dst[N] = {};
            v.compress_store(dst, mask, simd_stl::numeric::aligned_policy{});

            alignas(64) T expected[N];
            mask_compress_any<Simd>(src, src, expected, mask);

            simd_stl_assert(std::equal(expected, expected + N, dst));
        }

        {
            for (mask = 0; mask < N; ++mask) {
                alignas(64) T dst[N] = {};
                v.compress_store(dst, mask);
                
                alignas(64) T expected[N];
                mask_compress_any<Simd>(src, src, expected, mask);

                simd_stl_assert(std::equal(expected, expected + N, dst));
            }
        }

        {
            for (mask = 0; mask < N; ++mask) {
                alignas(64) T dst[N] = {};
                v.compress_store(dst, mask, simd_stl::numeric::aligned_policy{});

                alignas(64) T expected[N];
                mask_compress_any<Simd>(src, src, expected, mask);

                simd_stl_assert(std::equal(expected, expected + N, dst));
            }
        }
    }

    {
        std::vector<T> va(N), vb(N), vc(N);
        for (size_t i = 0; i < N; ++i) {
            va[i] = static_cast<T>(i + 1);
            vb[i] = static_cast<T>(i + 1);
            vc[i] = static_cast<T>(i + 2);
        }

        Simd a = Simd::load(va.data());
        Simd b = Simd::load(vb.data());
        Simd c = Simd::load(vc.data());

        simd_stl_assert(a == b);
        simd_stl_assert(a != c);

        auto mEq = a.mask_compare<simd_stl::numeric::simd_comparison::equal>(b);
        for (size_t i = 0; i < N; ++i) {
            simd_stl_assert(mEq[i] == true);
        }

        auto mNeq = a.mask_compare<simd_stl::numeric::simd_comparison::not_equal>(c);
        for (size_t i = 0; i < N; ++i) {
            simd_stl_assert(mNeq[i] == true);
        }

        auto mGt = c.mask_compare<simd_stl::numeric::simd_comparison::greater>(a);
        auto mLt = a.mask_compare<simd_stl::numeric::simd_comparison::less>(c);
        for (size_t i = 0; i < N; ++i) {
            simd_stl_assert(mGt[i] == true);
            simd_stl_assert(mLt[i] == true);
        }

        auto mGe = a.mask_compare<simd_stl::numeric::simd_comparison::greater_equal>(b);
        auto mLe = a.mask_compare<simd_stl::numeric::simd_comparison::less_equal>(b);
        for (size_t i = 0; i < N; ++i) {
            simd_stl_assert(mGe[i] == true);
            simd_stl_assert(mLe[i] == true);
        }
    }

    {
        alignas(64) T arr0[N], arrMax[N];
        for (size_t i = 0; i < N; ++i) {
            arr0[i] = 0;
            arrMax[i] = std::numeric_limits<T>::max();
        }

        Simd v0 = Simd::load(arr0);
        Simd vmax = Simd::load(arrMax);

        simd_stl_assert(v0 != vmax);

        auto mEq = v0.mask_compare<simd_stl::numeric::simd_comparison::equal>(v0);
        auto mNeq = v0.mask_compare<simd_stl::numeric::simd_comparison::not_equal>(vmax);
        for (size_t i = 0; i < N; ++i) {
            simd_stl_assert(mEq[i] == true);
            simd_stl_assert(mNeq[i] == true);
        }

        auto mLt = v0.mask_compare<simd_stl::numeric::simd_comparison::less>(vmax);
        auto mGt = vmax.mask_compare<simd_stl::numeric::simd_comparison::greater>(v0);
        auto mLe = v0.mask_compare<simd_stl::numeric::simd_comparison::less_equal>(v0);
        auto mGe = vmax.mask_compare<simd_stl::numeric::simd_comparison::greater_equal>(vmax);

        for (size_t i = 0; i < N; ++i) {
            simd_stl_assert(mLt[i] == true);
            simd_stl_assert(mGt[i] == true);
            simd_stl_assert(mLe[i] == true);
            simd_stl_assert(mGe[i] == true);
        }
    }

    {
        for (long long step = 1; step < (1LL << (std::numeric_limits<T>::digits - 2)); step <<= 1) {
            alignas(64) T arrA[N], arrB[N];
            for (size_t i = 0; i < N; ++i) {
                arrA[i] = step;
                arrB[i] = step + 1;
            }
            Simd vA = Simd::load(arrA);
            Simd vB = Simd::load(arrB);

            auto mEq = vA.mask_compare<simd_stl::numeric::simd_comparison::equal>(vA);
            auto mNeq = vA.mask_compare<simd_stl::numeric::simd_comparison::not_equal>(vB);
            auto mLt = vA.mask_compare<simd_stl::numeric::simd_comparison::less>(vB);
            auto mGt = vB.mask_compare<simd_stl::numeric::simd_comparison::greater>(vA);
            auto mLe = vA.mask_compare<simd_stl::numeric::simd_comparison::less_equal>(vA);
            auto mGe = vB.mask_compare<simd_stl::numeric::simd_comparison::greater_equal>(vB);

            for (size_t i = 0; i < N; ++i) {
                simd_stl_assert(mEq[i] == true);
                simd_stl_assert(mNeq[i] == true);
                simd_stl_assert(mLt[i] == true);
                simd_stl_assert(mGt[i] == true);
                simd_stl_assert(mLe[i] == true);
                simd_stl_assert(mGe[i] == true);
            }
        }
    }
  
    {
        simd_stl::numeric::__reduce_type<T> reduced = 0;

        alignas(64) T array[N] = {};
        for (auto i = 0; i < N; ++i) {
            array[i] = (unsigned char)(i * 0x7fdbu);
        }

        for (auto i = 0; i < N; ++i) {
            reduced += (unsigned char)(array[i]);
        }

        Simd a = Simd::load(array);
        auto simdReduced = a.reduce_add();

        simd_stl_assert(simdReduced == reduced);
    }


    {
        alignas(64) T arrA[N], arrB[N];

        for (size_t i = 0; i < N; ++i) {
            if (i % 4 == 0)
                arrA[i] = static_cast<T>(100);
            else if (i % 5 == 0)
                arrA[i] = static_cast<T>(-50);
            else
                arrA[i] = static_cast<T>(i - 2);

            if (i % 3 == 0)
                arrB[i] = static_cast<T>(200);
            else if (i % 7 == 0)
                arrB[i] = static_cast<T>(-100);
            else
                arrB[i] = static_cast<T>(N - i);
        }

        Simd a = Simd::load(arrA);
        Simd b = Simd::load(arrB);

        auto minVec = a.vertical_min(b);
        for (size_t i = 0; i < N; ++i) {
            simd_stl_assert(minVec.extract<T>(i) == std::min(arrA[i], arrB[i]));
        }

        auto maxVec = a.vertical_max(b);
        for (size_t i = 0; i < N; ++i) {
            simd_stl_assert(maxVec.extract<T>(i) == std::max(arrA[i], arrB[i]));
        }

        T minScalar = a.horizontal_min();
        T expectedMin = *std::min_element(arrA, arrA + N);
        simd_stl_assert(minScalar == expectedMin);

        T maxScalar = a.horizontal_max();
        T expectedMax = *std::max_element(arrA, arrA + N);
        simd_stl_assert(maxScalar == expectedMax);

        auto absVec = a.abs();
        for (size_t i = 0; i < N; ++i) {
            simd_stl_assert(absVec.extract<T>(i) == static_cast<T>(simd_stl::math::abs(arrA[i])));
        }
    }

    {
        alignas(64) T arr[N];
        alignas(64) T reversed[N];

        for (size_t i = 0; i < N; ++i) {
            arr[i] = static_cast<T>(i);
        }


        Simd simdVec = Simd::load(arr);
        simdVec.reverse();
        simdVec.store(reversed);

        for (size_t i = 0; i < N; ++i) {
            assert(reversed[i] == arr[N - 1 - i]);
        }
    }
}

template <simd_stl::arch::CpuFeature _Generation_, typename _RegisterPolicy_>
void testMethods() {
    testMethods<simd_stl::int8, _Generation_, _RegisterPolicy_>();
    testMethods<simd_stl::uint8, _Generation_, _RegisterPolicy_>();

    testMethods<simd_stl::int16, _Generation_, _RegisterPolicy_>();
    testMethods<simd_stl::uint16, _Generation_, _RegisterPolicy_>();

    testMethods<simd_stl::int32, _Generation_, _RegisterPolicy_>();
    testMethods<simd_stl::uint32, _Generation_, _RegisterPolicy_>();

    testMethods<simd_stl::int64, _Generation_, _RegisterPolicy_>();
    testMethods<simd_stl::uint64, _Generation_, _RegisterPolicy_>();

    testMethods<float, _Generation_, _RegisterPolicy_>();
    testMethods<double, _Generation_, _RegisterPolicy_>();
}

int main() {
    testMethods<simd_stl::arch::CpuFeature::SSE2, simd_stl::numeric::xmm128>();
    testMethods<simd_stl::arch::CpuFeature::SSE3, simd_stl::numeric::xmm128>();
    testMethods<simd_stl::arch::CpuFeature::SSSE3, simd_stl::numeric::xmm128>();
    testMethods<simd_stl::arch::CpuFeature::SSE41, simd_stl::numeric::xmm128>();
    testMethods<simd_stl::arch::CpuFeature::SSE42, simd_stl::numeric::xmm128>();

    testMethods<simd_stl::arch::CpuFeature::AVX2, simd_stl::numeric::ymm256>();
    testMethods<simd_stl::arch::CpuFeature::AVX2, simd_stl::numeric::xmm128>();


    testMethods<simd_stl::arch::CpuFeature::AVX512F, simd_stl::numeric::zmm512>();
    testMethods<simd_stl::arch::CpuFeature::AVX512BW, simd_stl::numeric::zmm512>();
    testMethods<simd_stl::arch::CpuFeature::AVX512DQ, simd_stl::numeric::zmm512>();

    testMethods<simd_stl::arch::CpuFeature::AVX512VLF, simd_stl::numeric::xmm128>();
    testMethods<simd_stl::arch::CpuFeature::AVX512VLBW, simd_stl::numeric::xmm128>();
    testMethods<simd_stl::arch::CpuFeature::AVX512VLBWDQ, simd_stl::numeric::xmm128>();
    testMethods<simd_stl::arch::CpuFeature::AVX512VLDQ, simd_stl::numeric::xmm128>();

    testMethods<simd_stl::arch::CpuFeature::AVX512VLF, simd_stl::numeric::ymm256>();
    testMethods<simd_stl::arch::CpuFeature::AVX512VLBW, simd_stl::numeric::ymm256>();
    testMethods<simd_stl::arch::CpuFeature::AVX512VLBWDQ, simd_stl::numeric::ymm256>();
    testMethods<simd_stl::arch::CpuFeature::AVX512VLDQ, simd_stl::numeric::ymm256>();

    return 0;
}
