#include <simd_stl/numeric/BasicSimd.h>
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

template <typename T, simd_stl::arch::CpuFeature Arch>
void testMethods() {
    using Simd = simd_stl::numeric::basic_simd<Arch, T>;
    constexpr size_t N = Simd::size();

    // --- Конструкторы ---
    {
        Simd v1;
        Simd v2(5);

        for (int i = 0; i < v2.size(); ++i) Assert(v2.extract<T>(i) == 5);

        alignas(16) T arr[N];
        for (int i = 0; i < N; ++i)
            arr[i] = i + 1;

        Simd v3 = Simd::loadUnaligned(arr);
        for (int i = 0; i < v2.size(); ++i)
            Assert(v3.extract<T>(i) == arr[i]);


        Simd v4(v3.unwrap());
        for (int i = 0; i < v2.size(); ++i) Assert(v4.extract<T>(i) == arr[i]);

        Simd v5(v3); // copy ctor
        for (int i = 0; i < v2.size(); ++i) Assert(v5.extract<T>(i) == arr[i]);
    }

    // --- fill / extract / insert ---
    {
        Simd v;
        v.fill<T>(42);
        for (int i = 0; i < v.size(); ++i) Assert(v.extract<T>(i) == 42);

        v.insert<T>(0, 99);
        Assert(v.extract<T>(0) == 99);
    }

    // --- extractWrapped ---
    {
        Simd v(7);
        auto ref = v.extractWrapped<T>(0);
        ref = 123;
        Assert(v.extract<T>(0) == 123);
    }

    // --- expand ---
    {
        /* Simd v(0);
         typename Simd::mask_type mask;
         v.expand(mask, 77);
         Assert(v.extract(0) == 77);*/
    }

    // --- convert---
    {
        using simd_stl::numeric::simd_cast;

        Simd v(5);
        auto v8 = v.convert<simd_stl::numeric::basic_simd<simd_stl::arch::CpuFeature::SSE2, simd_stl::int8>>();
        auto vDouble = simd_cast<simd_stl::numeric::basic_simd<simd_stl::arch::CpuFeature::SSE2, double>>(v8);
        auto vint = simd_cast<simd_stl::numeric::basic_simd<simd_stl::arch::CpuFeature::SSE2, simd_stl::int32>>(v);
    }

    // --- simd_cast ---
    {
        using simd_stl::numeric::simd_cast;
        Simd v(11);
        auto vOther = simd_cast<simd_stl::numeric::basic_simd<simd_stl::arch::CpuFeature::SSE2, float>>(v);
        auto vOther2 = simd_cast<simd_stl::arch::CpuFeature::SSE2, float>(v);
        auto vOther3 = simd_cast<simd_stl::arch::CpuFeature::SSE2>(v);
        auto vOther4 = simd_cast<__m128i>(v);
        auto vOther5 = simd_cast<int>(v);

        static_assert(std::is_same_v<decltype(vOther), decltype(vOther2)>);
        static_assert(std::is_same_v<decltype(vOther2), simd_stl::numeric::basic_simd<simd_stl::arch::CpuFeature::SSE2, float, simd_stl::numeric::xmm128>>);
        static_assert(std::is_same_v<decltype(vOther3), simd_stl::numeric::basic_simd<simd_stl::arch::CpuFeature::SSE2, typename Simd::value_type, simd_stl::numeric::xmm128>>);
        static_assert(std::is_same_v<decltype(vOther4), __m128i>);
        static_assert(std::is_same_v<decltype(vOther5), simd_stl::numeric::basic_simd<Simd::_Generation, int, typename Simd::policy_type>>);
    }

    // --- load/store aligned/unaligned ---
    {
        alignas(16) simd_stl::int32 arr[4] = { 10,20,30,40 };
        Simd v = Simd::loadAligned(arr);
        simd_stl::int32 out[4] = {};
        v.storeAligned(out);
        for (int i = 0; i < 4; ++i) Assert(out[i] == arr[i]);

        Simd v2 = Simd::loadUnaligned(arr);
        simd_stl::int32 out2[4] = {};
        v2.storeUnaligned(out2);
        for (int i = 0; i < 4; ++i) Assert(out2[i] == arr[i]);
    }

    // --- unwrap ---
    {
        Simd v(99);
        auto raw = v.unwrap();
        (void)raw; // smoke‑check
    }

    // --- maskLoad/maskStore aligned/unaligned ---
    {
        alignas(64) T src[N];
        alignas(64) T dst[N];

        for (size_t i = 0; i < N; ++i) src[i] = static_cast<T>(i + 1);
        for (size_t i = 0; i < N; ++i) dst[i] = static_cast<T>(100 + i);

        typename Simd::mask_type mask = 0;
        for (size_t i = 0; i < N; ++i)
            if (i % 2 == 0)
                mask |= (typename Simd::mask_type(1) << i);

        Simd loaded_unaligned = Simd::maskLoadUnaligned(src, mask);
        for (size_t i = 0; i < N; ++i) {
            if ((mask >> i) & 1)
                Assert(loaded_unaligned.extract<T>(i) == src[i]);
            else
                Assert(loaded_unaligned.extract<T>(i) == T(0));
        }

        // --- maskLoadAligned ---
        Simd loaded_aligned = Simd::maskLoadAligned(src, mask);
        for (size_t i = 0; i < N; ++i) {
            if ((mask >> i) & 1)
                Assert(loaded_aligned.extract<T>(i) == src[i]);
            else
                Assert(loaded_aligned.extract<T>(i) == T(0));
        }

        // --- maskStoreUnaligned ---
        Simd v(77);
        v.maskStoreUnaligned(dst, mask);
        for (size_t i = 0; i < N; ++i) {
            if ((mask >> i) & 1)
                Assert(dst[i] == T(77));
            else
                Assert(dst[i] == T(100 + i)); // не изменён
        }

        // --- maskStoreAligned ---
        for (size_t i = 0; i < N; ++i) dst[i] = static_cast<T>(200 + i);
        v.maskStoreAligned(dst, mask);
        for (size_t i = 0; i < N; ++i) {
            if (mask & (typename Simd::mask_type(1) << i))
                Assert(dst[i] == T(77));
            else
                Assert(dst[i] == T(200 + i));
        }
    }

    {
        alignas(64) T src[N];
        for (size_t i = 0; i < N; ++i) src[i] = static_cast<T>(i + 1);

        Simd v = Simd::loadUnaligned(src);


        typename Simd::mask_type mask = 0;
        for (size_t i = 0; i < N; i += 2)
            mask |= (typename Simd::mask_type(1) << i); // 0101... 

        // --- compressStoreUnaligned ---
        {
            alignas(64) T dst[N] = {};
            v.compressStoreUnaligned(dst, mask);

            alignas(64) T expected[N];
            mask_compress_any<Simd>(src, src, expected, mask);

            Assert(std::equal(expected, expected + N, dst));
        }

        // --- compressStoreAligned ---
        {
            alignas(64) T dst[N] = {};
            v.compressStoreAligned(dst, mask);

            alignas(64) T expected[N];
            mask_compress_any<Simd>(src, src, expected, mask);

            Assert(std::equal(expected, expected + N, dst));
        }
    }

    {
        std::vector<T> va(N), vb(N), vc(N);
        for (size_t i = 0; i < N; ++i) {
            va[i] = static_cast<T>(i + 1);
            vb[i] = static_cast<T>(i + 1);
            vc[i] = static_cast<T>(i + 2);
        }

        Simd a = Simd::loadUnaligned(va.data());
        Simd b = Simd::loadUnaligned(vb.data());
        Simd c = Simd::loadUnaligned(vc.data());

        // --- isEqual ---
        Assert(a.isEqual(b) && "isEqual failed on equal vectors");
        Assert(!a.isEqual(c) && "isEqual failed on different vectors");

        // --- maskEqual ---
        auto mEq = a.maskEqual(b);
        for (size_t i = 0; i < N; ++i) {
            Assert(mEq[i] == true);
        }

        // --- maskNotEqual ---
        auto mNeq = a.maskNotEqual(c);
        for (size_t i = 0; i < N; ++i) {
            Assert(mNeq[i] == true);
        }

        // --- maskGreater / maskLess ---
        auto mGt = c.maskGreater(a);
        auto mLt = a.maskLess(c);
        for (size_t i = 0; i < N; ++i) {
            Assert(mGt[i] == true);
            Assert(mLt[i] == true);
        }

        auto mGe = a.maskGreaterEqual(b);
        auto mLe = a.maskLessEqual(b);
        for (size_t i = 0; i < N; ++i) {
            Assert(mGe[i] == true);
            Assert(mLe[i] == true);
        }
    }

    {
        alignas(64) T dst[N] = {};
        T srcA[N], srcB[N];

        for (size_t i = 0; i < N; ++i) {
            srcA[i] = static_cast<T>(i + 1);
            srcB[i] = static_cast<T>(100 + i);
        }

        Simd a = Simd::loadUnaligned(srcA);
        Simd b = Simd::loadUnaligned(srcB);

        typename Simd::mask_type m = 0;
        for (size_t i = 0; i < N; i += 2)
            m |= (typename Simd::mask_type(1) << i);

        a.maskBlendStoreUnaligned(dst, m, b);

        for (size_t i = 0; i < N; ++i) {
            if (m & (typename Simd::mask_type(1) << i)) {
                Assert(dst[i] == srcA[i]);
            }
            else {
                Assert(dst[i] == srcB[i]);
            }
        }
    }

    {
        alignas(64) T dst[N] = {};
        T srcA[N], srcB[N];

        for (size_t i = 0; i < N; ++i) {
            srcA[i] = static_cast<T>(10 * (i + 1));
            srcB[i] = static_cast<T>(200 + i);
        }

        Simd a = Simd::loadUnaligned(srcA);
        Simd b = Simd::loadUnaligned(srcB);

        typename Simd::mask_type m = 0;

        for (size_t i = 0; i < N / 2; ++i) {
            m |= (typename Simd::mask_type(1) << i);
        }

        a.maskBlendStoreAligned(dst, m, b);

        for (size_t i = 0; i < N; ++i) {
            if (i < N / 2) {
                Assert(dst[i] == srcA[i]);
            }
            else {
                Assert(dst[i] == srcB[i]);
            }
        }
    }
    
    // Reduce by sum

    {
        simd_stl::numeric::_Reduce_type<T> reduced = 0;

        alignas(64) T array[N] = {};
        for (auto i = 0; i < N; ++i) {
            array[i] = (unsigned char)(i * 0x7fdbu);
        }

        for (auto i = 0; i < N; ++i) {
            reduced += (unsigned char)(array[i]);
        }

        Simd a = Simd::loadUnaligned(array);
        auto simdReduced = a.reduce();

        Assert(simdReduced == reduced);
    }


    {
        alignas(64) T arrA[N], arrB[N];
        for (size_t i = 0; i < N; ++i) {
            arrA[i] = static_cast<T>(i - 2);   // значения от -2
            arrB[i] = static_cast<T>(N - i);   // обратная последовательность
        }

        Simd a = Simd::loadUnaligned(arrA);
        Simd b = Simd::loadUnaligned(arrB);

        // --- min(vector, vector) ---
        auto minVec = a.min(b);
        for (size_t i = 0; i < N; ++i) {
            Assert(minVec.extract<T>(i) == std::min(arrA[i], arrB[i]));
        }

        // --- max(vector, vector) ---
        auto maxVec = a.max(b);
        for (size_t i = 0; i < N; ++i) {
            Assert(maxVec.extract<T>(i) == std::max(arrA[i], arrB[i]));
        }

        //// --- min() скалярный ---
        //T minScalar = a.min();
        //T expectedMin = *std::min_element(arrA, arrA + N);
        //Assert(minScalar == expectedMin);

        //// --- max() скалярный ---
        //T maxScalar = a.max();
        //T expectedMax = *std::max_element(arrA, arrA + N);
        //Assert(maxScalar == expectedMax);

        // --- abs() ---
        auto absVec = a.abs();
        for (size_t i = 0; i < N; ++i) {
            Assert(absVec.extract<T>(i) == static_cast<T>(simd_stl::math::abs(arrA[i])));
        }
    }
}

template <simd_stl::arch::CpuFeature _Generation_>
void testMethods() {
    testMethods<simd_stl::int8, _Generation_>();
    testMethods<simd_stl::uint8, _Generation_>();

    testMethods<simd_stl::int16, _Generation_>();
    testMethods<simd_stl::uint16, _Generation_>();

    testMethods<simd_stl::int32, _Generation_>();
    testMethods<simd_stl::uint32, _Generation_>();

    testMethods<simd_stl::int64, _Generation_>();
    testMethods<simd_stl::uint64, _Generation_>();

    testMethods<float, _Generation_>();
    testMethods<double, _Generation_>();
}

int main() {
    testMethods<simd_stl::arch::CpuFeature::SSE2>();
    testMethods<simd_stl::arch::CpuFeature::SSE3>();
    testMethods<simd_stl::arch::CpuFeature::SSSE3>();
    testMethods<simd_stl::arch::CpuFeature::SSE41>();
    testMethods<simd_stl::arch::CpuFeature::SSE42>();
    testMethods<simd_stl::arch::CpuFeature::AVX2>();
    testMethods<simd_stl::arch::CpuFeature::AVX512F>();
    testMethods<simd_stl::arch::CpuFeature::AVX512BW>();
    testMethods<simd_stl::arch::CpuFeature::AVX512DQ>();
    testMethods<simd_stl::arch::CpuFeature::AVX512VL>();

    return 0;
}
