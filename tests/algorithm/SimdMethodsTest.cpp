#include <simd_stl/datapar/SimdDataparAlgorithms.h>
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

template <typename T, simd_stl::arch::ISA Arch, uint32 _Width_>
void testMethods() {
    using Simd = simd_stl::datapar::simd<Arch, T, _Width_>;
    constexpr size_t N = Simd::size();

     {
        Simd v1;
        Simd v2(5);

        for (int i = 0; i < v2.size(); ++i) simd_stl_assert(v2.extract(i) == 5);

        alignas(16) T arr[N];
        for (int i = 0; i < N; ++i)
            arr[i] = i + 1;

        Simd v3 = simd_stl::datapar::load<Simd>(arr);
        for (int i = 0; i < v2.size(); ++i)
            simd_stl_assert(v3.extract(i) == arr[i]);


        Simd v4(v3.unwrap());
        for (int i = 0; i < v2.size(); ++i) simd_stl_assert(v4.extract(i) == arr[i]);

        Simd v5(v3);
        for (int i = 0; i < v2.size(); ++i) simd_stl_assert(v5.extract(i) == arr[i]);
    }

    {
        Simd v;
        v.fill(42);
        for (int i = 0; i < v.size(); ++i) simd_stl_assert(v.extract(i) == 42);

        v.insert(0, 99);
        simd_stl_assert(v.extract(0) == 99);
    }

    {
        Simd v(7);
        auto ref = v.extract_wrapped(0);
        ref = 123;
        simd_stl_assert(v.extract(0) == 123);
    }
    
    {
        using simd_stl::datapar::simd_cast;
        Simd v(11);
        auto vOther = simd_cast<simd_stl::datapar::simd128_sse2<float>>(v);
        auto vOther2 = simd_cast<simd_stl::arch::ISA::SSE2, float>(v);
        auto vOther3 = simd_cast<simd_stl::arch::ISA::SSE2>(v);
        auto vOther4 = simd_cast<__m128i>(v);
        auto vOther5 = simd_cast<int>(v);

        static_assert(std::is_same_v<decltype(vOther), decltype(vOther2)>);
        static_assert(std::is_same_v<decltype(vOther2), simd_stl::datapar::simd128_sse2<float>>);
        static_assert(std::is_same_v<decltype(vOther3), simd_stl::datapar::simd<simd_stl::arch::ISA::SSE2, typename Simd::value_type, 128>>);
        static_assert(std::is_same_v<decltype(vOther4), __m128i>);
        static_assert(std::is_same_v<decltype(vOther5), simd_stl::datapar::simd<Simd::__generation, int, typename Simd::policy_type>>);
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

        Simd loaded_unaligned = simd_stl::datapar::maskz_load<Simd>(src, mask);
        for (size_t i = 0; i < N; ++i) {
            if ((mask >> i) & 1)
                simd_stl_assert(loaded_unaligned.extract(i) == src[i]);
            else
                simd_stl_assert(loaded_unaligned.extract(i) == T(0));
        }

        Simd loaded_aligned = simd_stl::datapar::maskz_load<Simd>(src, mask, simd_stl::datapar::aligned_policy{});
        for (size_t i = 0; i < N; ++i) {
            if ((mask >> i) & 1)
                simd_stl_assert(loaded_aligned.extract(i) == src[i]);
            else
                simd_stl_assert(loaded_aligned.extract(i) == T(0));
        }

        Simd v(77);
        simd_stl::datapar::mask_store(dst, v, mask);
        for (size_t i = 0; i < N; ++i) {
            if ((mask >> i) & 1)
                simd_stl_assert(dst[i] == T(77));
            else
                simd_stl_assert(dst[i] == T(100 + i));
        }

        for (size_t i = 0; i < N; ++i) dst[i] = static_cast<T>(200 + i);
        simd_stl::datapar::mask_store(dst, v, mask, simd_stl::datapar::aligned_policy{});
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

        Simd v = simd_stl::datapar::load<Simd>(src);


        typename Simd::mask_type mask = 0;
        for (size_t i = 0; i < N; i += 2)
            mask |= (typename Simd::mask_type(1) << i);

        {
            alignas(64) T dst[N] = {};
            simd_stl::datapar::compress_store(dst, v, mask);

            alignas(64) T expected[N];
            mask_compress_any<Simd>(src, src, expected, mask);

            simd_stl_assert(std::equal(expected, expected + N, dst));
        }

        {
            alignas(64) T dst[N] = {};
            simd_stl::datapar::compress_store(dst, v, mask, simd_stl::datapar::aligned_policy{});

            alignas(64) T expected[N];
            mask_compress_any<Simd>(src, src, expected, mask);

            simd_stl_assert(std::equal(expected, expected + N, dst));
        }

        {
            for (mask = 0; mask != N; ++mask) {
                alignas(64) T dst[N] = {};
                simd_stl::datapar::compress_store(dst, v, mask);
                
                alignas(64) T expected[N];
                mask_compress_any<Simd>(src, src, expected, mask);

                simd_stl_assert(std::equal(expected, expected + N, dst));
            }
        }

        {
            for (mask = 0; mask != N; ++mask) {
                alignas(64) T dst[N] = {};
                simd_stl::datapar::compress_store(dst, v, mask, simd_stl::datapar::aligned_policy{});

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

        Simd a = simd_stl::datapar::load<Simd>(va.data());
        Simd b = simd_stl::datapar::load<Simd>(vb.data());
        Simd c = simd_stl::datapar::load<Simd>(vc.data());

        simd_stl_assert(a == b);
        simd_stl_assert(a != c);

        auto mEq = simd_stl::datapar::as_mask | (a == b);
        for (size_t i = 0; i < N; ++i) {
            simd_stl_assert(mEq[i] == true);
        }

        auto mNeq = (a != c) | simd_stl::datapar::as_mask;
        for (size_t i = 0; i < N; ++i) {
            simd_stl_assert(mNeq[i] == true);
        }

        auto mGt = (c > a) | simd_stl::datapar::as_mask;
        auto mLt = (a < c) | simd_stl::datapar::as_mask;
        for (size_t i = 0; i < N; ++i) {
            simd_stl_assert(mGt[i] == true);
            simd_stl_assert(mLt[i] == true);
        }

        auto mGe = (a >= b) | simd_stl::datapar::as_mask;
        auto mLe = (a <= b) | simd_stl::datapar::as_mask;
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

        Simd v0 = simd_stl::datapar::load<Simd>(arr0);
        Simd vmax = simd_stl::datapar::load<Simd>(arrMax);

        simd_stl_assert(v0 != vmax);

        auto mEq = (v0 == v0) | simd_stl::datapar::as_mask;
        auto mNeq = (v0 != vmax) | simd_stl::datapar::as_mask;
        for (size_t i = 0; i < N; ++i) {
            simd_stl_assert(mEq[i] == true);
            simd_stl_assert(mNeq[i] == true);
        }

        auto mLt = (v0 < vmax) | simd_stl::datapar::as_mask;
        auto mGt = (vmax > v0) | simd_stl::datapar::as_mask;
        auto mLe = (v0 <= v0) | simd_stl::datapar::as_mask;
        auto mGe = (vmax >= vmax) | simd_stl::datapar::as_mask;

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
            Simd vA = simd_stl::datapar::load<Simd>(arrA);
            Simd vB = simd_stl::datapar::load<Simd>(arrB);

            auto mEq = (vA == vA) | simd_stl::datapar::as_mask;
            auto mNeq = (vA != vB) | simd_stl::datapar::as_mask;
            auto mLt = (vA < vB) | simd_stl::datapar::as_mask;
            auto mGt = (vB > vA) | simd_stl::datapar::as_mask;
            auto mLe = (vA <= vA) | simd_stl::datapar::as_mask;
            auto mGe = (vB >= vB) | simd_stl::datapar::as_mask;

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
        simd_stl::datapar::__reduce_type<T> reduced = 0;

        alignas(64) T array[N] = {};
        for (auto i = 0; i < N; ++i) {
            array[i] = (unsigned char)(i * 0x7fdbu);
        }

        for (auto i = 0; i < N; ++i) {
            reduced += (unsigned char)(array[i]);
        }

        auto simdReduced = simd_stl::datapar::reduce(simd_stl::datapar::load<Simd>(array));

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

        Simd a = simd_stl::datapar::load<Simd>(arrA);
        Simd b = simd_stl::datapar::load<Simd>(arrB);

        auto minVec = simd_stl::datapar::vertical_min(a, b);
        for (size_t i = 0; i < N; ++i) {
            simd_stl_assert(minVec.extract(i) == std::min(arrA[i], arrB[i]));
        }

        auto maxVec = simd_stl::datapar::vertical_max(a, b);
        for (size_t i = 0; i < N; ++i) {
            simd_stl_assert(maxVec.extract(i) == std::max(arrA[i], arrB[i]));
        }

        T minScalar = simd_stl::datapar::horizontal_min(a);
        T expectedMin = *std::min_element(arrA, arrA + N);
        simd_stl_assert(minScalar == expectedMin);

        T maxScalar = simd_stl::datapar::horizontal_max(a);
        T expectedMax = *std::max_element(arrA, arrA + N);
        simd_stl_assert(maxScalar == expectedMax);

        auto absVec = simd_stl::datapar::abs(a);
        for (size_t i = 0; i < N; ++i) {
            simd_stl_assert(absVec.extract(i) == static_cast<T>(simd_stl::math::abs(arrA[i])));
        }
    }

    {
        alignas(64) T arr[N];
        alignas(64) T reversed[N];

        for (size_t i = 0; i < N; ++i) {
            arr[i] = static_cast<T>(i);
        }


        Simd simdVec = simd_stl::datapar::reverse(simd_stl::datapar::load<Simd>(arr));
        simd_stl::datapar::store(reversed, simdVec);

        for (size_t i = 0; i < N; ++i) {
            simd_stl_assert(reversed[i] == arr[N - 1 - i]);
        }
    }

    {
        Simd v1(10);
        Simd v2(10);

        auto mask = (v1 == v2) | simd_stl::datapar::as_mask;
        auto index_mask = (v1 == v2) | simd_stl::datapar::as_index_mask;

        simd_stl_assert(index_mask.any_of());
        simd_stl_assert(index_mask.all_of());
        simd_stl_assert(!index_mask.none_of());
        simd_stl_assert(static_cast<bool>(index_mask) == true);

        const auto a = index_mask.count_set();
        const auto b = mask.count_set();

        simd_stl_assert(index_mask.count_set() == mask.count_set());
        simd_stl_assert(mask.count_set() == simd_stl::datapar::reduce(mask));
        simd_stl_assert(index_mask.count_set() == simd_stl::datapar::reduce(index_mask));
        
        simd_stl_assert(index_mask.count_trailing_zero_bits() == 0);
        simd_stl_assert(index_mask.count_trailing_zero_bits() == mask.count_trailing_zero_bits());
        simd_stl_assert(index_mask.count_leading_zero_bits() == mask.count_leading_zero_bits());
    }

    {
        Simd v1(10);
        Simd v2(20);

        auto index_mask = simd_stl::datapar::as_index_mask | (v1 == v2);

        simd_stl_assert(!index_mask.any_of());
        simd_stl_assert(!index_mask.all_of());
        simd_stl_assert(index_mask.none_of());
        simd_stl_assert(index_mask.count_set() == 0 && simd_stl::datapar::reduce(index_mask) == 0);
    }

    for (size_t i = 0; i < N; ++i) {
        Simd v1(0);
        Simd v2(0);

        v1.insert(i, 42);
        v2.insert(i, 42);

        auto index_mask = (v1 == v2) | simd_stl::datapar::as_index_mask;
        auto mask = (v1 == v2) | simd_stl::datapar::as_mask;

        simd_stl_assert(index_mask.any_of());
        simd_stl_assert(index_mask.all_of());

        simd_stl_assert(index_mask.count_set() == mask.count_set());
        simd_stl_assert(mask.count_set() == simd_stl::datapar::reduce(mask));
        simd_stl_assert(index_mask.count_set() == simd_stl::datapar::reduce(index_mask));

        simd_stl_assert(index_mask.count_trailing_zero_bits() == mask.count_trailing_zero_bits());
        simd_stl_assert(index_mask.count_leading_zero_bits() == mask.count_leading_zero_bits());
    }

   {
        Simd v1(0);
        Simd v2(0);

        v1.insert(0, 1); v2.insert(0, 1);
        v1.insert(1, 1); v2.insert(1, 1);
        v1.insert(N - 1, 1); v2.insert(N - 1, 1);

        auto index_mask = (v1 == v2) | simd_stl::datapar::as_index_mask;
        auto mask = (v1 == v2) | simd_stl::datapar::as_mask;

        simd_stl_assert(index_mask.count_set() == mask.count_set());
    }

    {
        if constexpr (sizeof(T) > 1) {
            T val1 = 0;
            T val2 = 0;

            unsigned char* p = reinterpret_cast<unsigned char*>(&val2);
            p[0] = 0xFF;

            Simd v1(val1);
            Simd v2(val2);

            auto mask = (v1 == v2) | simd_stl::datapar::as_index_mask;

            simd_stl_assert(mask.none_of());
        }
    }
}

template <simd_stl::arch::ISA _Generation_, simd_stl::uint32 _Width_>
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
    testMethods<simd_stl::arch::ISA::SSE2, 128>();
    testMethods<simd_stl::arch::ISA::SSE3, 128>();
    testMethods<simd_stl::arch::ISA::SSSE3, 128>();
    testMethods<simd_stl::arch::ISA::SSE41, 128>();
    testMethods<simd_stl::arch::ISA::SSE42, 128>();

    testMethods<simd_stl::arch::ISA::AVX2, 128>();
    testMethods<simd_stl::arch::ISA::AVX2, 256>();

   /* testMethods<simd_stl::arch::ISA::AVX512F, 512>();
    testMethods<simd_stl::arch::ISA::AVX512BW, 512>();
    testMethods<simd_stl::arch::ISA::AVX512DQ, 512>();
    testMethods<simd_stl::arch::ISA::AVX512BWDQ, 512>();
    testMethods<simd_stl::arch::ISA::AVX512VBMI, 512>();
    testMethods<simd_stl::arch::ISA::AVX512VBMI2, 512>();
    testMethods<simd_stl::arch::ISA::AVX512VBMIDQ, 512>();
    testMethods<simd_stl::arch::ISA::AVX512VBMI2DQ, 512>();

    testMethods<simd_stl::arch::ISA::AVX512VLF, 128>();
    testMethods<simd_stl::arch::ISA::AVX512VLBW, 128>();
    testMethods<simd_stl::arch::ISA::AVX512VLBWDQ, 128>();
    testMethods<simd_stl::arch::ISA::AVX512VLDQ, 128>();
    
    testMethods<simd_stl::arch::ISA::AVX512VLF, 256>();
    testMethods<simd_stl::arch::ISA::AVX512VLBW, 256>();
    testMethods<simd_stl::arch::ISA::AVX512VLBWDQ, 256>();
    testMethods<simd_stl::arch::ISA::AVX512VLDQ, 256>();

    testMethods<simd_stl::arch::ISA::AVX512VBMIVL, 128>();
    testMethods<simd_stl::arch::ISA::AVX512VBMI2VL, 128>();
    testMethods<simd_stl::arch::ISA::AVX512VBMIVLDQ, 128>();
    testMethods<simd_stl::arch::ISA::AVX512VBMI2VLDQ, 128>();

    testMethods<simd_stl::arch::ISA::AVX512VBMIVL, 256>();
    testMethods<simd_stl::arch::ISA::AVX512VBMI2VL, 256>();
    testMethods<simd_stl::arch::ISA::AVX512VBMIVLDQ, 256>();
    testMethods<simd_stl::arch::ISA::AVX512VBMI2VLDQ, 256>();*/


    return 0;
}
