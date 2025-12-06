#include <simd_stl/numeric/BasicSimd.h>
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
bool areEqual(simd_stl::numeric::basic_simd<Arch, T>& simd, const std::vector<T>& vec) {
    std::vector<T> simd_data(vec.size());
    simd.storeUnaligned(simd_data.data());
    return std::equal(simd_data.begin(), simd_data.end(), vec.begin());
}

template <typename T, simd_stl::arch::CpuFeature Arch>
void testArithmeticOperations() {
    using Simd = simd_stl::numeric::basic_simd<Arch, T>;

    constexpr size_t num_elements = simd_stl::numeric::basic_simd<Arch, T>::template size(); // Get the number of elements in SIMD vector
    std::vector<T> initial_values(num_elements, 51);
    simd_stl::numeric::basic_simd<Arch, T> simd(initial_values.data());
    std::vector<T> array(num_elements);

    // Division by constexpr number 
    //std::vector<T> expected_constexpr_division(num_elements, 1);
    //simd.divideByConst<2>();
    //simd.storeUnaligned(array.data());
    //Assert(areEqual(simd, expected_constexpr_division) && "Constexpr division test failed");
    //simd = initial_values.data(); // Reset

    //// Addition
    std::vector<T> expected_addition(num_elements);
    std::transform(initial_values.begin(), initial_values.end(), expected_addition.begin(), [](T x) { return x + x; });
    simd = simd + simd;
    simd.storeUnaligned(array.data());
    Assert(areEqual(simd, expected_addition) && "Addition test failed");
    simd = initial_values.data(); // Reset

    // Subtraction
    std::vector<T> expected_subtraction(num_elements, 0);
    simd = simd - simd;
    simd.storeUnaligned(array.data());
    Assert(areEqual(simd, expected_subtraction) && "Subtraction test failed");
    simd = initial_values.data(); // Reset

    //// Multiplication
    //std::vector<T> expected_multiplication(num_elements);
    //std::transform(initial_values.begin(), initial_values.end(), expected_multiplication.begin(), [](T x) { return x * x; });
    //simd = simd * simd;
    //simd.storeUnaligned(array.data());
    //Assert(areEqual(simd, expected_multiplication) && "Multiplication test failed");
    //std::cout << "Multiplication test passed" << std::endl;
    //simd = initial_values.data(); // Reset

    //// Division (careful with integer division!)
    //std::vector<T> expected_division(num_elements, 1);
    //simd = simd / simd;
    //simd.storeUnaligned(array.data());
    //Assert(areEqual(simd, expected_division) && "Division test failed");
    //std::cout << "Division test passed" << std::endl;
    //simd = initial_values.data(); // Reset

    //// Bitwise AND
    if constexpr (std::is_integral_v<T>) {
        std::vector<T> expected_bitwise_and(num_elements, 51);
        simd = simd & simd;
        simd.storeUnaligned(array.data());
        Assert(areEqual(simd, expected_bitwise_and) && "Bitwise AND test failed");
        simd = initial_values.data(); // Reset
    }

    //// Bitwise OR
    if constexpr (std::is_integral_v<T>) {
        std::vector<T> expected_bitwise_or(num_elements, 51);
        simd = simd | simd;
        simd.storeUnaligned(array.data());
        Assert(areEqual(simd, expected_bitwise_or) && "Bitwise OR test failed");
        simd = initial_values.data(); // Reset
    }

    //// Bitwise XOR
    if constexpr (std::is_integral_v<T>) {
        std::vector<T> expected_bitwise_xor(num_elements, 0);
        simd = simd ^ simd;
        simd.storeUnaligned(array.data());
        Assert(areEqual(simd, expected_bitwise_xor) && "Bitwise XOR test failed");
        simd = initial_values.data(); // Reset
    }

    // Left Shift
    if constexpr (std::is_integral_v<T>) {
        std::vector<T> expected_left_shift(num_elements);
        std::transform(initial_values.begin(), initial_values.end(), expected_left_shift.begin(), [](T x) { return x << 2; }); // shift by 2 bits
        simd = simd << 2;
        simd.storeUnaligned(array.data());
        Assert(areEqual(simd, expected_left_shift) && "Left Shift test failed");
        simd = initial_values.data(); // Reset
    }

    // Right Shift
    if constexpr (std::is_integral_v<T>) {
        std::vector<T> expected_right_shift(num_elements);
        std::transform(initial_values.begin(), initial_values.end(), expected_right_shift.begin(), [](T x) { return x >> 2; }); // shift by 2 bits
        simd = simd >> 2;
        simd.storeUnaligned(array.data());
        Assert(areEqual(simd, expected_right_shift) && "Right Shift test failed");
        simd = initial_values.data(); // Reset
    }

    // Unary Minus
    std::vector<T> expected_unary_minus(num_elements);
    std::transform(initial_values.begin(), initial_values.end(), expected_unary_minus.begin(), std::negate<T>());
    simd = -simd;
    simd.storeUnaligned(array.data());
    
    Assert(areEqual(simd, expected_unary_minus) && "Unary Minus test failed");
    simd = initial_values.data(); // Reset

    // Bitwise NOT (Complement)
    if constexpr (std::is_integral_v<T>) {
        std::vector<T> expected_bitwise_not(num_elements);
        std::transform(initial_values.begin(), initial_values.end(), expected_bitwise_not.begin(), std::bit_not<T>());
        simd = ~simd;
        simd.storeUnaligned(array.data());
        Assert(areEqual(simd, expected_bitwise_not) && "Bitwise NOT test failed");
        simd = initial_values.data(); // Reset
    }
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

        alignas(16) T arr[4] = {1,2,3,4};
        Simd v3(arr);
        for (int i = 0; i < v2.size(); ++i) Assert(v3.extract<T>(i) == arr[i]);

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
        alignas(16) simd_stl::int32 arr[4] = {10,20,30,40};
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

    alignas(64) T src[N];
    for (size_t i = 0; i < N; ++i) src[i] = static_cast<T>(i + 1);

    Simd v(src);


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

    std::vector<T> va(N), vb(N), vc(N);
    for (size_t i = 0; i < N; ++i) {
        va[i] = static_cast<T>(i + 1);
        vb[i] = static_cast<T>(i + 1);
        vc[i] = static_cast<T>(i + 2);
    }

    Simd a(va.data());
    Simd b(vb.data());
    Simd c(vc.data());

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

    {
        alignas(64) T dst[N] = {};
        T srcA[N], srcB[N];

        for (size_t i = 0; i < N; ++i) {
            srcA[i] = static_cast<T>(i + 1);
            srcB[i] = static_cast<T>(100 + i);
        }

        Simd a(srcA);
        Simd b(srcB);

        typename Simd::mask_type m = 0;
        for (size_t i = 0; i < N; i += 2)
            m |= (typename Simd::mask_type(1) << i);

        a.maskBlendStoreUnaligned(dst, m, b);

        for (size_t i = 0; i < N; ++i) {
            if (m & (typename Simd::mask_type(1) << i)) {
                Assert(dst[i] == srcA[i]);
            } else {
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

        Simd a(srcA);
        Simd b(srcB);

        typename Simd::mask_type m = 0;

        for (size_t i = 0; i < N / 2; ++i) {
            m |= (typename Simd::mask_type(1) << i);
        }

        a.maskBlendStoreAligned(dst, m, b);

        for (size_t i = 0; i < N; ++i) {
            if (i < N/2) {
                Assert(dst[i] == srcA[i]);
            } else {
                Assert(dst[i] == srcB[i]);
            }
        }
    }
}

template <simd_stl::arch::CpuFeature _Generation_>
void testArithmetic() {
    testArithmeticOperations<simd_stl::int8, _Generation_>();
    testArithmeticOperations<simd_stl::uint8, _Generation_>();

    testArithmeticOperations<simd_stl::int16, _Generation_>();
    testArithmeticOperations<simd_stl::uint16, _Generation_>();

    testArithmeticOperations<simd_stl::int32, _Generation_>();
    testArithmeticOperations<simd_stl::uint32, _Generation_>();

    testArithmeticOperations<simd_stl::int64, _Generation_>();
    testArithmeticOperations<simd_stl::uint64, _Generation_>();

    testArithmeticOperations<float, _Generation_>();
    testArithmeticOperations<double, _Generation_>();
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
    testArithmetic<simd_stl::arch::CpuFeature::SSE2>();
    testArithmetic<simd_stl::arch::CpuFeature::SSE3>();
    testArithmetic<simd_stl::arch::CpuFeature::SSSE3>();
    testArithmetic<simd_stl::arch::CpuFeature::SSE41>();
    testArithmetic<simd_stl::arch::CpuFeature::SSE42>();
    testArithmetic<simd_stl::arch::CpuFeature::AVX2>();
    testArithmetic<simd_stl::arch::CpuFeature::AVX512F>();
    testArithmetic<simd_stl::arch::CpuFeature::AVX512BW>();
    testArithmetic<simd_stl::arch::CpuFeature::AVX512DQ>();
    testArithmetic<simd_stl::arch::CpuFeature::AVX512VL>();

    //testMethods<simd_stl::arch::CpuFeature::SSE2>();
    //testMethods<simd_stl::arch::CpuFeature::SSE3>();
    //testMethods<simd_stl::arch::CpuFeature::SSSE3>();
    //testMethods<simd_stl::arch::CpuFeature::SSE41>();
    //testMethods<simd_stl::arch::CpuFeature::SSE42>();
    testMethods<simd_stl::arch::CpuFeature::AVX2>();
    testMethods<simd_stl::arch::CpuFeature::AVX512F>();
  /*  testMethods<simd_stl::arch::CpuFeature::AVX512BW>();
    testMethods<simd_stl::arch::CpuFeature::AVX512DQ>();
    testMethods<simd_stl::arch::CpuFeature::AVX512VL>();*/

    return 0;
}
