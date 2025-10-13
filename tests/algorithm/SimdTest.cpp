#include <simd_stl/numeric/BasicSimd.h>
#include <string>

template <typename T, simd_stl::arch::CpuFeature Arch>
bool areEqual(simd_stl::numeric::basic_simd<Arch, T>& simd, const std::vector<T>& vec) {
    std::vector<T> simd_data(vec.size());
    simd.storeUnaligned(simd_data.data());
    return std::equal(simd_data.begin(), simd_data.end(), vec.begin());
}

template <typename T, simd_stl::arch::CpuFeature Arch>
void testArithmeticOperations() {
    constexpr size_t num_elements = simd_stl::numeric::basic_simd<Arch, T>::template size(); // Get the number of elements in SIMD vector
    std::vector<T> initial_values(num_elements, 51);
    simd_stl::numeric::basic_simd<Arch, T> simd(initial_values.data());
    std::vector<T> array(num_elements);

    // Division by constexpr number 
    //std::vector<T> expected_constexpr_division(num_elements, 1);
    //simd.divideByConst<2>();
    //simd.storeUnaligned(array.data());
    //assert(areEqual(simd, expected_constexpr_division) && "Constexpr division test failed");
    //simd = initial_values.data(); // Reset

    //// Addition
    std::vector<T> expected_addition(num_elements);
    std::transform(initial_values.begin(), initial_values.end(), expected_addition.begin(), [](T x) { return x + x; });
    simd = simd + simd;
    simd.storeUnaligned(array.data());
    assert(areEqual(simd, expected_addition) && "Addition test failed");
    simd = initial_values.data(); // Reset

    // Subtraction
    std::vector<T> expected_subtraction(num_elements, 0);
    simd = simd - simd;
    simd.storeUnaligned(array.data());
    assert(areEqual(simd, expected_subtraction) && "Subtraction test failed");
    simd = initial_values.data(); // Reset

    //// Multiplication
    //std::vector<T> expected_multiplication(num_elements);
    //std::transform(initial_values.begin(), initial_values.end(), expected_multiplication.begin(), [](T x) { return x * x; });
    //simd = simd * simd;
    //simd.storeUnaligned(array.data());
    //assert(areEqual(simd, expected_multiplication) && "Multiplication test failed");
    //std::cout << "Multiplication test passed" << std::endl;
    //simd = initial_values.data(); // Reset

    //// Division (careful with integer division!)
    //std::vector<T> expected_division(num_elements, 1);
    //simd = simd / simd;
    //simd.storeUnaligned(array.data());
    //assert(areEqual(simd, expected_division) && "Division test failed");
    //std::cout << "Division test passed" << std::endl;
    //simd = initial_values.data(); // Reset

    //// Bitwise AND
    if constexpr (std::is_integral_v<T>) {
        std::vector<T> expected_bitwise_and(num_elements, 51);
        simd = simd & simd;
        simd.storeUnaligned(array.data());
        assert(areEqual(simd, expected_bitwise_and) && "Bitwise AND test failed");
        simd = initial_values.data(); // Reset
    }

    //// Bitwise OR
    if constexpr (std::is_integral_v<T>) {
        std::vector<T> expected_bitwise_or(num_elements, 51);
        simd = simd | simd;
        simd.storeUnaligned(array.data());
        assert(areEqual(simd, expected_bitwise_or) && "Bitwise OR test failed");
        simd = initial_values.data(); // Reset
    }

    //// Bitwise XOR
    if constexpr (std::is_integral_v<T>) {
        std::vector<T> expected_bitwise_xor(num_elements, 0);
        simd = simd ^ simd;
        simd.storeUnaligned(array.data());
        assert(areEqual(simd, expected_bitwise_xor) && "Bitwise XOR test failed");
        simd = initial_values.data(); // Reset
    }

    // Left Shift
    if constexpr (std::is_integral_v<T>) {
        std::vector<T> expected_left_shift(num_elements);
        std::transform(initial_values.begin(), initial_values.end(), expected_left_shift.begin(), [](T x) { return x << 2; }); // shift by 2 bits
        simd = simd << 2;
        simd.storeUnaligned(array.data());
        assert(areEqual(simd, expected_left_shift) && "Left Shift test failed");
        simd = initial_values.data(); // Reset
    }

    // Right Shift
    if constexpr (std::is_integral_v<T>) {
        std::vector<T> expected_right_shift(num_elements);
        std::transform(initial_values.begin(), initial_values.end(), expected_right_shift.begin(), [](T x) { return x >> 2; }); // shift by 2 bits
        simd = simd >> 2;
        simd.storeUnaligned(array.data());
        assert(areEqual(simd, expected_right_shift) && "Right Shift test failed");
        simd = initial_values.data(); // Reset
    }

    // Unary Minus
    std::vector<T> expected_unary_minus(num_elements);
    std::transform(initial_values.begin(), initial_values.end(), expected_unary_minus.begin(), std::negate<T>());
    simd = -simd;
    simd.storeUnaligned(array.data());
    
    assert(areEqual(simd, expected_unary_minus) && "Unary Minus test failed");
    simd = initial_values.data(); // Reset

    // Bitwise NOT (Complement)
    if constexpr (std::is_integral_v<T>) {
        std::vector<T> expected_bitwise_not(num_elements);
        std::transform(initial_values.begin(), initial_values.end(), expected_bitwise_not.begin(), std::bit_not<T>());
        simd = ~simd;
        simd.storeUnaligned(array.data());
        assert(areEqual(simd, expected_bitwise_not) && "Bitwise NOT test failed");
        simd = initial_values.data(); // Reset
    }
}

template <typename T, simd_stl::arch::CpuFeature Arch>
void testMethods() {
    using Simd = simd_stl::numeric::basic_simd<Arch, T>;
    
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
        assert(v.extract(0) == 77);*/
    }

    // --- convert / bitcast / safeBitcast ---
    {
        Simd v(5);
        auto v8 = v.convert<simd_stl::numeric::basic_simd<simd_stl::arch::CpuFeature::SSE2, simd_stl::int8>>();
        auto vDouble = v.bitcast<double>();
        auto vSafe = Simd::template safeBitcast<simd_stl::numeric::basic_simd<simd_stl::arch::CpuFeature::SSE2, simd_stl::int32>>(v);
    }

    // --- cross‑arch bitcast ---
    {
        Simd v(11);
        auto vOther = v.bitcast<simd_stl::arch::CpuFeature::SSE2, float>();
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


    {

    }
}


int main() {
    
    testArithmeticOperations<simd_stl::int8, simd_stl::arch::CpuFeature::SSE2>();
    testArithmeticOperations<simd_stl::uint8, simd_stl::arch::CpuFeature::SSE2>();

    testArithmeticOperations<simd_stl::int16, simd_stl::arch::CpuFeature::SSE2>();
    testArithmeticOperations<simd_stl::uint16, simd_stl::arch::CpuFeature::SSE2>();

    testArithmeticOperations<simd_stl::int32, simd_stl::arch::CpuFeature::SSE2>();
    testArithmeticOperations<simd_stl::uint32, simd_stl::arch::CpuFeature::SSE2>();

    testArithmeticOperations<simd_stl::int64, simd_stl::arch::CpuFeature::SSE2>();
    testArithmeticOperations<simd_stl::uint64, simd_stl::arch::CpuFeature::SSE2>();


    testMethods<simd_stl::int8, simd_stl::arch::CpuFeature::SSE2>();
    testMethods<simd_stl::uint8, simd_stl::arch::CpuFeature::SSE2>();

    testMethods<simd_stl::int16, simd_stl::arch::CpuFeature::SSE2>();
    testMethods<simd_stl::uint16, simd_stl::arch::CpuFeature::SSE2>();

    testMethods<simd_stl::int32, simd_stl::arch::CpuFeature::SSE2>();
    testMethods<simd_stl::uint32, simd_stl::arch::CpuFeature::SSE2>();

    testMethods<simd_stl::int64, simd_stl::arch::CpuFeature::SSE2>();
    testMethods<simd_stl::uint64, simd_stl::arch::CpuFeature::SSE2>();

    testMethods<float, simd_stl::arch::CpuFeature::SSE2>();
    testMethods<double, simd_stl::arch::CpuFeature::SSE2>();

    return 0;
}
