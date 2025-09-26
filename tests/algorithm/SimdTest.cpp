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

    //// Addition
    //std::vector<T> expected_addition(num_elements);
    //std::transform(initial_values.begin(), initial_values.end(), expected_addition.begin(), [](T x) { return x + x; });
    //simd = simd + simd;
    //simd.storeUnaligned(array.data());
    //assert(areEqual(simd, expected_addition) && "Addition test failed");
    //simd = initial_values.data(); // Reset

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
    //if constexpr (std::is_integral_v<T>) {
    //    std::vector<T> expected_bitwise_and(num_elements, 51);
    //    simd = simd & simd;
    //    simd.storeUnaligned(array.data());
    //    assert(areEqual(simd, expected_bitwise_and) && "Bitwise AND test failed");
    //    simd = initial_values.data(); // Reset
    //}

    //// Bitwise OR
    //if constexpr (std::is_integral_v<T>) {
    //    std::vector<T> expected_bitwise_or(num_elements, 51);
    //    simd = simd | simd;
    //    simd.storeUnaligned(array.data());
    //    assert(areEqual(simd, expected_bitwise_or) && "Bitwise OR test failed");
    //    simd = initial_values.data(); // Reset
    //}

    //// Bitwise XOR
    //if constexpr (std::is_integral_v<T>) {
    //    std::vector<T> expected_bitwise_xor(num_elements, 0);
    //    simd = simd ^ simd;
    //    simd.storeUnaligned(array.data());
    //    assert(areEqual(simd, expected_bitwise_xor) && "Bitwise XOR test failed");
    //    simd = initial_values.data(); // Reset
    //}

    //// Left Shift
    //if constexpr (std::is_integral_v<T>) {
    //    std::vector<T> expected_left_shift(num_elements);
    //    std::transform(initial_values.begin(), initial_values.end(), expected_left_shift.begin(), [](T x) { return x << 2; }); // shift by 2 bits
    //    simd = simd << 2;
    //    simd.storeUnaligned(array.data());
    //    assert(areEqual(simd, expected_left_shift) && "Left Shift test failed");
    //    std::cout << "Left Shift test passed" << std::endl;
    //    simd = initial_values.data(); // Reset
    //}

    ////// Right Shift
    ////if constexpr (std::is_integral_v<T>) {
    ////    std::vector<T> expected_right_shift(num_elements);
    ////    std::transform(initial_values.begin(), initial_values.end(), expected_right_shift.begin(), [](T x) { return x >> 2; }); // shift by 2 bits
    ////    simd = simd >> 2;
    ////    simd.storeUnaligned(array.data());
    ////    assert(areEqual(simd, expected_right_shift) && "Right Shift test failed");
    ////    std::cout << "Right Shift test passed" << std::endl;
    ////    simd = initial_values.data(); // Reset
    ////}

    //// Unary Minus
    //std::vector<T> expected_unary_minus(num_elements);
    //std::transform(initial_values.begin(), initial_values.end(), expected_unary_minus.begin(), std::negate<T>());
    //simd = -simd;
    //simd.storeUnaligned(array.data());
    //
    //assert(areEqual(simd, expected_unary_minus) && "Unary Minus test failed");
    //simd = initial_values.data(); // Reset

    //// Bitwise NOT (Complement)
    //if constexpr (std::is_integral_v<T>) {
    //    std::vector<T> expected_bitwise_not(num_elements);
    //    std::transform(initial_values.begin(), initial_values.end(), expected_bitwise_not.begin(), std::bit_not<T>());
    //    simd = ~simd;
    //    simd.storeUnaligned(array.data());
    //    assert(areEqual(simd, expected_bitwise_not) && "Bitwise NOT test failed");
    //    simd = initial_values.data(); // Reset
    //}
}

int main() {
    //testArithmeticOperations<simd_stl::int8, simd_stl::arch::CpuFeature::SSE2>();
    //testArithmeticOperations<simd_stl::uint8, simd_stl::arch::CpuFeature::SSE2>();

    //testArithmeticOperations<simd_stl::int16, simd_stl::arch::CpuFeature::SSE2>();
    //testArithmeticOperations<simd_stl::uint16, simd_stl::arch::CpuFeature::SSE2>();

    //testArithmeticOperations<simd_stl::int32, simd_stl::arch::CpuFeature::SSE2>();
    //testArithmeticOperations<simd_stl::uint32, simd_stl::arch::CpuFeature::SSE2>();

    testArithmeticOperations<simd_stl::int64, simd_stl::arch::CpuFeature::SSE2>();
    testArithmeticOperations<simd_stl::uint64, simd_stl::arch::CpuFeature::SSE2>();


    //testArithmeticOperations<float, simd_stl::arch::CpuFeature::SSE2>();
    //testArithmeticOperations<float, simd_stl::arch::CpuFeature::SSE2>();

 //   testArithmeticOperations<double, simd_stl::arch::CpuFeature::SSE2>();
   // testArithmeticOperations<double, simd_stl::arch::CpuFeature::SSE2>();

    return 0;
}
