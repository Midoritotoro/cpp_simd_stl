#include <simd_stl/datapar/BasicSimd.h>
#include <string>

template <typename T, simd_stl::arch::CpuFeature Arch>
bool areEqual(simd_stl::datapar::simd<Arch, T>& simd, const std::vector<T>& vec) {
    std::vector<T> simd_data(vec.size());
    simd.storeUnaligned(simd_data.data());
    return std::equal(simd_data.begin(), simd_data.end(), vec.begin());
}

template <typename T, simd_stl::arch::CpuFeature Arch>
void testArithmeticOperations() {
    using Simd = simd_stl::datapar::simd<Arch, T>;

    constexpr size_t num_elements = simd_stl::datapar::simd<Arch, T>::template size(); // Get the number of elements in SIMD vector
    std::vector<T> initial_values(num_elements, 51);
    simd_stl::datapar::simd<Arch, T> simd = simd_stl::datapar::simd<Arch, T>::loadUnaligned(initial_values.data());
    std::vector<T> array(num_elements);

    // Division by constexpr number 
    //std::vector<T> expected_constexpr_division(num_elements, 1);
    //simd.divideByConst<2>();
    //simd.storeUnaligned(array.data());
    //simd_stl_assert(areEqual(simd, expected_constexpr_division) && "Constexpr division test failed");
    //simd = simd_stl::datapar::basic_simd<Arch, T>::loadUnaligned(initial_values.data()); // Reset

    //// Addition
    std::vector<T> expected_addition(num_elements);
    std::transform(initial_values.begin(), initial_values.end(), expected_addition.begin(), [](T x) { return x + x; });
    simd = simd + simd;
    simd.storeUnaligned(array.data());
    simd_stl_assert(areEqual(simd, expected_addition) && "Addition test failed");
    simd = simd_stl::datapar::simd<Arch, T>::loadUnaligned(initial_values.data()); // Reset

    // Subtraction
    std::vector<T> expected_subtraction(num_elements, 0);
    simd = simd - simd;
    simd.storeUnaligned(array.data());
    simd_stl_assert(areEqual(simd, expected_subtraction) && "Subtraction test failed");
    simd = simd_stl::datapar::simd<Arch, T>::loadUnaligned(initial_values.data()); // Reset

    //// Multiplication
    //std::vector<T> expected_multiplication(num_elements);
    //std::transform(initial_values.begin(), initial_values.end(), expected_multiplication.begin(), [](T x) { return x * x; });
    //simd = simd * simd;
    //simd.storeUnaligned(array.data());
    //simd_stl_assert(areEqual(simd, expected_multiplication) && "Multiplication test failed");
    //std::cout << "Multiplication test passed" << std::endl;
    //simd = simd_stl::datapar::basic_simd<Arch, T>::loadUnaligned(initial_values.data()); // Reset

    //// Division (careful with integer division!)
    //std::vector<T> expected_division(num_elements, 1);
    //simd = simd / simd;
    //simd.storeUnaligned(array.data());
    //simd_stl_assert(areEqual(simd, expected_division) && "Division test failed");
    //std::cout << "Division test passed" << std::endl;
    //simd = simd_stl::datapar::basic_simd<Arch, T>::loadUnaligned(initial_values.data()); // Reset

    //// Bitwise AND
    if constexpr (std::is_integral_v<T>) {
        std::vector<T> expected_bitwise_and(num_elements, 51);
        simd = simd & simd;
        simd.storeUnaligned(array.data());
        simd_stl_assert(areEqual(simd, expected_bitwise_and) && "Bitwise AND test failed");
        simd = simd_stl::datapar::simd<Arch, T>::loadUnaligned(initial_values.data()); // Reset
    }

    //// Bitwise OR
    if constexpr (std::is_integral_v<T>) {
        std::vector<T> expected_bitwise_or(num_elements, 51);
        simd = simd | simd;
        simd.storeUnaligned(array.data());
        simd_stl_assert(areEqual(simd, expected_bitwise_or) && "Bitwise OR test failed");
        simd = simd_stl::datapar::simd<Arch, T>::loadUnaligned(initial_values.data()); // Reset
    }

    //// Bitwise XOR
    if constexpr (std::is_integral_v<T>) {
        std::vector<T> expected_bitwise_xor(num_elements, 0);
        simd = simd ^ simd;
        simd.storeUnaligned(array.data());
        simd_stl_assert(areEqual(simd, expected_bitwise_xor) && "Bitwise XOR test failed");
        simd = simd_stl::datapar::simd<Arch, T>::loadUnaligned(initial_values.data()); // Reset
    }

    // Left Shift
    if constexpr (std::is_integral_v<T>) {
        std::vector<T> expected_left_shift(num_elements);
        std::transform(initial_values.begin(), initial_values.end(), expected_left_shift.begin(), [](T x) { return x << 2; }); // shift by 2 bits
        simd = simd << 2;
        simd.storeUnaligned(array.data());
        simd_stl_assert(areEqual(simd, expected_left_shift) && "Left Shift test failed");
        simd = simd_stl::datapar::simd<Arch, T>::loadUnaligned(initial_values.data()); // Reset
    }

    // Right Shift
    if constexpr (std::is_integral_v<T>) {
        std::vector<T> expected_right_shift(num_elements);
        std::transform(initial_values.begin(), initial_values.end(), expected_right_shift.begin(), [](T x) { return x >> 2; }); // shift by 2 bits
        simd = simd >> 2;
        simd.storeUnaligned(array.data());
        simd_stl_assert(areEqual(simd, expected_right_shift) && "Right Shift test failed");
        simd = simd_stl::datapar::simd<Arch, T>::loadUnaligned(initial_values.data()); // Reset
    }

    // Unary Minus
    std::vector<T> expected_unary_minus(num_elements);
    std::transform(initial_values.begin(), initial_values.end(), expected_unary_minus.begin(), std::negate<T>());
    simd = -simd;
    simd.storeUnaligned(array.data());
    
    simd_stl_assert(areEqual(simd, expected_unary_minus) && "Unary Minus test failed");
    simd = simd_stl::datapar::simd<Arch, T>::loadUnaligned(initial_values.data()); // Reset

    // Bitwise NOT (Complement)
    if constexpr (std::is_integral_v<T>) {
        std::vector<T> expected_bitwise_not(num_elements);
        std::transform(initial_values.begin(), initial_values.end(), expected_bitwise_not.begin(), std::bit_not<T>());
        simd = ~simd;
        simd.storeUnaligned(array.data());
        simd_stl_assert(areEqual(simd, expected_bitwise_not) && "Bitwise NOT test failed");
        simd = simd_stl::datapar::simd<Arch, T>::loadUnaligned(initial_values.data()); // Reset
    }
}

template <simd_stl::arch::CpuFeature _Generation_>
void testArithmetic() {
    testArithmeticOperations<simd_stl::int8, _>();
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

    return 0;
}
