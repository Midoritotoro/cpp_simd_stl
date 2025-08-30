#include <simd_stl/algorithm/find/Find.h>

#include <iostream>

void testFind() {
    // Test case 1: Ёлемент найден в начале вектора
    std::vector<int> vec1 = { 1, 2, 3, 4, 5 };
    auto simd_result1 = simd_stl::algorithm::find(vec1.begin(), vec1.end(), 1);
    auto std_result1 = std::find(vec1.begin(), vec1.end(), 1);
    assert(simd_result1 == std_result1);
    assert(*simd_result1 == 1);

    // Test case 2: Ёлемент найден в середине вектора
    std::vector<int> vec2 = { 10, 20, 30, 40, 50 };
    auto simd_result2 = simd_stl::algorithm::find(vec2.begin(), vec2.end(), 30);
    auto std_result2 = std::find(vec2.begin(), vec2.end(), 30);
    assert(simd_result2 == std_result2);
    assert(*simd_result2 == 30);

    // Test case 3: Ёлемент найден в конце вектора
    std::vector<int> vec3 = { 100, 200, 300, 400, 500 };
    auto simd_result3 = simd_stl::algorithm::find(vec3.begin(), vec3.end(), 500);
    auto std_result3 = std::find(vec3.begin(), vec3.end(), 500);
    assert(simd_result3 == std_result3);
    assert(*simd_result3 == 500);

    // Test case 4: Ёлемент не найден
    std::vector<int> vec4 = { 1, 2, 3, 4, 5 };
    auto simd_result4 = simd_stl::algorithm::find(vec4.begin(), vec4.end(), 6);
    auto std_result4 = std::find(vec4.begin(), vec4.end(), 6);
    assert(simd_result4 == vec4.end());
    assert(std_result4 == vec4.end());
    assert(simd_result4 == std_result4);

    // Test case 5: ѕустой вектор
    std::vector<int> vec5 = {};
    auto simd_result5 = simd_stl::algorithm::find(vec5.begin(), vec5.end(), 1);
    auto std_result5 = std::find(vec5.begin(), vec5.end(), 1);
    assert(simd_result5 == vec5.end());
    assert(std_result5 == vec5.end());
    assert(simd_result5 == std_result5);

    // Test case 6: Ѕольшой вектор (с вашими данными)
    std::vector<int> vec6;
    vec6.reserve(64);
    vec6 = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 29, 1, 2, 3, 4, 5, 6, 7, 8, 9, 29, 1, 2, 3, 4, 5, 6, 7, 8, 9, 29,
            1, 2, 3, 4, 5, 6, 7, 8, 9, 29, 1, 2, 3, 4, 5, 6, 7, 8, 9, 29, 1, 2, 3, 4, 5, 6, 7, 8, 9, 29, 14 };
    auto simd_result6 = simd_stl::algorithm::find(vec6.begin(), vec6.end(), 14);
    auto std_result6 = std::find(vec6.begin(), vec6.end(), 14);
    assert(simd_result6 == std_result6);
    assert(*simd_result6 == 14);

    // Test case 7: Ёлемент находитс€ в конце выровненного SIMD блока
    std::vector<int> vec7 = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17 }; // Not divisible by 16
    auto simd_result7 = simd_stl::algorithm::find(vec7.begin(), vec7.end(), 17);
    auto std_result7 = std::find(vec7.begin(), vec7.end(), 17);
    assert(simd_result7 == std_result7);
    assert(*simd_result7 == 17);

    // Test case 8: ѕоиск в начале большого вектора. ¬ажно дл€ проверки корректности SIMD-оптимизаций.
    std::vector<int> vec8(1024, 0);
    vec8[0] = 42;
    auto simd_result8 = simd_stl::algorithm::find(vec8.begin(), vec8.end(), 42);
    auto std_result8 = std::find(vec8.begin(), vec8.end(), 42);
    assert(simd_result8 == std_result8);
    assert(*simd_result8 == 42);

    // Test case 9: ѕоиск в конце большого вектора.
    std::vector<int> vec9(1024, 0);
    vec9[1023] = 42;
    auto simd_result9 = simd_stl::algorithm::find(vec9.begin(), vec9.end(), 42);
    auto std_result9 = std::find(vec9.begin(), vec9.end(), 42);
    assert(simd_result9 == std_result9);
    assert(*simd_result9 == 42);

    // Test case 10: ѕоиск элемента, близкого к началу, но не в самом начале.
    std::vector<int> vec10(1024, 0);
    vec10[7] = 42;
    auto simd_result10 = simd_stl::algorithm::find(vec10.begin(), vec10.end(), 42);
    auto std_result10 = std::find(vec10.begin(), vec10.end(), 42);
    assert(simd_result10 == std_result10);
    assert(*simd_result10 == 42);

    std::cout << "All tests passed!" << std::endl;
}

int main() {
	testFind();
	return 0;
}