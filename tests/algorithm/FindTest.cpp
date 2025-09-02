#include <simd_stl/algorithm/find/Find.h>

#include <iostream>
#include <iomanip>

void testFind() {
    std::vector<int> vec1 = { 1, 2, 3, 4, 5 };
    auto simd_result1 = simd_stl::algorithm::find(vec1.begin(), vec1.end(), 1);
    auto std_result1 = std::find(vec1.begin(), vec1.end(), 1);
    Assert(simd_result1 == std_result1);
    Assert(*simd_result1 == 1);


    std::vector<int> vec2 = { 10, 20, 30, 40, 50 };
    auto simd_result2 = simd_stl::algorithm::find(vec2.begin(), vec2.end(), 30);
    auto std_result2 = std::find(vec2.begin(), vec2.end(), 30);
    Assert(simd_result2 == std_result2);
    Assert(*simd_result2 == 30);


    std::vector<int> vec3 = { 100, 200, 300, 400, 500 };
    auto simd_result3 = simd_stl::algorithm::find(vec3.begin(), vec3.end(), 500);
    auto std_result3 = std::find(vec3.begin(), vec3.end(), 500);
    Assert(simd_result3 == std_result3);
    Assert(*simd_result3 == 500);


    std::vector<int> vec4 = { 1, 2, 3, 4, 5 };
    auto simd_result4 = simd_stl::algorithm::find(vec4.begin(), vec4.end(), 6);
    auto std_result4 = std::find(vec4.begin(), vec4.end(), 6);
    Assert(simd_result4 == vec4.end());
    Assert(std_result4 == vec4.end());
    Assert(simd_result4 == std_result4);


    std::vector<int> vec5 = {};
    auto simd_result5 = simd_stl::algorithm::find(vec5.begin(), vec5.end(), 1);
    auto std_result5 = std::find(vec5.begin(), vec5.end(), 1);
    Assert(simd_result5 == vec5.end());
    Assert(std_result5 == vec5.end());
    Assert(simd_result5 == std_result5);


    std::vector<int> vec6;
    vec6.reserve(64);
    vec6 = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 29, 1, 2, 3, 4, 5, 6, 7, 8, 9, 29, 1, 2, 3, 4, 5, 6, 7, 8, 9, 29,
            1, 2, 3, 4, 5, 6, 7, 8, 9, 29, 1, 2, 3, 4, 5, 6, 7, 8, 9, 29, 1, 2, 3, 4, 5, 6, 7, 8, 9, 29, 14 };
    auto simd_result6 = simd_stl::algorithm::find(vec6.begin(), vec6.end(), 14);
    auto std_result6 = std::find(vec6.begin(), vec6.end(), 14);
    Assert(simd_result6 == std_result6);
    Assert(*simd_result6 == 14);


    std::vector<int> vec7 = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17 }; // Not divisible by 16
    auto simd_result7 = simd_stl::algorithm::find(vec7.begin(), vec7.end(), 17);
    auto std_result7 = std::find(vec7.begin(), vec7.end(), 17);
    Assert(simd_result7 == std_result7);
    Assert(*simd_result7 == 17);

    std::vector<int> vec8(1024, 0);
    vec8[0] = 42;
    auto simd_result8 = simd_stl::algorithm::find(vec8.begin(), vec8.end(), 42);
    auto std_result8 = std::find(vec8.begin(), vec8.end(), 42);
    Assert(simd_result8 == std_result8);
    Assert(*simd_result8 == 42);


    std::vector<int> vec9(1024, 0);
    vec9[1023] = 42;
    auto simd_result9 = simd_stl::algorithm::find(vec9.begin(), vec9.end(), 42);
    auto std_result9 = std::find(vec9.begin(), vec9.end(), 42);
    Assert(simd_result9 == std_result9);
    Assert(*simd_result9 == 42);


    std::vector<int> vec10(1024, 0);
    vec10[7] = 42;
    auto simd_result10 = simd_stl::algorithm::find(vec10.begin(), vec10.end(), 42);
    auto std_result10 = std::find(vec10.begin(), vec10.end(), 42);
    Assert(simd_result10 == std_result10);
    Assert(*simd_result10 == 42);
}

int main() {
    testFind();
    return 0;
}
