#include <simd_stl/algorithm/find/Find.h>

#include <iostream>
#include <iomanip>

template <typename _IntType_>
void testFind() {
    std::vector<_IntType_> vec1 = { 1, 2, 3, 4, 5 };
    auto simd_result1 = simd_stl::algorithm::find(vec1.begin(), vec1.end(), 1);
    auto std_result1 = std::find(vec1.begin(), vec1.end(), 1);
    simd_stl_assert(simd_result1 == std_result1);
    simd_stl_assert(*simd_result1 == 1);


    std::vector<_IntType_> vec2 = { 10, 20, 30, 40, 50 };
    auto simd_result2 = simd_stl::algorithm::find(vec2.begin(), vec2.end(), 30);
    auto std_result2 = std::find(vec2.begin(), vec2.end(), 30);
    simd_stl_assert(simd_result2 == std_result2);
    simd_stl_assert(*simd_result2 == 30);


    std::vector<_IntType_> vec3 = { 100, 110, 120, 122, 123 };
    auto simd_result3 = simd_stl::algorithm::find(vec3.begin(), vec3.end(), 123);
    auto std_result3 = std::find(vec3.begin(), vec3.end(), 123);
    simd_stl_assert(simd_result3 == std_result3);
    simd_stl_assert(*simd_result3 == 123);


    std::vector<_IntType_> vec4 = { 1, 2, 3, 4, 5 };
    auto simd_result4 = simd_stl::algorithm::find(vec4.begin(), vec4.end(), 6);
    auto std_result4 = std::find(vec4.begin(), vec4.end(), 6);
    simd_stl_assert(simd_result4 == vec4.end());
    simd_stl_assert(std_result4 == vec4.end());
    simd_stl_assert(simd_result4 == std_result4);


    std::vector<_IntType_> vec5 = {};
    auto simd_result5 = simd_stl::algorithm::find(vec5.begin(), vec5.end(), 1);
    auto std_result5 = std::find(vec5.begin(), vec5.end(), 1);
    simd_stl_assert(simd_result5 == vec5.end());
    simd_stl_assert(std_result5 == vec5.end());
    simd_stl_assert(simd_result5 == std_result5);


    std::vector<_IntType_> vec6;
    vec6.reserve(64);
    vec6 = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 29, 1, 2, 3, 4, 5, 6, 7, 8, 9, 29, 1, 2, 3, 4, 5, 6, 7, 8, 9, 29,
            1, 2, 3, 4, 5, 6, 7, 8, 9, 29, 1, 2, 3, 4, 5, 6, 7, 8, 9, 29, 1, 2, 3, 4, 5, 6, 7, 8, 9, 29, 14 };
    auto simd_result6 = simd_stl::algorithm::find(vec6.begin(), vec6.end(), 14);
    auto std_result6 = std::find(vec6.begin(), vec6.end(), 14);
    simd_stl_assert(simd_result6 == std_result6);
    simd_stl_assert(*simd_result6 == 14);


    std::vector<_IntType_> vec7 = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17 }; // Not divisible by 16
    auto simd_result7 = simd_stl::algorithm::find(vec7.begin(), vec7.end(), 17);
    auto std_result7 = std::find(vec7.begin(), vec7.end(), 17);
    simd_stl_assert(simd_result7 == std_result7);
    simd_stl_assert(*simd_result7 == 17);

    std::vector<_IntType_> vec8(1024, 0);
    vec8[0] = 42;
    auto simd_result8 = simd_stl::algorithm::find(vec8.begin(), vec8.end(), 42);
    auto std_result8 = std::find(vec8.begin(), vec8.end(), 42);
    simd_stl_assert(simd_result8 == std_result8);
    simd_stl_assert(*simd_result8 == 42);


    std::vector<_IntType_> vec9(1024, 0);
    vec9[1023] = 42;
    auto simd_result9 = simd_stl::algorithm::find(vec9.begin(), vec9.end(), 42);
    auto std_result9 = std::find(vec9.begin(), vec9.end(), 42);
    simd_stl_assert(simd_result9 == std_result9);
    simd_stl_assert(*simd_result9 == 42);


    std::vector<_IntType_> vec10(1024, 0);
    vec10[7] = 42;
    auto simd_result10 = simd_stl::algorithm::find(vec10.begin(), vec10.end(), 42);
    auto std_result10 = std::find(vec10.begin(), vec10.end(), 42);
    simd_stl_assert(simd_result10 == std_result10);
    simd_stl_assert(*simd_result10 == 42);

    std::vector<_IntType_> vec11(753, 0);
    auto simd_result11 = simd_stl::algorithm::find(vec11.begin(), vec11.end(), 42);
    auto std_result11 = std::find(vec11.begin(), vec11.end(), 42);
    simd_stl_assert(simd_result11 == std_result11);
}

void testFindIf() {
    std::vector<int> v{ 1, 2, 3, 4, 5 };
    {
        auto it = simd_stl::algorithm::find_if(v.begin(), v.end(),
            [](int x) { return x > 3; });
        simd_stl_assert(it != v.end());
        simd_stl_assert(*it == 4);
    }

    {
        auto it = simd_stl::algorithm::find_if(v.begin(), v.end(),
            [](int x) { return x == 1; });
        simd_stl_assert(it == v.begin());
        simd_stl_assert(*it == 1);
    }

    {
        auto it = simd_stl::algorithm::find_if(v.begin(), v.end(),
            [](int x) { return x == 5; });
        simd_stl_assert(it != v.end());
        simd_stl_assert(*it == 5);
    }

    {
        auto it = simd_stl::algorithm::find_if(v.begin(), v.end(),
            [](int x) { return x == 99; });
        simd_stl_assert(it == v.end());
    }

    {
        std::vector<int> empty;
        auto it = simd_stl::algorithm::find_if(empty.begin(), empty.end(),
            [](int) { return true; });
        simd_stl_assert(it == empty.end());
    }

}

void testFindIfNot() {
    std::vector<int> v{ 1, 2, 3, 4, 5 };

    {
        auto it = simd_stl::algorithm::find_if_not(v.begin(), v.end(),
            [](int x) { return x == 1; });
        simd_stl_assert(it != v.end());
        simd_stl_assert(*it == 2);
    }

    {
        auto it = simd_stl::algorithm::find_if_not(v.begin(), v.end(),
            [](int x) { return x < 5; });
        simd_stl_assert(it != v.end());
        simd_stl_assert(*it == 5);
    }

    {
        auto it = simd_stl::algorithm::find_if_not(v.begin(), v.end(),
            [](int x) { return x < 10; });
        simd_stl_assert(it == v.end());
    }

    {
        std::vector<int> empty;
        auto it = simd_stl::algorithm::find_if_not(empty.begin(), empty.end(),
            [](int) { return false; });
        simd_stl_assert(it == empty.end());
    }
}

int main() {
    testFind<int>();
    testFind<short>();
    testFind<long long>();
    testFind<char>();

    testFind<float>();
    testFind<double>();


    testFindIf();
    testFindIfNot();

    return 0;
}
