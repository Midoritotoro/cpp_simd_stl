#include <simd_stl/algorithm/find/Contains.h>

#include <vector>
#include <array>

using simd_stl::algorithm::contains;

void containsTest() {
    // 1. Пустой диапазон
    {
        std::vector<int> v;
        simd_stl_assert(!contains(v.begin(), v.end(), 42));
    }

    // 2. Элемент присутствует
    {
        std::vector<int> v = { 1, 2, 3, 4, 5 };
        simd_stl_assert(contains(v.begin(), v.end(), 3));
        simd_stl_assert(contains(v.begin(), v.end(), 1));
        simd_stl_assert(contains(v.begin(), v.end(), 5));
    }

    // 3. Элемент отсутствует
    {
        std::vector<int> v = { 10, 20, 30 };
        simd_stl_assert(!contains(v.begin(), v.end(), 25));
        simd_stl_assert(!contains(v.begin(), v.end(), 0));
    }

    // 4. Проверка с массивом (итераторы на сырой массив)
    {
        int arr[] = { 7, 8, 9 };
        simd_stl_assert(contains(std::begin(arr), std::end(arr), 8));
        simd_stl_assert(!contains(std::begin(arr), std::end(arr), 42));
    }

    // 5. Проверка с const‑диапазоном
    {
        const std::array<char, 4> arr = { 'a', 'b', 'c', 'd' };
        simd_stl_assert(contains(arr.begin(), arr.end(), 'c'));
        simd_stl_assert(!contains(arr.begin(), arr.end(), 'z'));
    }

    // 6. Проверка с разными типами (например, поиск double в массиве double)
    {
        std::array<double, 3> arr = { 1.0, 2.5, 3.14 };
        simd_stl_assert(contains(arr.begin(), arr.end(), 3.14));
        simd_stl_assert(!contains(arr.begin(), arr.end(), 2.0));
    }

    // 7. Проверка с указателями (итераторы = сырые указатели)
    {
        int arr[] = { 100, 200, 300 };
        simd_stl_assert(contains(&arr[0], &arr[3], 200));
        simd_stl_assert(!contains(&arr[0], &arr[3], 400));
    }
}


int main() {
    containsTest();
    return 0;
}
