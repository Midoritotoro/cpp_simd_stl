﻿#include <cassert>
#include <vector>
#include <string>
#include <list>
#include <deque>
#include <array>
#include <iostream>
#include <algorithm>
#include <iterator>
#include <chrono>
#include <simd_stl/algorithm/copy/MoveBackward.h>
#include <simd_stl/numeric/BasicSimd.h>

template <typename It1, typename It2>
void assert_equal(It1 first1, It1 last1, It2 first2) {
    auto n = std::distance(first1, last1);
    for (decltype(n) i = 0; i < n; ++i) {
        assert(*(first1 + i) == *(first2 + i));
    }
}

// Нетривиальный тип для проверки перемещений
struct MoveCounted {
    int value;
    static inline size_t moves = 0;
    MoveCounted() : value(0) {}
    explicit MoveCounted(int v) : value(v) {}
    MoveCounted(const MoveCounted&) = delete;
    MoveCounted& operator=(const MoveCounted&) = delete;
    MoveCounted(MoveCounted&& other) noexcept : value(other.value) {
        ++moves;
        other.value = -999;
    }
    MoveCounted& operator=(MoveCounted&& other) noexcept {
        value = other.value;
        ++moves;
        other.value = -999;
        return *this;
    }
    bool operator==(const MoveCounted& rhs) const noexcept { return value == rhs.value; }
};

int main() {
    // Пустой диапазон
    {
        std::vector<int> src;
        std::vector<int> dst;
        auto out = simd_stl::algorithm::move_backward(src.begin(), src.end(), dst.end());
        assert(out == dst.end());
    }

    // Один элемент
    {
        std::vector<int> src = { 42 };
        std::vector<int> dst(1, 0);
        auto out = simd_stl::algorithm::move_backward(src.begin(), src.end(), dst.end());
        assert(out == dst.begin());
        assert(dst[0] == 42);
    }

    // Несколько элементов
    {
        std::vector<int> src = { 1,2,3,4,5 };
        std::vector<int> dst(5, 0);
        auto out = simd_stl::algorithm::move_backward(src.begin(), src.end(), dst.end());
        assert(out == dst.begin());
        assert_equal(dst.begin(), dst.end(), std::vector<int>{1, 2, 3, 4, 5}.begin());
    }

    // Строки
    {
        std::string src = "hello";
        std::string dst(5, '_');
        auto out = simd_stl::algorithm::move_backward(src.begin(), src.end(), dst.end());
        assert(out == dst.begin());
        assert(dst == "hello");
    }

    // Перекрытие: сдвиг вправо
    {
        std::vector<int> v = { 1,2,3,4,5,6,7,8 };
        // simd_stl::algorithm::move_backward [0..4) → [2..6)
        simd_stl::algorithm::move_backward(v.begin(), v.begin() + 4, v.begin() + 6);
        std::vector<int> expected = { 1,2,1,2,3,4,7,8 };
        assert(v == expected);
    }

    // Перекрытие: сдвиг влево
    {
        std::vector<int> v = { 10,20,30,40,50 };
        // simd_stl::algorithm::move_backward [2..5) → [0..3)
        simd_stl::algorithm::move_backward(v.begin() + 2, v.end(), v.begin() + 3);
        std::vector<int> expected = { 30,40,50,40,50 };
        assert(v == expected);
    }

    {
        const size_t N = 1'000'000;
        std::vector<int> src(N);
        for (size_t i = 0; i < N; ++i) src[i] = static_cast<int>(i);

        std::vector<int> dst(N, -1);

        auto start = std::chrono::high_resolution_clock::now();
        auto out = simd_stl::algorithm::move_backward(src.begin(), src.end(), dst.end());
        auto stop = std::chrono::high_resolution_clock::now();

        assert(out == dst.begin());
        for (size_t i = 0; i < N; i += N / 10) assert(dst[i] == static_cast<int>(i));

        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
        std::cout << "Large simd_stl::algorithm::move_backward of " << N << " ints took " << ms << " ms\n";
    }

    {
        MoveCounted::moves = 0;
        std::vector<MoveCounted> src;
        for (int i = 0; i < 100; ++i) src.emplace_back(i);
        std::vector<MoveCounted> dst(src.size());

        auto out = simd_stl::algorithm::move_backward(src.begin(), src.end(), dst.end());
        assert(out == dst.begin());
        for (int i = 0; i < 100; ++i) assert(dst[i].value == i);
        assert(MoveCounted::moves >= 100);
    }
    // deque
    {
        std::deque<int> src = { 1,2,3,4,5 };
        std::deque<int> dst(5);
        auto out = simd_stl::algorithm::move_backward(src.begin(), src.end(), dst.end());
        assert(out == dst.begin());
        for (size_t i = 0; i < src.size(); ++i) assert(dst[i] == i + 1);
    }

    // list
    {
        std::list<int> src = { 7,8,9 };
        std::list<int> dst(3);
        auto out = simd_stl::algorithm::move_backward(src.begin(), src.end(), dst.end());
        auto itS = src.begin();
        auto itD = dst.begin();
        for (; itS != src.end(); ++itS, ++itD) assert(*itS == *itD);
    }

    // array
    {
        std::array<int, 4> src = { 1,2,3,4 };
        std::array<int, 4> dst = {};
        auto out = simd_stl::algorithm::move_backward(src.begin(), src.end(), dst.end());
        assert(out == dst.begin());
        for (size_t i = 0; i < src.size(); ++i) assert(dst[i] == src[i]);
    }

    return 0;
}
