#include <cassert>
#include <vector>
#include <string>
#include <list>
#include <deque>
#include <array>
#include <iostream>
#include <algorithm>
#include <iterator>
#include <random>
#include <chrono>
#include <simd_stl/algorithm/Copy/CopyBackward.h>

template <typename It1, typename It2>
void assert_equal(It1 first1, It1 last1, It2 first2) {
    auto n = std::distance(first1, last1);
    for (decltype(n) i = 0; i < n; ++i) {
        assert(*(first1 + i) == *(first2 + i));
    }
}

int main() {
    {
        std::vector<int> src;
        std::vector<int> dst;
        auto out = simd_stl::algorithm::copy_backward(src.begin(), src.end(), dst.end());
        assert(out == dst.end());
    }

    {
        std::vector<int> src = { 42 };
        std::vector<int> dst(1, 0);
        auto out = simd_stl::algorithm::copy_backward(src.begin(), src.end(), dst.end());
        assert(out == dst.begin());
        assert(dst[0] == 42);
    }

    {
        std::vector<int> src = { 1,2,3,4,5 };
        std::vector<int> dst(5, 0);
        auto out = simd_stl::algorithm::copy_backward(src.begin(), src.end(), dst.end());
        assert(out == dst.begin());
        assert_equal(src.begin(), src.end(), dst.begin());
    }

    {
        std::string src = "hello";
        std::string dst(5, '_');
        auto out = simd_stl::algorithm::copy_backward(src.begin(), src.end(), dst.end());
        assert(out == dst.begin());
        assert(src == dst);
    }

    {
        std::vector<int> v = { 1,2,3,4,5,6,7,8 };
        simd_stl::algorithm::copy_backward(v.begin(), v.begin() + 4, v.begin() + 6);
        std::vector<int> expected = { 1,2,1,2,3,4,7,8 };
        assert(v == expected);
    }

    {
        std::vector<int> v = { 10,20,30,40,50 };
        simd_stl::algorithm::copy_backward(v.begin(), v.begin() + 3, v.end());
        std::vector<int> expected = { 10,20,10,20,30 };
        assert(v == expected);
    }

    {
        const size_t N = 1'000'000;
        std::vector<int> src(N);
        for (size_t i = 0; i < N; ++i) src[i] = static_cast<int>(i);

        std::vector<int> dst(N, -1);

        auto out = simd_stl::algorithm::copy_backward(src.begin(), src.end(), dst.end());

        assert(out == dst.begin());
        assert(src == dst);

    }

    {
        std::deque<int> src = { 1,2,3,4,5 };
        std::deque<int> dst(5);
        auto out = simd_stl::algorithm::copy_backward(src.begin(), src.end(), dst.end());
        assert(out == dst.begin());
        for (size_t i = 0; i < src.size(); ++i) assert(src[i] == dst[i]);
    }

    {
        std::list<int> src = { 7,8,9 };
        std::list<int> dst(3);
        auto out = simd_stl::algorithm::copy_backward(src.begin(), src.end(), dst.end());
        auto itS = src.begin();
        auto itD = dst.begin();
        for (; itS != src.end(); ++itS, ++itD) assert(*itS == *itD);
    }

    return 0;
}
