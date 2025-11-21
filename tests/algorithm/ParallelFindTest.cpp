#include <simd_stl/concurrency/algorithm/find/ParallelFind.h>

void parallelFindTest() {
    std::vector<int> v{ 1, 2, 3, 4, 5 };

    {
        auto it = simd_stl::algorithm::find(simd_stl::concurrency::execution::sequenced,
            v.begin(), v.end(), 3);
        assert(it != v.end());
        assert(*it == 3);
    }
    {
        auto it = simd_stl::algorithm::find(simd_stl::concurrency::execution::sequenced,
            v.begin(), v.end(), 99);
        assert(it == v.end());
    }

    {
        auto it = simd_stl::algorithm::find(simd_stl::concurrency::execution::parallel,
            v.begin(), v.end(), 1);
        assert(it == v.begin());
        assert(*it == 1);
    }
    //{
    //    auto it = simd_stl::algorithm::find(simd_stl::concurrency::execution::parallel,
    //        v.begin(), v.end(), 5);
    //    assert(it != v.end());
    //    assert(*it == 5);
    //}

    //{
    //    auto it = simd_stl::algorithm::find(simd_stl::concurrency::execution::parallel_unsequenced,
    //        v.begin(), v.end(), 2);
    //    assert(it != v.end());
    //    assert(*it == 2);
    //}
    //{
    //    auto it = simd_stl::algorithm::find(simd_stl::concurrency::execution::parallel_unsequenced,
    //        v.begin(), v.end(), 42);
    //    assert(it == v.end());
    //}

    //{
    //    auto it = simd_stl::algorithm::find(simd_stl::concurrency::execution::unsequenced,
    //        v.begin(), v.end(), 4);
    //    assert(it != v.end());
    //    assert(*it == 4);
    //}
    //{
    //    auto it = simd_stl::algorithm::find(simd_stl::concurrency::execution::unsequenced,
    //        v.begin(), v.end(), -1);
    //    assert(it == v.end());
    //}

    //{
    //    std::vector<int> empty;
    //    auto it = simd_stl::algorithm::find(simd_stl::concurrency::execution::sequenced,
    //        empty.begin(), empty.end(), 1);
    //    assert(it == empty.end());
    //}
}

int main() {
	parallelFindTest();
	return 0;
}