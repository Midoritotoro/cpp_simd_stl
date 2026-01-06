#include <simd_stl/concurrency/algorithm/find/ParallelFind.h>

void parallelFindTest() {
    std::vector<int> v{ 1, 2, 3, 4, 5 };

    {
        auto it = simd_stl::algorithm::find(simd_stl::concurrency::execution::sequenced,
            v.begin(), v.end(), 3);
        simd_stl_assert(it != v.end());
        simd_stl_assert(*it == 3);
    }
    {
        auto it = simd_stl::algorithm::find(simd_stl::concurrency::execution::sequenced,
            v.begin(), v.end(), 99);
        simd_stl_assert(it == v.end());
    }

    {
        auto it = simd_stl::algorithm::find(simd_stl::concurrency::execution::parallel,
            v.begin(), v.end(), 1);
        simd_stl_assert(it == v.begin());
        simd_stl_assert(*it == 1);
    }
    //{
    //    auto it = simd_stl::algorithm::find(simd_stl::concurrency::execution::parallel,
    //        v.begin(), v.end(), 5);
    //    simd_stl_assert(it != v.end());
    //    simd_stl_assert(*it == 5);
    //}

    //{
    //    auto it = simd_stl::algorithm::find(simd_stl::concurrency::execution::parallel_unsequenced,
    //        v.begin(), v.end(), 2);
    //    simd_stl_assert(it != v.end());
    //    simd_stl_assert(*it == 2);
    //}
    //{
    //    auto it = simd_stl::algorithm::find(simd_stl::concurrency::execution::parallel_unsequenced,
    //        v.begin(), v.end(), 42);
    //    simd_stl_assert(it == v.end());
    //}

    //{
    //    auto it = simd_stl::algorithm::find(simd_stl::concurrency::execution::unsequenced,
    //        v.begin(), v.end(), 4);
    //    simd_stl_assert(it != v.end());
    //    simd_stl_assert(*it == 4);
    //}
    //{
    //    auto it = simd_stl::algorithm::find(simd_stl::concurrency::execution::unsequenced,
    //        v.begin(), v.end(), -1);
    //    simd_stl_assert(it == v.end());
    //}

    //{
    //    std::vector<int> empty;
    //    auto it = simd_stl::algorithm::find(simd_stl::concurrency::execution::sequenced,
    //        empty.begin(), empty.end(), 1);
    //    simd_stl_assert(it == empty.end());
    //}
}

int main() {
	parallelFindTest();
	return 0;
}