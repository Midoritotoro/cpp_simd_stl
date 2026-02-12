#include <simd_stl/algorithm/minmax/Min.h>
#include <benchmarks/tools/BenchmarkHelper.h>

#include <ranges>
#include <utility>

template <
    typename _Type_,
    SizeForBenchmark sizeForBenchmark>
class StdMinBenchmark {
public:
    static void MinRange(benchmark::State& state) noexcept {
        _Type_ array[sizeForBenchmark];
        for (int i = 0; i < sizeForBenchmark; ++i)
            array[i] = i;

        while (state.KeepRunning()) {
            auto result = std::ranges::min(array);
            benchmark::DoNotOptimize(result);
            benchmark::ClobberMemory();
        }
    }
};

template <
    typename _Type_,
    SizeForBenchmark sizeForBenchmark>
class SimdStlMinBenchmark {
public:
    static void MinRange(benchmark::State& state) noexcept {
        _Type_ array[sizeForBenchmark];
        for (int i = 0; i < sizeForBenchmark; ++i)
            array[i] = i;

        while (state.KeepRunning()) {
            auto result = simd_stl::algorithm::min_range(array, array + sizeForBenchmark);
            benchmark::DoNotOptimize(result);
            benchmark::ClobberMemory();
        }
    }
};

SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlMinBenchmark, StdMinBenchmark, simd_stl::uint8, MinRange);
SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlMinBenchmark, StdMinBenchmark, simd_stl::uint16, MinRange);
SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlMinBenchmark, StdMinBenchmark, simd_stl::uint32, MinRange);
SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlMinBenchmark, StdMinBenchmark, simd_stl::uint64, MinRange);
SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlMinBenchmark, StdMinBenchmark, float, MinRange);
SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlMinBenchmark, StdMinBenchmark, double, MinRange);

SIMD_STL_BENCHMARK_MAIN();
