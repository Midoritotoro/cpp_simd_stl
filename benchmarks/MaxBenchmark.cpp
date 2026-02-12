#include <simd_stl/algorithm/minmax/Max.h>
#include <benchmarks/tools/BenchmarkHelper.h>

#include <ranges>
#include <utility>

template <
    typename _Type_,
    SizeForBenchmark sizeForBenchmark>
class StdMaxBenchmark {
public:
    static void MaxRange(benchmark::State& state) noexcept {
        _Type_ array[sizeForBenchmark];
        for (int i = 0; i < sizeForBenchmark; ++i)
            array[i] = i;

        while (state.KeepRunning()) {
            auto result = std::ranges::max(array);
            benchmark::DoNotOptimize(result);
            benchmark::ClobberMemory();
        }
    }
};

template <
    typename _Type_,
    SizeForBenchmark sizeForBenchmark>
class SimdStlMaxBenchmark {
public:
    static void MaxRange(benchmark::State& state) noexcept {
        _Type_ array[sizeForBenchmark];
        for (int i = 0; i < sizeForBenchmark; ++i)
            array[i] = i;

        while (state.KeepRunning()) {
            auto result = simd_stl::algorithm::max_range(array, array + sizeForBenchmark);
            benchmark::DoNotOptimize(result);
            benchmark::ClobberMemory();
        }
    }
};

SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlMaxBenchmark, StdMaxBenchmark, simd_stl::uint8, MaxRange);
SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlMaxBenchmark, StdMaxBenchmark, simd_stl::uint16, MaxRange);
SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlMaxBenchmark, StdMaxBenchmark, simd_stl::uint32, MaxRange);
SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlMaxBenchmark, StdMaxBenchmark, simd_stl::uint64, MaxRange);
SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlMaxBenchmark, StdMaxBenchmark, float, MaxRange);
SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlMaxBenchmark, StdMaxBenchmark, double, MaxRange);

SIMD_STL_BENCHMARK_MAIN();
