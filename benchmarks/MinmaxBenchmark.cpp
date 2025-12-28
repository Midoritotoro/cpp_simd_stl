#include <simd_stl/algorithm/minmax/Minmax.h>
#include <benchmarks/tools/BenchmarkHelper.h>

#include <ranges>
#include <utility>

template <
    typename _Type_,
    SizeForBenchmark sizeForBenchmark>
class StdMinmaxBenchmark {
public:
    static void MinmaxRange(benchmark::State& state) noexcept {
        _Type_ array[sizeForBenchmark];
        for (int i = 0; i < sizeForBenchmark; ++i)
            array[i] = i;

        while (state.KeepRunning()) {
            auto result = std::ranges::minmax(array);
            benchmark::DoNotOptimize(result);
            benchmark::ClobberMemory();
        }
    }
};

template <
    typename _Type_,
    SizeForBenchmark sizeForBenchmark>
class SimdStlMinmaxBenchmark {
public:
    static void MinmaxRange(benchmark::State& state) noexcept {
        _Type_ array[sizeForBenchmark];
        for (int i = 0; i < sizeForBenchmark; ++i)
            array[i] = i;

        while (state.KeepRunning()) {
            auto result = simd_stl::algorithm::minmax_range(array, array + sizeForBenchmark);
            benchmark::DoNotOptimize(result);
            benchmark::ClobberMemory();
        }
    }
};

SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlMinmaxBenchmark, StdMinmaxBenchmark, simd_stl::uint8, MinmaxRange);
SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlMinmaxBenchmark, StdMinmaxBenchmark, simd_stl::uint16, MinmaxRange);
SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlMinmaxBenchmark, StdMinmaxBenchmark, simd_stl::uint32, MinmaxRange);
SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlMinmaxBenchmark, StdMinmaxBenchmark, simd_stl::uint64, MinmaxRange);
SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlMinmaxBenchmark, StdMinmaxBenchmark, float, MinmaxRange);
SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlMinmaxBenchmark, StdMinmaxBenchmark, double, MinmaxRange);

SIMD_STL_BENCHMARK_MAIN();
