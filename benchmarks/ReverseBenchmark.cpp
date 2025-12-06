#include <simd_stl/algorithm/order/Reverse.h>
#include <algorithm>
#include <benchmarks/tools/BenchmarkHelper.h>

template <
    typename _Char_,
    SizeForBenchmark sizeForBenchmark>
class StdReverseBenchmark {
public:
    static inline auto array = FixedArray<_Char_, sizeForBenchmark>{};

    static void Reverse(benchmark::State& state) noexcept {
        while (state.KeepRunning()) {
            std::reverse(array.data, array.data + sizeForBenchmark);
            benchmark::DoNotOptimize(array.data);
        }
    }
};

template <
    typename _Char_,
    SizeForBenchmark sizeForBenchmark>
class SimdStlReverseBenchmark {
public:
    static inline auto array = FixedArray<_Char_, sizeForBenchmark>{};

    static void Reverse(benchmark::State& state) noexcept {

        while (state.KeepRunning()) {
            simd_stl::algorithm::reverse(array.data, array.data + sizeForBenchmark);
            benchmark::DoNotOptimize(array.data);
        }
    }
};

SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlReverseBenchmark, StdReverseBenchmark, simd_stl::int8, Reverse);

SIMD_STL_BENCHMARK_MAIN();