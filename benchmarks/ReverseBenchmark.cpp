#include <simd_stl/algorithm/order/Reverse.h>
#include <algorithm>
#include <benchmarks/tools/BenchmarkHelper.h>

template <
    typename _Char_,
    SizeForBenchmark sizeForBenchmark = SizeForBenchmark::Large>
class StdReverseBenchmark {
public:
    static void Reverse(benchmark::State& state) noexcept {
        static constexpr FixedArray<_Char_, sizeForBenchmark> textArray = FixedArray<_Char_, sizeForBenchmark>{};

        while (state.KeepRunning()) {
            std::reverse(textArray.data, textArray.data + sizeForBenchmark);
            benchmark::DoNotOptimize(textArray.data);
        }
    }
};

template <
    typename _Char_,
    SizeForBenchmark sizeForBenchmark = SizeForBenchmark::Large>
class SimdStlReverseBenchmark {
public:
    static void Reverse(benchmark::State& state) noexcept {
        static constexpr FixedArray<_Char_, sizeForBenchmark> textArray = FixedArray<_Char_, sizeForBenchmark>{};

        while (state.KeepRunning()) {
            simd_stl::algorithm::reverse(textArray.data, textArray.data + sizeForBenchmark);
            benchmark::DoNotOptimize(textArray.data);
        }
    }
};

SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlReverseBenchmark, StdReverseBenchmark, simd_stl::int8, Reverse);

SIMD_STL_BENCHMARK_MAIN();