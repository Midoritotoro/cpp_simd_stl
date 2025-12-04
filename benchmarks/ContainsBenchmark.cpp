#include <simd_stl/algorithm/find/Contains.h>
#include <benchmarks/tools/BenchmarkHelper.h>

#include <ranges>

template <
    typename _Char_,
    SizeForBenchmark sizeForBenchmark = SizeForBenchmark::Large>
class StdFindBenchmark {
public:
    static void Find(benchmark::State& state) noexcept {
        static constexpr auto textArray = FixedArray<_Char_, sizeForBenchmark>{};

        while (state.KeepRunning()) {
            benchmark::DoNotOptimize(std::ranges::contains(textArray.data, textArray.data + sizeForBenchmark, sizeForBenchmark - 1));
        }
    }
};

template <
    typename _Char_,
    SizeForBenchmark sizeForBenchmark = SizeForBenchmark::Large>
class SimdStlFindBenchmark {
public:
    static void Find(benchmark::State& state) noexcept {
        static constexpr auto textArray = FixedArray<_Char_, sizeForBenchmark>{};

        while (state.KeepRunning()) {
            benchmark::DoNotOptimize(simd_stl::algorithm::contains(textArray.data, textArray.data + sizeForBenchmark, sizeForBenchmark - 1));
        }
    }
};

SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlFindBenchmark, StdFindBenchmark, long long, Find);

SIMD_STL_BENCHMARK_MAIN();