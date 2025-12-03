#include <simd_stl/algorithm/find/Find.h>
#include <benchmarks/tools/BenchmarkHelper.h>

#include <uchar.h>
#include <wchar.h>

template <
    typename _Char_,
    SizeForBenchmark sizeForBenchmark = SizeForBenchmark::Large>
class StdFindBenchmark {
public:
    static void Find(benchmark::State& state) noexcept {
        static constexpr auto textArray = FixedArray<_Char_, sizeForBenchmark>{};

        while (state.KeepRunning())
            benchmark::DoNotOptimize(std::find(textArray.data, textArray.data + sizeForBenchmark, sizeForBenchmark - 1));
    }
};

template <
    typename _Char_,
    SizeForBenchmark sizeForBenchmark = SizeForBenchmark::Large>
class SimdStlFindBenchmark {
public:
    static void Find(benchmark::State& state) noexcept {
        static constexpr auto textArray = FixedArray<_Char_, sizeForBenchmark>{};

        while (state.KeepRunning())
            benchmark::DoNotOptimize(simd_stl::algorithm::find(textArray.data, textArray.data + sizeForBenchmark, sizeForBenchmark - 1));
    }
};

SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(SimdStlFindBenchmark, StdFindBenchmark, simd_stl::int16, Find);

SIMD_STL_BENCHMARK_MAIN();