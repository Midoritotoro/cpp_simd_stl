#include <simd_stl/math/BitMath.h>
#include <benchmarks/tools/BenchmarkHelper.h>

#include <simd_stl/compatibility/SimdCompatibility.h>

class BsrInstructionForClzBenchmark {
public:
    static void Clz(benchmark::State& state) noexcept {
        static constexpr unsigned int value = 0b00000000000000000000000000000001;
        unsigned long index = 0;

        while (state.KeepRunning()) {
            _BitScanReverse(&index, value);
            benchmark::DoNotOptimize(index);
        }
    }
};

class LzcntInstructionForClzBenchmark {
public:
    static void Clz(benchmark::State& state) noexcept {
        static constexpr unsigned int value = 0b00000000000000000000000000000001;

        while (state.KeepRunning()) {
            benchmark::DoNotOptimize(_lzcnt_u32(value));
        }
    }
};


class SimdStlClzBenchmark {
public:
    static void Clz(benchmark::State& state) noexcept {
        static constexpr unsigned int value = 0b00000000000000000000000000000001;

        while (state.KeepRunning())
            benchmark::DoNotOptimize(simd_stl::math::count_leading_zero_bits(value));
    }
};

class StdClzBenchmark {
public:
    static void Clz(benchmark::State& state) noexcept {
        static constexpr unsigned int value = 0b00000000000000000000000000000001;

        while (state.KeepRunning())
            benchmark::DoNotOptimize(std::countl_zero(value));
    }
};

SIMD_STL_ADD_BENCHMARK_WITH_CUSTOM_REPITITIONS(SimdStlClzBenchmark::Clz, BsrInstructionForClzBenchmark::Clz, 1000000);
SIMD_STL_ADD_BENCHMARK_WITH_CUSTOM_REPITITIONS(SimdStlClzBenchmark::Clz, LzcntInstructionForClzBenchmark::Clz, 1000000);
SIMD_STL_ADD_BENCHMARK_WITH_CUSTOM_REPITITIONS(SimdStlClzBenchmark::Clz, StdClzBenchmark::Clz, 1000000);

SIMD_STL_BENCHMARK_MAIN();