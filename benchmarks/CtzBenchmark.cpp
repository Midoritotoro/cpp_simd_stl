#include <simd_stl/math/BitMath.h>
#include <benchmarks/tools/BenchmarkHelper.h>

#include <simd_stl/compatibility/SimdCompatibility.h>

class BsfInstructionForCtzBenchmark {
public:
    static void Ctz(benchmark::State& state) noexcept {
        static constexpr unsigned int value = 0b10000000000000000000000000000000;
        unsigned long index = 0;

        while (state.KeepRunning()) {
            _BitScanForward(&index, value);
            benchmark::DoNotOptimize(index);
        }
    }
};

class TzcntInstructionForCtzBenchmark {
public:
    static void Ctz(benchmark::State& state) noexcept {
        static constexpr unsigned int value = 0b10000000000000000000000000000000;

        while (state.KeepRunning()) {
            benchmark::DoNotOptimize(_tzcnt_u32(value));
        }
    }
};


class SimdStlCtzBenchmark {
public:
    static void Ctz(benchmark::State& state) noexcept {
        static constexpr unsigned int value = 0b10000000000000000000000000000000;

        while (state.KeepRunning()) {
            benchmark::DoNotOptimize(simd_stl::math::CountTrailingZeroBits(value));
        }
    }
};

class StdCtzBenchmark {
public:
    static void Ctz(benchmark::State& state) noexcept {
        static constexpr unsigned int value = 0b10000000000000000000000000000000;

        while (state.KeepRunning()) {
            benchmark::DoNotOptimize(std::countr_zero(value));
        }
    }
};

//SIMD_STL_ADD_BENCHMARK_WITH_CUSTOM_REPITITIONS(SimdStlCtzBenchmark::Ctz, BsfInstructionForCtzBenchmark::Ctz, 1000000);
//SIMD_STL_ADD_BENCHMARK_WITH_CUSTOM_REPITITIONS(SimdStlCtzBenchmark::Ctz, TzcntInstructionForCtzBenchmark::Ctz, 1000000);
SIMD_STL_ADD_BENCHMARK_WITH_CUSTOM_REPITITIONS(SimdStlCtzBenchmark::Ctz, StdCtzBenchmark::Ctz, 1000000);

SIMD_STL_BENCHMARK_MAIN();