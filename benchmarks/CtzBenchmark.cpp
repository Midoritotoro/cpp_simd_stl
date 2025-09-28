#include <simd_stl/math/BitMath.h>
#include <benchmarks/tools/BenchmarkHelper.h>

#include <simd_stl/compatibility/SimdCompatibility.h>

class BsfInstructionForCtzBenchmark {
public:
    static void Ctz(benchmark::State& state) noexcept {
        static constexpr unsigned int value = 0b10000000000000000000000000000000;
        unsigned long index = 0;

        while (state.KeepRunning()) {
#if defined(simd_stl_cpp_gnu) || defined(simd_stl_cpp_clang)
            __asm__ volatile(
                "bsf %1, %0"
                : "=r"(index)
                : "r"(value)
            :);
#elif defined(simd_stl_cpp_msvc)
            _BitScanForward(&index, value);
#endif
            benchmark::DoNotOptimize(index);
        }
    }
};

class TzcntInstructionForCtzBenchmark {
public:
    static void Ctz(benchmark::State& state) noexcept {
        static constexpr unsigned int value = 0b10000000000000000000000000000000;
        uint32_t index = 0;

        while (state.KeepRunning()) {
#if defined(simd_stl_cpp_gnu) || defined(simd_stl_cpp_clang)
            __asm__ volatile("tzcnt %1, %0"
                : "=r"(index)
                : "r"(value)
                :
            );
#elif defined(simd_stl_cpp_msvc)
            index = static_cast<uint32_t>(_tzcnt_u32(value));
#endif
            benchmark::DoNotOptimize(index);
        }
    }
};


class SimdStlCtzBenchmark {
public:
    static void Ctz(benchmark::State& state) noexcept {
        static constexpr unsigned int value = 0b10000000000000000000000000000000;
        uint32_t index = 0;

        while (state.KeepRunning()) {
            // benchmark::DoNotOptimize(index = static_cast<uint32_t>(simd_stl::math::CountTrailingZeroBits(value)));
            index = static_cast<uint32_t>(simd_stl::math::CountTrailingZeroBits(value));
            benchmark::DoNotOptimize(index);
        }
    }
};

SIMD_STL_ADD_BENCHMARK_WITH_CUSTOM_REPITITIONS(SimdStlCtzBenchmark::Ctz, BsfInstructionForCtzBenchmark::Ctz, 100000);
SIMD_STL_ADD_BENCHMARK_WITH_CUSTOM_REPITITIONS(SimdStlCtzBenchmark::Ctz, TzcntInstructionForCtzBenchmark::Ctz, 100000);


SIMD_STL_BENCHMARK_MAIN();