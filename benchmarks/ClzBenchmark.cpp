#include <simd_stl/math/BitMath.h>
#include <benchmarks/tools/BenchmarkHelper.h>

#include <simd_stl/compatibility/SimdCompatibility.h>

class BsfInstructionForClzBenchmark {
public:
    static void Clz(benchmark::State& state) noexcept {
        static constexpr unsigned int value = 0b00000000000000000000000000000001;
        unsigned long index = 0;

        while (state.KeepRunning()) {
#if defined(simd_stl_cpp_gnu) || defined(simd_stl_cpp_clang)
            __asm__ volatile(
                "bsr %1, %0"
                : "=r"(index)
                : "r"(value)
                : );
#elif defined(simd_stl_cpp_msvc)
            _BitScanReverse(&index, value);
#endif
            benchmark::DoNotOptimize(index);
        }
    }
};

class LzcntInstructionForClzBenchmark {
public:
    static void Clz(benchmark::State& state) noexcept {
        static constexpr unsigned int value = 0b00000000000000000000000000000001;
        uint32_t index = 0;

        while (state.KeepRunning()) {
#if defined(simd_stl_cpp_gnu) || defined(simd_stl_cpp_clang)
            __asm__ volatile("lzcnt %1, %0"
                : "=r"(index)
                : "r"(value)
                :
                );
#elif defined(simd_stl_cpp_msvc)
            index = static_cast<uint32_t>(_lzcnt_u32(value));
#endif
            benchmark::DoNotOptimize(index);
        }
    }
};


class SimdStlClzBenchmark {
public:
    static void Clz(benchmark::State& state) noexcept {
        static constexpr unsigned int value = 0b00000000000000000000000000000001;

        while (state.KeepRunning())
            benchmark::DoNotOptimize(simd_stl::math::CountLeadingZeroBits(value));
    }
};

SIMD_STL_ADD_BENCHMARK_WITH_CUSTOM_REPITITIONS(SimdStlClzBenchmark::Clz, BsfInstructionForClzBenchmark::Clz, 100000);
SIMD_STL_ADD_BENCHMARK_WITH_CUSTOM_REPITITIONS(SimdStlClzBenchmark::Clz, LzcntInstructionForClzBenchmark::Clz, 100000);


SIMD_STL_BENCHMARK_MAIN();