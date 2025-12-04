#pragma once 

#include <simd_stl/compatibility/Compatibility.h>
#include <src/simd_stl/utility/Assert.h>

#if defined(min)
#  undef min
#endif // defined(min)

#if defined(max)
#  undef max
#endif // defined(max)

#include <benchmark/benchmark.h>

#include <iostream>
#include <map>

#include <sstream>
#include <array>

#include <iomanip>

#if !defined(SIMD_STL_BENCHMARK_REPITITIONS)
#  define SIMD_STL_BENCHMARK_REPITITIONS 1000000
#endif // SIMD_STL_BENCHMARK_REPITITIONS

#if !defined(SIMD_STL_BENCHMARK_ITERATIONS)
#  define SIMD_STL_BENCHMARK_ITERATIONS 1
#endif // SIMD_STL_BENCHMARK_ITERATIONS

#if defined(SIMD_STL_BENCHMARK_IN_MILLISECONDS)
#  define SIMD_STL_BENCHMARK_UNIT_OF_MEASUREMENT benchmark::kMillisecond
#else 
#  define SIMD_STL_BENCHMARK_UNIT_OF_MEASUREMENT benchmark::kNanosecond
#endif // SIMD_STL_BENCHMARK_UNIT_OF_MEASUREMENT

enum SizeForBenchmark : simd_stl::uint32 {
    Tiny        = 16,       
    VerySmall   = 32,   
    Small       = 64,    
    MediumSmall = 128,
    Medium      = 256,
    MediumLarge = 512,
    Large       = 1024,
    VeryLarge   = 2048,
    Huge        = 4096,
    ExtraHuge   = 8192,
    MegaHuge    = 16384,
    GigaHuge    = 32768,
    TeraHuge    = 65536  
};

#define SIMD_STL_FIXED_INTEGER_ARRAY(name) \
    template <typename _Type_, size_t N>\
    struct name {\
        _Type_ data[N + 1]{};\
    \
        constexpr name() {\
            for (size_t i = 0; i < N; ++i) {\
                data[i] = i;\
            }\
        }\
    }

#define SIMD_STL_FIXED_CHAR_ARRAY(name, prefix) \
    template <typename _Type_, size_t N>\
    struct name {\
        _Type_ data[N + 1]{};\
    \
        constexpr name() {\
            for (size_t i = 0; i < N; ++i) {\
                data[i] = prefix'A' + (i % 26);\
            }\
            data[N] = prefix'\0';\
        }\
    };

#define SIMD_STL_FIXED_REVERSED_CHAR_ARRAY(name, prefix) \
    template <typename _Type_, size_t N>\
    struct name {\
        _Type_ data[N + 1]{};\
    \
        constexpr name() {\
            for (size_t i = 0; i < N; ++i) {\
                data[i] = prefix'Z' - (i % 26);\
            }\
            data[N] = prefix'\0';\
        }\
    };

#define SIMD_STL_ADD_SPECIALIZATION_TO_FIXED_CHAR_ARRAY(name, type, prefix) \
    template <size_t N>\
    struct name<type, N> {\
        type data[N + 1]{};\
    \
        constexpr name() {\
            for (size_t i = 0; i < N; ++i) {\
                data[i] = prefix'A' + (i % 26);\
            }\
            data[N] = prefix'\0';\
        }\
    };

#define SIMD_STL_ADD_SPECIALIZATION_TO_FIXED_INTEGER_ARRAY(name, type) \
    template <size_t N>\
    struct name<type, N> {\
        type data[N + 1]{};\
    \
        constexpr name() {\
            for (size_t i = 0; i < N; ++i) {\
                data[i] = i;\
            }\
        }\
    }

#define SIMD_STL_ADD_SPECIALIZATION_TO_FIXED_REVERSED_CHAR_ARRAY(name, type, prefix) \
    template <size_t N>\
    struct name<type, N> {\
        type data[N + 1]{};\
    \
        constexpr name() {\
            for (size_t i = 0; i < N; ++i) {\
                data[i] = prefix'Z' - (i % 26);\
            }\
            data[N] = prefix'\0';\
        }\
    };


SIMD_STL_FIXED_CHAR_ARRAY(FixedArray, );
#if __cpp_lib_char8_t
SIMD_STL_ADD_SPECIALIZATION_TO_FIXED_CHAR_ARRAY(FixedArray, char8_t, u8);
#endif
SIMD_STL_ADD_SPECIALIZATION_TO_FIXED_CHAR_ARRAY(FixedArray, char16_t, u);
SIMD_STL_ADD_SPECIALIZATION_TO_FIXED_CHAR_ARRAY(FixedArray, char32_t, U);
SIMD_STL_ADD_SPECIALIZATION_TO_FIXED_CHAR_ARRAY(FixedArray, wchar_t, L);



SIMD_STL_FIXED_REVERSED_CHAR_ARRAY(FixedReversedArray, );
#if __cpp_lib_char8_t
SIMD_STL_ADD_SPECIALIZATION_TO_FIXED_REVERSED_CHAR_ARRAY(FixedReversedArray, char8_t, u8);
#endif
SIMD_STL_ADD_SPECIALIZATION_TO_FIXED_REVERSED_CHAR_ARRAY(FixedReversedArray, char16_t, u);
SIMD_STL_ADD_SPECIALIZATION_TO_FIXED_REVERSED_CHAR_ARRAY(FixedReversedArray, char32_t, U);
SIMD_STL_ADD_SPECIALIZATION_TO_FIXED_REVERSED_CHAR_ARRAY(FixedReversedArray, wchar_t, L);


SIMD_STL_FIXED_INTEGER_ARRAY(FixedIntegerArray);
SIMD_STL_ADD_SPECIALIZATION_TO_FIXED_INTEGER_ARRAY(FixedIntegerArray, simd_stl::uint8);
SIMD_STL_ADD_SPECIALIZATION_TO_FIXED_INTEGER_ARRAY(FixedIntegerArray, simd_stl::uint16);
SIMD_STL_ADD_SPECIALIZATION_TO_FIXED_INTEGER_ARRAY(FixedIntegerArray, simd_stl::uint32);
SIMD_STL_ADD_SPECIALIZATION_TO_FIXED_INTEGER_ARRAY(FixedIntegerArray, simd_stl::uint64);

#if !defined(SIMD_STL_ADD_BENCHMARK_WITH_CUSTOM_REPITITIONS)
#  define SIMD_STL_ADD_BENCHMARK_WITH_CUSTOM_REPITITIONS(benchFirst, benchSecond, repititions)                   \
     BENCHMARK(benchFirst)->Unit(SIMD_STL_BENCHMARK_UNIT_OF_MEASUREMENT)    \
        ->Iterations(SIMD_STL_BENCHMARK_ITERATIONS)                         \
        ->Repetitions(repititions)                       \
        ->ReportAggregatesOnly(true)                                    \
        ->DisplayAggregatesOnly(true);                                  \
    BENCHMARK(benchSecond)->Unit(SIMD_STL_BENCHMARK_UNIT_OF_MEASUREMENT)    \
        ->Iterations(SIMD_STL_BENCHMARK_ITERATIONS)                         \
        ->Repetitions(repititions)                       \
        ->ReportAggregatesOnly(true)                                    \
        ->DisplayAggregatesOnly(true);                                   
#endif // SIMD_STL_ADD_BENCHMARK_WITH_CUSTOM_REPITITIONS



#if !defined(SIMD_STL_ADD_BENCHMARK)
#  define SIMD_STL_ADD_BENCHMARK(benchFirst, benchSecond)                   \
     BENCHMARK(benchFirst)->Unit(SIMD_STL_BENCHMARK_UNIT_OF_MEASUREMENT)    \
        ->Iterations(SIMD_STL_BENCHMARK_ITERATIONS)                         \
        ->Repetitions(SIMD_STL_BENCHMARK_REPITITIONS)                       \
        ->ReportAggregatesOnly(true)                                    \
        ->DisplayAggregatesOnly(true);                                  \
    BENCHMARK(benchSecond)->Unit(SIMD_STL_BENCHMARK_UNIT_OF_MEASUREMENT)    \
        ->Iterations(SIMD_STL_BENCHMARK_ITERATIONS)                         \
        ->Repetitions(SIMD_STL_BENCHMARK_REPITITIONS)                       \
        ->ReportAggregatesOnly(true)                                    \
        ->DisplayAggregatesOnly(true);                                   
#endif // SIMD_STL_ADD_BENCHMARK

#if !defined(SIMD_STL_ADD_BENCHMARK_ARGS)
#  define SIMD_STL_ADD_BENCHMARK_ARGS(benchFirst, benchSecond, ...)         \
     BENCHMARK(benchFirst)->Unit(SIMD_STL_BENCHMARK_UNIT_OF_MEASUREMENT)    \
        ->Iterations(SIMD_STL_BENCHMARK_ITERATIONS)                         \
        ->Repetitions(SIMD_STL_BENCHMARK_REPITITIONS)                       \
        ->ReportAggregatesOnly(true)                                    \
        ->DisplayAggregatesOnly(true)                                   \
        ->Args({__VA_ARGS__});                                          \
    BENCHMARK(benchSecond)->Unit(SIMD_STL_BENCHMARK_UNIT_OF_MEASUREMENT)    \
        ->Iterations(SIMD_STL_BENCHMARK_ITERATIONS)                         \
        ->Repetitions(SIMD_STL_BENCHMARK_REPITITIONS)                       \
        ->ReportAggregatesOnly(true)                                    \
        ->DisplayAggregatesOnly(true)                                   \
        ->Args({__VA_ARGS__});                                            
#endif // SIMD_STL_ADD_BENCHMARK_ARGS


#if !defined(SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE)
#  define SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE(nameFirst, nameSecond, Type, funcN)                                                                                                          \
    SIMD_STL_ADD_BENCHMARK(SIMD_STL_ECHO(nameFirst<Type, SizeForBenchmark::Tiny>::funcN), SIMD_STL_ECHO(nameSecond<Type, SizeForBenchmark::Tiny>::funcN));                    \
    SIMD_STL_ADD_BENCHMARK(SIMD_STL_ECHO(nameFirst<Type, SizeForBenchmark::VerySmall>::funcN),SIMD_STL_ECHO(nameSecond<Type, SizeForBenchmark::VerySmall>::funcN));           \
    SIMD_STL_ADD_BENCHMARK(SIMD_STL_ECHO(nameFirst<Type, SizeForBenchmark::Small>::funcN), SIMD_STL_ECHO(nameSecond<Type, SizeForBenchmark::Small>::funcN));                  \
    SIMD_STL_ADD_BENCHMARK(SIMD_STL_ECHO(nameFirst<Type, SizeForBenchmark::MediumSmall>::funcN), SIMD_STL_ECHO(nameSecond<Type, SizeForBenchmark::MediumSmall>::funcN));      \
    SIMD_STL_ADD_BENCHMARK(SIMD_STL_ECHO(nameFirst<Type, SizeForBenchmark::Medium>::funcN), SIMD_STL_ECHO(nameSecond<Type, SizeForBenchmark::Medium>::funcN));                \
    SIMD_STL_ADD_BENCHMARK(SIMD_STL_ECHO(nameFirst<Type, SizeForBenchmark::MediumLarge>::funcN),SIMD_STL_ECHO(nameSecond<Type, SizeForBenchmark::MediumLarge>::funcN));       \
    SIMD_STL_ADD_BENCHMARK(SIMD_STL_ECHO(nameFirst<Type, SizeForBenchmark::Large>::funcN),SIMD_STL_ECHO(nameSecond<Type, SizeForBenchmark::Large>::funcN));                   \
    SIMD_STL_ADD_BENCHMARK(SIMD_STL_ECHO(nameFirst<Type, SizeForBenchmark::VeryLarge>::funcN),SIMD_STL_ECHO(nameSecond<Type, SizeForBenchmark::VeryLarge>::funcN));           \
    SIMD_STL_ADD_BENCHMARK(SIMD_STL_ECHO(nameFirst<Type, SizeForBenchmark::Huge>::funcN), SIMD_STL_ECHO(nameSecond<Type, SizeForBenchmark::Huge>::funcN));                    \
    SIMD_STL_ADD_BENCHMARK(SIMD_STL_ECHO(nameFirst<Type, SizeForBenchmark::ExtraHuge>::funcN), SIMD_STL_ECHO(nameSecond<Type, SizeForBenchmark::ExtraHuge>::funcN));          \
    SIMD_STL_ADD_BENCHMARK(SIMD_STL_ECHO(nameFirst<Type, SizeForBenchmark::MegaHuge>::funcN), SIMD_STL_ECHO(nameSecond<Type, SizeForBenchmark::MegaHuge>::funcN));            \
    SIMD_STL_ADD_BENCHMARK(SIMD_STL_ECHO(nameFirst<Type, SizeForBenchmark::GigaHuge>::funcN), SIMD_STL_ECHO(nameSecond<Type, SizeForBenchmark::GigaHuge>::funcN));            \
    SIMD_STL_ADD_BENCHMARK(SIMD_STL_ECHO(nameFirst<Type, SizeForBenchmark::TeraHuge>::funcN), SIMD_STL_ECHO(nameSecond<Type, SizeForBenchmark::TeraHuge>::funcN));
#endif // SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE

#if !defined(SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE_ARGS)
#  define SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE_ARGS(nameFirst, nameSecond, Type, funcN, ...)                                                                                                                \
    SIMD_STL_ADD_BENCHMARK_ARGS(SIMD_STL_ECHO(nameFirst<Type, SizeForBenchmark::Tiny>::funcN), SIMD_STL_ECHO(nameSecond<Type, SizeForBenchmark::Tiny>::funcN), __VA_ARGS__);                  \
    SIMD_STL_ADD_BENCHMARK_ARGS(SIMD_STL_ECHO(nameFirst<Type, SizeForBenchmark::VerySmall>::funcN),SIMD_STL_ECHO(nameSecond<Type, SizeForBenchmark::VerySmall>::funcN), __VA_ARGS__);         \
    SIMD_STL_ADD_BENCHMARK_ARGS(SIMD_STL_ECHO(nameFirst<Type, SizeForBenchmark::Small>::funcN), SIMD_STL_ECHO(nameSecond<Type, SizeForBenchmark::Small>::funcN), __VA_ARGS__);                \
    SIMD_STL_ADD_BENCHMARK_ARGS(SIMD_STL_ECHO(nameFirst<Type, SizeForBenchmark::MediumSmall>::funcN), SIMD_STL_ECHO(nameSecond<Type, SizeForBenchmark::MediumSmall>::funcN), __VA_ARGS__);    \
    SIMD_STL_ADD_BENCHMARK_ARGS(SIMD_STL_ECHO(nameFirst<Type, SizeForBenchmark::Medium>::funcN), SIMD_STL_ECHO(nameSecond<Type, SizeForBenchmark::Medium>::funcN), __VA_ARGS__);              \
    SIMD_STL_ADD_BENCHMARK_ARGS(SIMD_STL_ECHO(nameFirst<Type, SizeForBenchmark::MediumLarge>::funcN),SIMD_STL_ECHO(nameSecond<Type, SizeForBenchmark::MediumLarge>::funcN), __VA_ARGS__);     \
    SIMD_STL_ADD_BENCHMARK_ARGS(SIMD_STL_ECHO(nameFirst<Type, SizeForBenchmark::Large>::funcN),SIMD_STL_ECHO(nameSecond<Type, SizeForBenchmark::Large>::funcN), __VA_ARGS__);                 \
    SIMD_STL_ADD_BENCHMARK_ARGS(SIMD_STL_ECHO(nameFirst<Type, SizeForBenchmark::VeryLarge>::funcN),SIMD_STL_ECHO(nameSecond<Type, SizeForBenchmark::VeryLarge>::funcN), __VA_ARGS__);         \
    SIMD_STL_ADD_BENCHMARK_ARGS(SIMD_STL_ECHO(nameFirst<Type, SizeForBenchmark::Huge>::funcN), SIMD_STL_ECHO(nameSecond<Type, SizeForBenchmark::Huge>::funcN), __VA_ARGS__);                  \
    SIMD_STL_ADD_BENCHMARK_ARGS(SIMD_STL_ECHO(nameFirst<Type, SizeForBenchmark::ExtraHuge>::funcN), SIMD_STL_ECHO(nameSecond<Type, SizeForBenchmark::ExtraHuge>::funcN), __VA_ARGS__);        \
    SIMD_STL_ADD_BENCHMARK_ARGS(SIMD_STL_ECHO(nameFirst<Type, SizeForBenchmark::MegaHuge>::funcN), SIMD_STL_ECHO(nameSecond<Type, SizeForBenchmark::MegaHuge>::funcN), __VA_ARGS__);          \
    SIMD_STL_ADD_BENCHMARK_ARGS(SIMD_STL_ECHO(nameFirst<Type, SizeForBenchmark::GigaHuge>::funcN), SIMD_STL_ECHO(nameSecond<Type, SizeForBenchmark::GigaHuge>::funcN), __VA_ARGS__);          \
    SIMD_STL_ADD_BENCHMARK_ARGS(SIMD_STL_ECHO(nameFirst<Type, SizeForBenchmark::TeraHuge>::funcN), SIMD_STL_ECHO(nameSecond<Type, SizeForBenchmark::TeraHuge>::funcN), __VA_ARGS__);
#endif // SIMD_STL_ADD_BENCHMARKS_FOR_EACH_SIZE_ARGS

#if !defined(SIMD_STL_BENCHMARK_MAIN)
#define SIMD_STL_BENCHMARK_MAIN()                                                   \
    int main(int argc, char** argv) {                                           \
        benchmark::MaybeReenterWithoutASLR(argc, argv);                         \
        char arg0_default[] = "benchmark";                                      \
        char* args_default = reinterpret_cast<char*>(arg0_default);             \
        if (!argv) {                                                            \
            argc = 1;                                                           \
            argv = &args_default;                                               \
        }                                                                       \
        ::benchmark::Initialize(&argc, argv);                                   \
        if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;     \
        BenchmarksCompareReporter reporter;                                     \
        ::benchmark::RunSpecifiedBenchmarks(&reporter);                         \
        ::benchmark::Shutdown();                                                \
        return 0;                                                               \
    }                                                                           \
    int main(int, char**)
#endif // SIMD_STL_BENCHMARK_MAIN


static struct Loc {
    Loc() {
        std::setlocale(LC_ALL, "en_US.UTF-8");
    }
} localeSet;


enum LogColor {
    COLOR_DEFAULT,
    COLOR_RED,
    COLOR_GREEN,
    COLOR_YELLOW,
    COLOR_BLUE,
    COLOR_MAGENTA,
    COLOR_CYAN,
    COLOR_WHITE
};

#ifdef simd_stl_os_win
typedef WORD PlatformColorCode;
#else
typedef const char* PlatformColorCode;
#endif

PlatformColorCode GetPlatformColorCode(LogColor color) {
#ifdef simd_stl_os_win
    switch (color) {
    case COLOR_RED:
        return FOREGROUND_RED;
    case COLOR_GREEN:
        return FOREGROUND_GREEN;
    case COLOR_YELLOW:
        return FOREGROUND_RED | FOREGROUND_GREEN;
    case COLOR_BLUE:
        return FOREGROUND_BLUE;
    case COLOR_MAGENTA:
        return FOREGROUND_BLUE | FOREGROUND_RED;
    case COLOR_CYAN:
        return FOREGROUND_BLUE | FOREGROUND_GREEN;
    case COLOR_WHITE:  // fall through to default
    default:
        return 0;
    }
#else
    switch (color) {
    case COLOR_RED:
        return "1";
    case COLOR_GREEN:
        return "2";
    case COLOR_YELLOW:
        return "3";
    case COLOR_BLUE:
        return "4";
    case COLOR_MAGENTA:
        return "5";
    case COLOR_CYAN:
        return "6";
    case COLOR_WHITE:
        return "7";
    default:
        return nullptr;
    };
#endif
}

std::string FormatString(const char* msg, va_list args) {
    // we might need a second shot at this, so pre-emptivly make a copy
    va_list args_cp;
    va_copy(args_cp, args);

    std::size_t size = 256;
    char local_buff[256];
    auto ret = vsnprintf(local_buff, size, msg, args_cp);

    va_end(args_cp);

    // currently there is no error handling for failure, so this is hack.
    Assert(ret >= 0);

    if (ret == 0) {  // handle empty expansion
        return {};
    }
    if (static_cast<size_t>(ret) < size) {
        return local_buff;
    }
    // we did not provide a long enough buffer on our first attempt.
    size = static_cast<size_t>(ret) + 1;  // + 1 for the null byte
    std::unique_ptr<char[]> buff(new char[size]);
    ret = vsnprintf(buff.get(), size, msg, args);
    Assert(ret > 0 && (static_cast<size_t>(ret)) < size);
    return buff.get();
}

std::string FormatString(const char* msg, ...) {
    va_list args;
    va_start(args, msg);
    auto tmp = FormatString(msg, args);
    va_end(args);
    return tmp;
}


void ColorPrintf(
    std::ostream& out,
    LogColor color,
    const char* fmt,
    ...) 
{
    va_list args;
    va_start(args, fmt);

#ifdef simd_stl_os_win
    const HANDLE stdout_handle = GetStdHandle(STD_OUTPUT_HANDLE);

    // Gets the current text color.
    CONSOLE_SCREEN_BUFFER_INFO buffer_info;
    GetConsoleScreenBufferInfo(stdout_handle, &buffer_info);
    const WORD original_color_attrs = buffer_info.wAttributes;

    const WORD original_background_attrs =
        original_color_attrs & (BACKGROUND_RED | BACKGROUND_GREEN |
            BACKGROUND_BLUE | BACKGROUND_INTENSITY);

    SetConsoleTextAttribute(stdout_handle, GetPlatformColorCode(color) |
        FOREGROUND_INTENSITY |
        original_background_attrs);
    out << FormatString(fmt, args);

    out << std::flush;
    // Restores the text and background color.
    SetConsoleTextAttribute(stdout_handle, original_color_attrs);
#else
    const char* color_code = GetPlatformColorCode(color);
    if (color_code != nullptr) {
        out << FormatString("\033[0;3%sm", color_code);
    }
    out << FormatString(fmt, args) << "\033[m";
#endif
    va_end(args);
}


inline std::vector<benchmark::BenchmarkReporter::Run> _benchmarks;

class BenchmarksCompareReporter : public benchmark::ConsoleReporter
{
public:
    BenchmarksCompareReporter() {}

    bool ReportContext(const Context& context) override {
        return ConsoleReporter::ReportContext(context);
    }

    void ReportRuns(const std::vector<Run>& reports) override {
        std::ostream& out = GetOutputStream();
        ConsoleReporter::ReportRuns(reports);

        for (const auto& run : reports) {
            _benchmarks.push_back(run);

            if (_benchmarks.size() % 8 == 0) {
                const auto& firstBenchmark = _benchmarks[0];
                const auto& secondBenchmark = _benchmarks[4];

                const auto realTimeDifference = (secondBenchmark.GetAdjustedRealTime() / firstBenchmark.GetAdjustedRealTime());

                const auto firstBenchmarkFullName   = firstBenchmark.benchmark_name();
                const auto secondBenchmarkFullName  = secondBenchmark.benchmark_name();
                
                const auto firstBenchmarkPrettyName = firstBenchmarkFullName.substr(0, firstBenchmarkFullName.find('/'));
                const auto secondBenchmarkPrettyName = secondBenchmarkFullName.substr(0, secondBenchmarkFullName.find('/'));

                if (realTimeDifference > 1.0f)
                    ColorPrintf(out, COLOR_BLUE, "%s faster than %s by a %f%s\n", firstBenchmarkPrettyName.c_str(),
                        secondBenchmarkPrettyName.c_str(), std::abs(realTimeDifference * 100 - 100), "%");

                else if (realTimeDifference < 1.0f)
                    ColorPrintf(out, COLOR_RED, "%s slower than %s by a %f%s\n", firstBenchmarkPrettyName.c_str(),
                        secondBenchmarkPrettyName.c_str(), std::abs(realTimeDifference * 100 - 100), "%");

                else
                    ColorPrintf(out, COLOR_WHITE, "Benchmarks are equal\n");

                _benchmarks.resize(0);
            }
        }
    }
};