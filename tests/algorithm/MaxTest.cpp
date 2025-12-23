#include <cAssert>
#include <cstdint>
#include <vector>
#include <list>
#include <deque>
#include <limits>
#include <optional>
#include <type_traits>
#include <simd_stl/algorithm/minmax/Max.h>

template <class T>
T ref_max_scalar(T a, T b) { return (a < b) ? b : a; }

template <class It>
std::optional<typename std::iterator_traits<It>::value_type>
ref_max_range(It first, It last) {
    using V = typename std::iterator_traits<It>::value_type;
    if (first == last) return {};
    V best = *first;
    for (++first; first != last; ++first)
        if (best < *first) best = *first;
    return best;
}

template <class T>
std::vector<T> make_data(std::size_t bytes) {
    std::size_t n = std::max<std::size_t>(1, bytes / sizeof(T));
    std::vector<T> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = T(i % 97);
    return v;
}

template <class Cont, class T>
void test_container(std::size_t bytes) {
    using simd_stl::algorithm::max_range;
    Cont c;
    auto v = make_data<T>(bytes);
    c.insert(c.end(), v.begin(), v.end());
    auto got = max_range<typename Cont::iterator, T>(c.begin(), c.end());
    auto ref = ref_max_range(c.begin(), c.end());
    Assert(got.has_value() && ref.has_value());
    Assert(*got == *ref);
}

template <class T>
void run_tests_for_type() {
    using simd_stl::algorithm::max;

    Assert(max<T>(1, 2) == ref_max_scalar<T>(1, 2));
    Assert(max<T>(2, 1) == ref_max_scalar<T>(2, 1));
    Assert(max<T>(0, 0) == ref_max_scalar<T>(0, 0));
    Assert(max<T>(std::numeric_limits<T>::min(),
        std::numeric_limits<T>::max())
        == ref_max_scalar<T>(std::numeric_limits<T>::min(),
            std::numeric_limits<T>::max()));


    for (std::size_t sz : { 0u, 16u, 64u, 512u, 4000u}) {
        test_container<std::vector<T>, T>(sz);
        test_container<std::list<T>, T>(sz);
        test_container<std::deque<T>, T>(sz);
    }
}

int main() {
    run_tests_for_type<int8_t>();
    run_tests_for_type<uint8_t>();
    run_tests_for_type<int16_t>();
    run_tests_for_type<uint16_t>();
    run_tests_for_type<int32_t>();
    run_tests_for_type<uint32_t>();
    run_tests_for_type<int64_t>();
    run_tests_for_type<uint64_t>();

    return 0;
}
