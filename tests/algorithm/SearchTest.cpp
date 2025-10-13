#include <simd_stl/algorithm/find/Search.h>
#include <algorithm>

void testSearch() {
#define ARRAY_SIZE(arr) (sizeof(arr) / sizeof(arr[0]))

    int haystack1[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20 };
    int needle1[] = { 1, 2, 3, 4 };
    auto result1 = simd_stl::algorithm::search(haystack1, haystack1 + ARRAY_SIZE(haystack1), needle1, needle1 + ARRAY_SIZE(needle1));
    Assert(result1 == haystack1);

    int haystack2[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20 };
    int needle2[] = { 9, 10, 11, 12 };
    auto result2 = simd_stl::algorithm::search(haystack2, haystack2 + ARRAY_SIZE(haystack2), needle2, needle2 + ARRAY_SIZE(needle2));
    Assert(result2 == haystack2 + 8);

    int haystack3[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20 };
    int needle3[] = { 17, 18, 19, 20 };
    auto result3 = simd_stl::algorithm::search(haystack3, haystack3 + ARRAY_SIZE(haystack3), needle3, needle3 + ARRAY_SIZE(needle3));
    Assert(result3 == haystack3 + 16);

    int haystack4[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19 };
    int needle4[] = { 5, 6, 7, 8 };
    auto result4 = simd_stl::algorithm::search(haystack4 + 1, haystack4 + ARRAY_SIZE(haystack4), needle4, needle4 + ARRAY_SIZE(needle4));
    Assert(result4 == haystack4 + 5);

    unsigned long long haystack5[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19 };
    unsigned long long needle5[] = { 6, 7, 8 };
    auto result5 = simd_stl::algorithm::search(haystack5 + 2, haystack5 + 20, needle5, needle5 + 3);
    Assert(result5 == haystack5 + 6);

    int needle6[] = { 1, 2, 3, 4 };
    int* haystack6 = new int[1020];
    for (int i = 0; i < 1020; ++i) haystack6[i] = 0;
    haystack6[1016] = 1;
    haystack6[1017] = 2;
    haystack6[1018] = 3;
    haystack6[1019] = 4;

    auto result6 = simd_stl::algorithm::search(haystack6, haystack6 + 1020, needle6, needle6 + ARRAY_SIZE(needle6));
    Assert(result6 == haystack6 + 1016);
    delete[] haystack6;

    int needle7[] = { 1, 2, 3, 4 };
    int haystack7[1024] = { 0 };

    auto result7 = simd_stl::algorithm::search(haystack7, haystack7 + ARRAY_SIZE(haystack7), needle7, needle7 + ARRAY_SIZE(needle7));
    Assert(result7 == haystack7 + ARRAY_SIZE(haystack7));

    int haystack8[] = { 1, 2, 3, 4, 5 };
    int needle8[] = { 3 };
    auto result8 = simd_stl::algorithm::search(haystack8, haystack8 + ARRAY_SIZE(haystack8), needle8, needle8 + ARRAY_SIZE(needle8));
    Assert(result8 == haystack8 + 2);

    int haystack9[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    int needle9[] = { 2, 3, 4 };
    auto result9 = simd_stl::algorithm::search(haystack9, haystack9 + ARRAY_SIZE(haystack9), needle9, needle9 + ARRAY_SIZE(needle9));
    Assert(result9 == haystack9 + 1);

    int haystack10[] = { 1, 2 };
    int needle10[] = { 1, 2 };
    auto result10 = simd_stl::algorithm::search(haystack10, haystack10 + ARRAY_SIZE(haystack10), needle10, needle10 + ARRAY_SIZE(needle10));
    Assert(result10 == haystack10);

    int haystack11[] = { 1, 0, 3, 4, 5, 0, 7, 8 };
    int needle11[] = { 3, 4 };
    auto result11 = simd_stl::algorithm::search(haystack11, haystack11 + ARRAY_SIZE(haystack11), needle11, needle11 + ARRAY_SIZE(needle11));
    Assert(result11 == haystack11 + 2);
#undef ARRAY_SIZE
}

int main() {
	testSearch();
	return 0;
}