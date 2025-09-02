#include <simd_stl/algorithm/find/Search.h>
#include <algorithm>


void testSearch() {
    std::vector<int> haystack1 = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20 };
    std::vector<int> needle1 = { 1, 2, 3, 4 };
    auto result1 = simd_stl::algorithm::search(haystack1.begin(), haystack1.end(),
        needle1.begin(), needle1.end());
    Assert(result1 == haystack1.begin());


    std::vector<int> haystack2 = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20 };
    std::vector<int> needle2 = { 9, 10, 11, 12 };
    auto result2 = simd_stl::algorithm::search(haystack2.begin(), haystack2.end(),
        needle2.begin(), needle2.end());
    Assert(result2 == haystack2.begin() + 8);
  
    std::vector<int> haystack3 = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20 };
    std::vector<int> needle3 = { 17, 18, 19, 20 };
    auto result3 = simd_stl::algorithm::search(haystack3.begin(), haystack3.end(),
        needle3.begin(), needle3.end());
    Assert(result3 == haystack3.begin() + 16);

    std::vector<int> haystack4 = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19 };
    std::vector<int> needle4 = { 5, 6, 7, 8 };
    auto result4 = simd_stl::algorithm::search(haystack4.begin() + 1, haystack4.end(),
        needle4.begin(), needle4.end());
    Assert(result4 == haystack4.begin() + 5);

    std::vector<int> haystack5 = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19 };
    std::vector<int> needle5 = { 6, 7, 8 };
    auto result5 = simd_stl::algorithm::search(haystack5.begin() + 2, haystack5.end(),
        needle5.begin(), needle5.end()); 

    Assert(result5 == haystack5.begin() + 6);
   

    std::vector<int> haystack6(1020, 0);
    std::vector<int> needle6 = { 1, 2, 3, 4 };
    haystack6[1016] = 1;
    haystack6[1017] = 2;
    haystack6[1018] = 3;
    haystack6[1019] = 4;

    auto result6 = simd_stl::algorithm::search(haystack6.begin(), haystack6.end(), needle6.begin(), needle6.end());
    Assert(result6 == haystack6.begin() + 1016);
   
    std::vector<int> haystack7(1024, 0);
    std::vector<int> needle7 = { 1, 2, 3, 4 };
    auto result7 = simd_stl::algorithm::search(haystack7.begin(), haystack7.end(), needle7.begin(), needle7.end());
    Assert(result7 == haystack7.end());
   

    std::vector<int> haystack8 = { 1, 2, 3, 4, 5 };
    std::vector<int> needle8 = { 3 };

    auto result8 = simd_stl::algorithm::search(haystack8.begin(), haystack8.end(), needle8.begin(), needle8.end());

    Assert(result8 == haystack8.begin() + 2);
   
    std::vector<int> haystack9 = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    std::vector<int> needle9 = { 2, 3, 4 }; 

    auto result9 = simd_stl::algorithm::search(haystack9.begin(), haystack9.end(), needle9.begin(), needle9.end());

    Assert(result9 == haystack9.begin() + 1);
    
    std::vector<int> haystack10 = { 1, 2 };
    std::vector<int> needle10 = { 1, 2 };

    auto result10 = simd_stl::algorithm::search(haystack10.begin(), haystack10.end(), needle10.begin(), needle10.end());

    Assert(result10 == haystack10.begin());
  
    std::vector<int> haystack11 = { 1, 0, 3, 4, 5, 0, 7, 8 };
    std::vector<int> needle11 = { 3, 4 };

    auto result11 = simd_stl::algorithm::search(haystack11.begin(), haystack11.end(), needle11.begin(), needle11.end());

    Assert(result11 == haystack11.begin() + 2);
}


int main() {
	testSearch();
	return 0;
}