#include <simd_stl/algorithm/find/Search.h>
#include <algorithm>

template <typename It1, typename It2>
void check_search(It1 first1, It1 last1, It2 first2, It2 last2) {
    auto std_res = std::search(first1, last1, first2, last2);
    auto simd_res = simd_stl::algorithm::search(first1, last1, first2, last2);
    simd_stl_assert(std_res == simd_res);
}

template <typename It1, typename It2, typename Pred>
void check_search(It1 first1, It1 last1, It2 first2, It2 last2, Pred pred) {
    auto std_res = std::search(first1, last1, first2, last2, pred);
    auto simd_res = simd_stl::algorithm::search(first1, last1, first2, last2, pred);
    simd_stl_assert(simd_res == std_res);
}

int main() {
    {
        std::vector<int> a, b;
        check_search(a.begin(), a.end(), b.begin(), b.end());
    }
    {
        std::vector<int> a{ 1,2,3,4,5 };
        std::vector<int> b{ 1,2,3,4,5 };
        check_search(a.begin(), a.end(), b.begin(), b.end());
    }
    {
        std::vector<int> a{ 1,2,3,4,5 };
        std::vector<int> b{ 1,2 };
        check_search(a.begin(), a.end(), b.begin(), b.end());
    }
    {
        std::vector<int> a{ 1,2,3,4,5 };
        std::vector<int> b{ 4,5 };
        check_search(a.begin(), a.end(), b.begin(), b.end());
    }
    {
        std::vector<int> a{ 1,2,3,4,5 };
        std::vector<int> b{ 9,9 };
        check_search(a.begin(), a.end(), b.begin(), b.end());
    }
    {
        std::vector<int> a{ 1,2,1,2,3,1,2,3,4 };
        std::vector<int> b{ 1,2,3 };
        check_search(a.begin(), a.end(), b.begin(), b.end());
    }
    {
        std::string s = "Hello World";
        std::string sub = "World";
        check_search(s.begin(), s.end(), sub.begin(), sub.end());
    }
    {
        std::string s = "abcdef";
        std::string sub = "gh";
        check_search(s.begin(), s.end(), sub.begin(), sub.end());
    }
    {
        std::string s = "CaseInsensitive";
        std::string sub = "insensitive";
        auto pred = [](char a, char b) { return std::tolower(a) == std::tolower(b); };
        check_search(s.begin(), s.end(), sub.begin(), sub.end(), pred);
    }
    /*{
        std::list<int> a{ 1,2,3,4,5,6 };
        std::list<int> b{ 3,4 };
        check_search(a.begin(), a.end(), b.begin(), b.end());
    }*/
    {
        std::vector<int> a(10000, 1);
        a[5000] = 42; a[5001] = 43; a[5002] = 44;
        std::vector<int> b{ 42,43,44 };
        check_search(a.begin(), a.end(), b.begin(), b.end());
    }
    {
        std::vector<int> a{ 1,2,3 };
        std::vector<int> b{ 1,2,3,4,5 };
        check_search(a.begin(), a.end(), b.begin(), b.end());
    }

    return 0;
}