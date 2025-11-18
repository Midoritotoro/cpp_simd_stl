#include <simd_stl/concurrency/ThreadPool.h>

int function(int* ptr) noexcept {
	*ptr++ = (*ptr) + 1;
}

int main() {
	std::vector<int> vector;
	vector.reserve(100);

	std::iota(vector.begin(), vector.end(), 0);
	
	simd_stl::concurrency::thread_pool pool;
	
	for (int i = 0; i < vector.size(); ++i)
		pool.submit(function, vector.begin() + i);

	pool.wait_all();

	for (int i = 0; i < vector.size(); ++i)
		std::cout << vector[i];

	return 0;
}