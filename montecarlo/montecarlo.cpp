#include <omp.h>
#include <cmath>
#include <iostream>

using namespace std;

int main(int argc, char *argv[]) {
	long N;
	if (argc == 2) N = strtol(argv[1], nullptr, 0);
	else {
		cout << "Enter N: ";
		cin >> N;
	}
	int T = omp_get_max_threads();
	auto nt = (decltype(N))ceil(N / T);
	cout << "Running with N = " << (nt*T) << endl;
	decltype(N) ac = 0;
#pragma omp parallel
	{
		decltype(N) lac = 0;
		float x, y;
		for (decltype(N) i = 0; i<nt; i++) {
			x = (float)rand() / RAND_MAX;
			y = (float)rand() / RAND_MAX;
			if (x*x + y * y <= 1) lac++;
		}
#pragma omp critical
		ac += lac;
	}
	cout << "Value of PI = " << (ac*4. / N) << endl;
	cin.get();
	cin.get();
}
