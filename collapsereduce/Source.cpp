#include <omp.h>
#include <iostream>

int main() {
	int A[10][10];
	for (int i = 0; i < 10; i++)
		for (int j = 0; j < 10; j++)
			A[i][j] = i*j;
	int s = 0, c = 0;
#pragma omp parallel
	{
		c += 1;
#pragma omp for reduction(+:s)
		for (int i = 0; i < 10; i++)
			for (int j = 0; j < 10; j++)
				s += A[i][j];
	}
	std::cout << s << std::endl << c << std::endl;
	std::cin.get();
}