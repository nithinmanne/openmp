#include <omp.h>
#include <iostream>
#include <cmath>
#include <cstdlib>

#define n 16
#define MAX 100
#define eta 0.001

double *func(double a[n][n], double b[n]);

int main() {
	double a[n][n], b[n];
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++)
			a[i][j] = 0;
		a[i][i] = 1;
		b[i] = 1;
	}
	double *x = func(a, b);
	std::cin.get();
}

double *func(double a[n][n], double b[n]) {
	double x[n], newv[n];	//new is an operator in C++, hence I used name newv
	int i, j, k;
	double sum;
#pragma omp parallel for
	for (i = 0; i < n; i++)
		x[i] = 0.;
	while (true) {
#pragma omp parallel for private(sum)
		for (j = 0; j < n; j++) {
			sum = 0.;
#pragma omp parallel for reduction(+:sum)
			for (k = 0; k < n; k++)
				if (k != j)
					sum += a[j][k] * x[k];
			newv[j] = (1. / a[j][j])*(b[j] - sum);
		}
		double conv = 0;
#pragma omp parallel for reduction(+:conv)
		for (j = 0; j < n; j++) {
			conv += std::abs(x[j]-newv[j]);
			x[j] = newv[j];
		}
		std::cout << conv << std::endl;
		if (conv < eta) break;
	}
}