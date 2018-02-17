#include <omp.h>
#include <cmath>
#include <fstream>
#include <unordered_set>

using namespace std;

#define L 100.
#define W 200.
#define D 400.
#define N 1000
#define T 3600
#define Rm 1.
#define Mm 1.
#define G 1.
#define delt .005

typedef struct cord {
	double x;
	double y;
	double z;
} cord;

inline cord operator+(const cord& p1, const cord& p2) {
	return { p1.x + p2.x, p1.y + p2.y, p1.z + p2.z };
}
inline cord operator-(const cord& p1, const cord& p2) {
	return { p1.x - p2.x, p1.y - p2.y, p1.z - p2.z };
}
inline cord& operator+=(cord& p1, const cord& p2) {
	p1.x += p2.x;
	p1.y += p2.y;
	p1.z += p2.z;
	return p1;
}
inline cord operator*(const double& m, const cord& p) {
	return { m * p.x, m * p.y, m * p.z };
}
inline cord operator*(const cord& p, const double& m) {
	return { m * p.x, m * p.y, m * p.z };
}
inline ostream& operator<<(ostream& stream, const cord& p) {
	stream << p.x << " " << p.y << " " << p.z;
	return stream;
}
inline double mag(const cord& p) {
	return sqrt(p.x*p.x + p.y*p.y + p.z*p.z);
}


int main() {
	remove("trajectory.txt");
	remove("log.txt");
	ofstream traj, log;
	traj.open("trajectory.txt");
	log.open("log.txt");
	traj << N << " " << T << " " << delt << endl;
	log << "Running with N = " << N << endl;
	log << "Running with T = " << T << endl;
	log << "Running with delt = " << delt << endl;
	log << "Number of Threads = " << omp_get_max_threads() << endl;
	auto *rad = new double[N];
	auto *mas = new double[N];
	auto *pos = new cord[N];
	auto *vel = new cord[N];
	auto *accp = new cord[N];
	auto *acc = new cord[N];
	for (decltype(N) i = 0; i<N; i++) {
		rad[i] = .5;
		mas[i] = 1;
		pos[i] = { L*rand() / RAND_MAX, W*rand() / RAND_MAX, D*rand() / RAND_MAX };
		vel[i] = { 0., 0., 0. };
		accp[i] = { 0., 0., 0. };
		acc[i] = { 0., 0., 0. };
	}
	auto M = (decltype(T))ceil(T / delt);
	log << "Running " << M << " iterations" << endl;
	auto main_start = omp_get_wtime();
	for (decltype(M) i = 0; i<M; i++) {
		auto start = omp_get_wtime();
		unordered_set<decltype(M)> collisions;
#pragma omp parallel for
		for (decltype(N) j = 0; j<N; j++) {
			acc[j] = { 0., 0., 0. };
			for (decltype(N) k = 0; k<j; k++)
				if (k != j) {
					auto d = mag(pos[k] - pos[j]);
					acc[j] += (G*mas[k] / pow(d, 3))*(pos[k] - pos[j]);
					if (d < rad[j] + rad[k])
#pragma omp critical
						collisions.insert((j > k ? j : k)*N + (j > k ? k : j));
				}
		}
#pragma omp parallel for
		for (decltype(N) j = 0; j<N; j++)
			pos[j] += delt * vel[j] + 0.5*delt*delt*acc[j];
#pragma omp parallel for
		for (decltype(N) j = 0; j<N; j++) {
			vel[j] += .5*delt*(acc[j] + accp[j]);
			if ((pos[j].x + rad[j] > L) || (pos[j].x - rad[j] < 0)) vel[j].x = -vel[j].x;
			if ((pos[j].y + rad[j] > W) || (pos[j].y - rad[j] < 0)) vel[j].y = -vel[j].y;
			if ((pos[j].z + rad[j] > D) || (pos[j].z - rad[j] < 0)) vel[j].z = -vel[j].z;
		}
#pragma omp parallel for
		for (decltype(N) j = 0; j<N; j++)
			accp[j] = acc[j];

		for (decltype(N) j = 0; j < N; j++)
			traj << pos[j] << endl;
		log << "Iteration " << i << " Time " << (omp_get_wtime() - start) << " Collisions " << collisions.size() << endl;
	}
	log << "Ran " << M << " iterations in " << (omp_get_wtime() - main_start) << endl;
	traj.close();
	log.close();
}
