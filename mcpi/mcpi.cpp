// mcpi.cpp: Definiert den Einstiegspunkt für die Anwendung.
//

#define _USE_MATH_DEFINES

#include "mcpi.h"
#include <omp.h>

#include <cmath>
#include <random>
#include <string>
#include <vector>
#include <iomanip>
#include <atomic>

using namespace std;

int64_t numberOfPointsInCircleGlobal = 0;
atomic<int64_t> numberOfPointsInCircleGlobalAtomic;

bool isInCircle(double x, double y) {
	return (x * x + y * y <= 1);
}

int64_t generateRandomPointLocal(int threadID, int64_t npoints) {
	mt19937 generator(threadID);
	uniform_real_distribution<double> dist(0.0, 1.0);

	vector<pair<double, double>> points;
	int64_t numberOfPointsInCircle = 0;

	for (int i = 0; i < npoints; i++) {

		double x = dist(generator);
		double y = dist(generator);

		if (isInCircle(x, y)) numberOfPointsInCircle++;
	}

	return numberOfPointsInCircle;

}

void generateRandomPointGlobal(int threadID, int64_t npoints) {
	mt19937 generator(threadID);
	uniform_real_distribution<double> dist(0.0, 1.0);

	vector<pair<double, double>> points;

	for (int i = 0; i < npoints; i++) {
		double x = dist(generator);
		double y = dist(generator);


		if (isInCircle(x, y)) numberOfPointsInCircleGlobal++;
	}
}

void generateRandomPointGlobalAtomic(int threadID, int64_t npoints) {
	mt19937 generator(threadID);
	uniform_real_distribution<double> dist(0.0, 1.0);

	vector<pair<double, double>> points;

	for (int i = 0; i < npoints; i++) {
		double x = dist(generator);
		double y = dist(generator);


		if (isInCircle(x, y)) numberOfPointsInCircleGlobalAtomic++;
	}
}

double calculatePiLocal(int numThreads, int64_t numPoints)
{
	int64_t pointsPerBatch = ceil(numPoints / numThreads);
	int64_t remaining_points = numPoints;

	vector<int64_t> pointsPerBatchArray;
	for (int i = 0; i < numThreads; i++) {
		if (remaining_points < pointsPerBatch) {
			pointsPerBatch = remaining_points;
		}
		pointsPerBatchArray.push_back(pointsPerBatch);
		remaining_points -= pointsPerBatch;
	}
	vector<int64_t> pointsInCircle;

	omp_set_num_threads(numThreads);
	#pragma omp parallel for
	for (int i = 0; i < numThreads; i++){
		pointsInCircle.push_back(generateRandomPointLocal(i, pointsPerBatchArray[i]));
	}

	int64_t sum = 0;
	for (auto const& element : pointsInCircle) {
		sum += element;
	}

	return 4.0 * sum / numPoints;	
}

double calculatePiGlobal(int numThreads, int64_t numPoints) {
	int64_t pointsPerBatch = ceil(numPoints / numThreads);
	int64_t remaining_points = numPoints;

	vector<int64_t> pointsPerBatchArray;
	for (int i = 0; i < numThreads; i++) {
		if (remaining_points < pointsPerBatch) {
			pointsPerBatch = remaining_points;
		}
		pointsPerBatchArray.push_back(pointsPerBatch);
		remaining_points -= pointsPerBatch;
	}

	omp_set_num_threads(numThreads);
	#pragma omp parallel for
	for (int i = 0; i < numThreads; i++) {
		generateRandomPointGlobal(i, pointsPerBatchArray[i]);
	}

	return 4.0 * numberOfPointsInCircleGlobal / numPoints;
}

double calculatePiGlobalAtomic(int numThreads, int64_t numPoints) {
	numberOfPointsInCircleGlobalAtomic = 0;
	int64_t pointsPerBatch = ceil(numPoints / numThreads);
	int64_t remaining_points = numPoints;

	vector<int64_t> pointsPerBatchArray;
	for (int i = 0; i < numThreads; i++) {
		if (remaining_points < pointsPerBatch) {
			pointsPerBatch = remaining_points;
		}
		pointsPerBatchArray.push_back(pointsPerBatch);
		remaining_points -= pointsPerBatch;
	}

	omp_set_num_threads(numThreads);
	#pragma omp parallel for
	for (int i = 0; i < numThreads; i++) {
		generateRandomPointGlobalAtomic(i, pointsPerBatchArray[i]);
	}

	return 4.0 * numberOfPointsInCircleGlobalAtomic / numPoints;
}

int main(int argc, char* argv[])
{
	int64_t numberOfPoints = 1024;
	int numberOfStartedThreads = 8;
	int strategyID = 4;

	if (argc > 1) {
		numberOfPoints = strtoll(argv[1], NULL, 10);
		numberOfStartedThreads = stoi(argv[2]);
		strategyID = stoi(argv[3]);
	}

	double estimatedPi;

	switch (strategyID)
	{
	case 1:
		estimatedPi = calculatePiLocal(numberOfStartedThreads, numberOfPoints);
		break;
	case 2:
		estimatedPi = calculatePiGlobal(numberOfStartedThreads, numberOfPoints);
		break;
	case 3:
		estimatedPi = calculatePiGlobalAtomic(numberOfStartedThreads, numberOfPoints);
		break;
	default:
		estimatedPi = M_PI;
		break;
	}

	cout << setprecision(15) << std::fixed << estimatedPi;
	return 0;
}