#pragma once
#include <fstream>
#include <iostream>
#include <vector>

using namespace std;

class Heatmap
{
public:
    double *data;
    double *futureData;
    int width;
    int height;

    Heatmap(int w, int h);

    Heatmap(const Heatmap &copyHeatMap);

    __host__ __device__ double getValue(int x, int y);

    __host__ __device__ double getValue(pair<int, int> coordinates);

    __host__ __device__ void setValue(int x, int y, double value);

    __host__ __device__ void setValue(pair<int, int> coordinates, double value);

    __host__ __device__ double getFutureValue(int x, int y);

    __host__ __device__ double getFutureValue(pair<int, int> coordinates);

    __host__ __device__ void setFutureValue(int x, int y, double value);

    __host__ __device__ void setFutureValue(pair<int, int> coordinates, double value);

    __host__ __device__ void overrideDataWithFutureData();

    __host__ __device__ int getWidth();

    __host__ __device__ int getHeight();

    __host__ __device__ long getSize();

    __host__ __device__ pair<int, int> getCoordinatesFromIndex(int index);

    void print();

    void printFormattedOutput();

    void printFormattedOutputCout();

    __device__ void printFormattedOutputDevice();

    void printAtCoords(vector<pair<int, int>>);
};
