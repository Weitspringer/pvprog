#pragma once
#include <vector>
#include <iostream>

using namespace std;

class Lifecycle
{
    vector<pair<pair<int, int>, pair<int, int>>> data;

public:
    __host__ __device__ void addValue(pair<int, int> coordinates, pair<int, int> lifespan);

    __host__ __device__ vector<pair<int, int>> getValuesByCoordinates(pair<int, int> coordinates);

    __host__ __device__ vector<pair<int, int>> getCellsByRound(int round);

    void print();

};