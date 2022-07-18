#pragma once

#include "heatmap.h"
#include "lifecycle.h"

#include <iostream>
#include <string>
#include <sstream>

using namespace std;

void readData(string filename, Lifecycle& lifecycles);

void readData(string filename, vector<pair<int, int>>& coords);

__host__ __device__ double calculateFutureTemperature(Heatmap& heatmap, int x, int y);

void updateHotspots(Heatmap& heatmap, Lifecycle& lifecycles, int currentRound);

__global__ void updateHotspotsDevice(Heatmap* heatmap, pair<int, int>* activeCells, int numberOfCells);