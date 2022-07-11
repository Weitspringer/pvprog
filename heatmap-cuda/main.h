#include "utils.h"
#include "heatmap.h"
#include "lifecycle.h"

#include <cuda_profiler_api.h>

int main(int argc, char** argv);

__global__ void _cuda_simulate_round(Heatmap *heatmap, Heatmap *futureHeatmap, int numElements);
