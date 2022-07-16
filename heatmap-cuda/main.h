#include "utils.h"
#include "heatmap.h"
#include "lifecycle.h"

#include <cuda_profiler_api.h>

int main(int argc, char** argv);

__global__ void simulateRoundWithCuda(Heatmap* d_heatmap, int numberOfElements);

__global__ void swapDataWithFutureData(Heatmap* d_heatmap);
