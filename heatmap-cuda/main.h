#include "utils.h"
#include "heatmap.h"
#include "lifecycle.h"

#include <cuda_profiler_api.h>

int main(int argc, char** argv);

__global__ void _cuda_simulate_round(double* d_data, double* futureData, int numberOfElements);
