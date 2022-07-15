#include "main.h"

using namespace std;

__global__ void simulateRoundWithCuda(double* d_data, double* futureData, int numberOfElements)
{
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    while (globalIdx < numberOfElements)
    {
        futureData[globalIdx] = 1;
        globalIdx += blockDim.x * gridDim.x;
        __syncthreads();
    }
}

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        cout << "Error; not enough parameters specified, continuing with default parameters!" << endl;
        // return -2;
    }

    int fieldWidth = 20;
    int fieldHeight = 7;
    int numberOfRounds = 17;
    string hotspotFileName = (argc > 4) ? argv[4] : "hotspots.csv";
    string coordsFileName = (argc > 5) ? argv[5] : "";

    cout << "Reading arguments..." << endl;
    if (argc > 4)
    {
        fieldWidth = stoi(argv[1]);
        fieldHeight = stoi(argv[2]);
        numberOfRounds = stoi(argv[3]);
        hotspotFileName = argv[4];
    }

    if (argc > 5)
    {
        coordsFileName = argv[5];
    }

    Heatmap heatmap(fieldWidth, fieldHeight);
    Lifecycle lifecycles = Lifecycle();
    vector<pair<int, int>> coords;

    readData(hotspotFileName, lifecycles);
    readData(coordsFileName, coords);

    for (auto const &xy : coords)
    {
        cout << xy.first << ", " << xy.second << endl;
    }

    int numberOfElements = heatmap.getSize();

    int threadsPerBlock = 256;
    int blocksPerGrid = (numberOfElements + threadsPerBlock - 1) / threadsPerBlock;

    // Copy all data to device
    double *d_data;
    cudaMalloc(&d_data, numberOfElements);
    double *futureData;
    cudaMalloc(&futureData, numberOfElements);
    cudaMemcpy(d_data, heatmap.data, numberOfElements, cudaMemcpyHostToDevice);
    cudaMemcpy(futureData, heatmap.data, numberOfElements, cudaMemcpyHostToDevice);
    // Run Kernel
    simulateRoundWithCuda<<<threadsPerBlock, blocksPerGrid>>>(d_data, futureData, numberOfElements);
    // Copy data to host
    cudaDeviceSynchronize();
    cudaMemcpy(heatmap.data, futureData, numberOfElements, cudaMemcpyDeviceToHost);
    // Cleanup
    cudaFree(futureData);
    cudaFree(d_data);

    if (coords.empty())
    {
        cout << "Print all coordinates." << endl;
        heatmap.printFormattedOutput();
    }
    else
    {
        cout << "Print selected coordinates." << endl;
        heatmap.printAtCoords(coords);
    }

    return 0;
}