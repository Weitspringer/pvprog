#include "main.h"

using namespace std;

__global__ void simulateRoundWithCuda(Heatmap *heatmap, Heatmap *futureHeatmap, int numElements)
{
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    while (globalIdx < numElements)
    {
        pair<int, int> coordinates = heatmap->getCoordinatesFromIndex(globalIdx);
        futureHeatmap->setValue(coordinates, calculateFutureTemperature(*heatmap, coordinates.first, coordinates.second));

        globalIdx += blockDim.x * gridDim.x;
        __syncthreads();
    }
    heatmap = futureHeatmap;
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

    int threadsPerBlock = 256;
    int blocksPerGrid = (heatmap.getSize() + threadsPerBlock - 1) / threadsPerBlock;

    // Copy data to device
    Heatmap futureHeatmap(heatmap.getWidth(), heatmap.getHeight());
    Heatmap *futureHeatmapPointer = &futureHeatmap;
    Heatmap *heatmapPointer = &heatmap;
    
    // Copy data to device
    cudaMalloc(&futureHeatmapPointer, sizeof(futureHeatmap));
    cudaMalloc(&heatmapPointer, sizeof(heatmap));

    for (int i = 0; i < numberOfRounds; i++)
    {
        updateHotspots(heatmap, lifecycles, i);
        cout << "Round " << i << ", before simulation: " << endl;
        heatmap.printFormattedOutputCout();
        cout << endl;
        cudaMemcpy(&futureHeatmap, &futureHeatmap, sizeof(futureHeatmap), cudaMemcpyHostToDevice);
        cudaMemcpy(&heatmap, &heatmap, sizeof(heatmap), cudaMemcpyHostToDevice);
        simulateRoundWithCuda<<<threadsPerBlock, blocksPerGrid>>>(&heatmap, &futureHeatmap, heatmap.getSize());
        cudaDeviceSynchronize();
        cudaMemcpy(&heatmap, &futureHeatmap, sizeof(heatmap), cudaMemcpyDeviceToHost);
        cout << "Round " << i << ", after simulation" << endl;
        heatmap.printFormattedOutputCout();
        cout << endl;
        updateHotspots(heatmap, lifecycles, i + 1);
    }

    // Copy results back to device
    cudaFree(&futureHeatmap);
    cudaFree(&heatmap);

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