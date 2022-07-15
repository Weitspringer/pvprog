#include "main.h"

using namespace std;

__global__ void simulateRoundWithCuda(Heatmap* d_heatmap, int numberOfElements)
{
    // Calculate position in a flattened array
    int threadPositionFlat = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadPositionFlat < numberOfElements)
    {
        pair<int, int> coordinates = d_heatmap->getCoordinatesFromIndex(threadPositionFlat);
        d_heatmap->setValue(coordinates.first, coordinates.second, calculateFutureTemperature(*d_heatmap, coordinates.first, coordinates.second));
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

    updateHotspots(heatmap, lifecycles, 0);
    heatmap.printFormattedOutputCout();

    // Create class storage on device and copy top level class
    Heatmap *d_heatmap;
    cudaMalloc((void **)&d_heatmap, sizeof(Heatmap));
    cudaMemcpy(d_heatmap, &heatmap, sizeof(Heatmap), cudaMemcpyHostToDevice);
    // Make an allocated region on device for use by pointer in class
    double *d_data;
    cudaMalloc((void **)&d_data, sizeof(double)*numberOfElements);
    cudaMemcpy(d_data, heatmap.data, sizeof(double)*numberOfElements, cudaMemcpyHostToDevice);
    int *d_width;
    cudaMalloc((void **)&d_width, sizeof(int));
    cudaMemcpy(d_width, &heatmap.width, sizeof(int), cudaMemcpyHostToDevice);
    int *d_height;
    cudaMalloc((void **)&d_height, sizeof(int));
    cudaMemcpy(d_height, &heatmap.height, sizeof(int), cudaMemcpyHostToDevice);
    // Copy pointer to allocated device storage to device class
    cudaMemcpy(&(d_heatmap->data), &d_data, sizeof(double *), cudaMemcpyHostToDevice);

    // Run Kernel
    simulateRoundWithCuda<<<threadsPerBlock, blocksPerGrid>>>(d_heatmap, numberOfElements);
    
    // Copy data to host
    cudaDeviceSynchronize();
    cudaMemcpy(&d_data, &(d_heatmap->data), sizeof(double)*numberOfElements, cudaMemcpyDeviceToDevice);
    cudaMemcpy(heatmap.data, d_data, sizeof(double)*numberOfElements, cudaMemcpyDeviceToHost);
    // Cleanup
    cudaFree(d_heatmap);
    cudaFree(d_data);
    cudaFree(d_width);
    cudaFree(d_height);

    //for (int i = 0; i < numberOfRounds; i++)
    //{
    //    updateHotspots(heatmap, lifecycles, i);
    //    cout << "updateHotspots in Round " << i << "/" << numberOfRounds << endl;
    //    simulateRound(heatmap);
    //    cout << "simulateRound in Round " << i << "/" << endl;
    //    updateHotspots(heatmap, lifecycles, i+1);
    //    cout << "updateHotspots(i+1) in Round " << i << "/" << endl;
    //}

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