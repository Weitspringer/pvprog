#include "main.h"

using namespace std;

__global__ void simulateRoundWithCuda(Heatmap* d_heatmap, long numberOfElements)
{
    // Calculate position in a flattened array
    long threadPositionFlat = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadPositionFlat < numberOfElements)
    {
        pair<int, int> coordinates = d_heatmap->getCoordinatesFromIndex(threadPositionFlat);
        d_heatmap->setFutureValue(coordinates, calculateFutureTemperature(*d_heatmap, coordinates.first, coordinates.second));
    }
}

__global__ void swapDataWithFutureData(Heatmap* d_heatmap)
{
    d_heatmap->overrideDataWithFutureData();
}

__global__ void printDeviceData(Heatmap* heatmap) {
    heatmap->printFormattedOutputDevice();
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

    long numberOfElements = heatmap.getSize();

    long threadsPerBlock = numberOfElements / 8;
    long blocksPerGrid = (numberOfElements + threadsPerBlock - 1) / threadsPerBlock;

    // Create class storage on device and copy top level class
    Heatmap *d_heatmap;
    cudaMalloc((void **)&d_heatmap, sizeof(Heatmap));
    cudaMemcpy(d_heatmap, &heatmap, sizeof(Heatmap), cudaMemcpyHostToDevice);
    // Make an allocated region on device for use by pointer in class
    double *d_data;
    cudaMalloc((void **)&d_data, sizeof(double)*numberOfElements);
    cudaMemcpy(d_data, heatmap.data, sizeof(double)*numberOfElements, cudaMemcpyHostToDevice);
    cudaMemcpy(&(d_heatmap->data), &d_data, sizeof(double *), cudaMemcpyHostToDevice);
    double *d_futureData;
    cudaMalloc((void **)&d_futureData, sizeof(double)*numberOfElements);
    cudaMemcpy(&(d_heatmap->futureData), &d_futureData, sizeof(double *), cudaMemcpyHostToDevice);

    for (int i = 0; i < numberOfRounds; i++) 
    {
        // Update of hotspots in round i
        // Get active cells for round i
        vector<pair<int, int>> activeCellsVector = lifecycles.getCellsByRound(i);
        int numberOfActiveCells = activeCellsVector.size();
        pair<int, int>* activeCellsArray = &activeCellsVector[0];
        // Copy to device
        pair<int, int>* d_activeCells;
        for (int j = 0; j < numberOfActiveCells; j++) {
            //cout << "Round " << i << ", active cell: " << activeCellsArray[j].first << ", " << activeCellsArray[j].second << endl;
        }
        cudaMalloc((void **)&d_activeCells, sizeof(pair<int, int>)*numberOfActiveCells);
        cudaMemcpy(d_activeCells, activeCellsArray, sizeof(pair<int, int>)*numberOfActiveCells, cudaMemcpyHostToDevice);
        updateHotspotsDevice<<<threadsPerBlock, blocksPerGrid>>>(d_heatmap, d_activeCells, numberOfActiveCells);
        cudaFree(d_activeCells);
        // Run simulation kernel
        simulateRoundWithCuda<<<threadsPerBlock, blocksPerGrid>>>(d_heatmap, numberOfElements);
        swapDataWithFutureData<<<1,1>>>(d_heatmap);
        // Update of hotspots in round i+1
        // Get active cells for round i+1
        activeCellsVector = lifecycles.getCellsByRound(i+1);
        numberOfActiveCells = activeCellsVector.size();
        activeCellsArray = &activeCellsVector[0];
        cudaMalloc((void **)&d_activeCells, sizeof(pair<int, int>)*numberOfActiveCells);
        cudaMemcpy(d_activeCells, activeCellsArray, sizeof(pair<int, int>)*numberOfActiveCells, cudaMemcpyHostToDevice);
        updateHotspotsDevice<<<threadsPerBlock, blocksPerGrid>>>(d_heatmap, d_activeCells, numberOfActiveCells);
        cudaFree(d_activeCells);
    }
    
    // Copy data to host
    //printDeviceData<<<1,1>>>(d_heatmap);
    cudaMemcpy(&d_data, &(d_heatmap->data), sizeof(double *), cudaMemcpyDeviceToHost);
    cudaMemcpy(heatmap.data, d_data, sizeof(double)*numberOfElements, cudaMemcpyDeviceToHost);
    
    // Cleanup
    cudaFree(d_heatmap);
    cudaFree(d_data);
    cudaFree(d_futureData);

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