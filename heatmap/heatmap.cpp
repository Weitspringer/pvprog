#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <iterator>
#include <cmath>
#include <vector>

using namespace std;

const int NUM_THREADS = 4;

class Lifecycle
{
    vector<pair<pair<int, int>, pair<int, int>>> data;

public:
    void addValue(pair<int, int> coordinates, pair<int, int> lifespan)
    {
        data.push_back(pair(coordinates, lifespan));
    }

    vector<pair<int, int>> getValuesByCoordinates(pair<int, int> coordinates)
    {
        vector<pair<int, int>> values;
        for (auto const &entry : data)
        {
            if (entry.first == coordinates)
            {
                values.push_back(entry.second);
            }
        }
        return values;
    }

    vector<pair<int, int>> getCellsByRound(int round)
    {
        vector<pair<int, int>> cells;
        for (auto const &entry : data)
        {
            if ( (entry.second.first >= round) && (entry.second.second > round))
            {
                cells.push_back(entry.first);
            }
        }
        return cells;
    }

    void print()
    {
        for (auto const &entry : data)
        {
            pair<int, int> key = entry.first;
            pair<int, int> value = entry.second;
            cout << "zelle (" << key.first << "," << key.second << "): start: " << value.first << " end: " << value.second << endl;
        }
        cout << endl;
    }

};

class Heatmap
{
    double *data;
    int width;
    int height;

public:
    Heatmap(int w, int h)
    {
        data = new double[w * h];
        width = w;
        height = h;
        for (int i = 0; i < w * h; i++)
        {
            data[i] = 0.0;
        }
    }

    Heatmap(const Heatmap &copyHeatMap)
    {
        data = copyHeatMap.data;
        width = copyHeatMap.width;
        height = copyHeatMap.height;
    }

    double getValue(int x, int y)
    {
        return data[x + y * width];
    }

    double getValue(pair<int, int> coordinates)
    {
        return getValue(coordinates.first, coordinates.second);
    }

    void setValue(int x, int y, double value)
    {
        data[x + y * width] = value;
    }

    void setValue(pair<int, int> coordinates, double value)
    {
        setValue(coordinates.first, coordinates.second, value);
    }

    int getWidth()
    {
        return width;
    }

    int getHeight()
    {
        return height;
    }

    int getSize()
    {
        return width * height;
    }

    pair<int, int> getCoordinatesFromIndex(int index)
    {
        return pair(index % width, (int)index / width);
    }

    void print()
    {
        for (int i = 0; i < width * height; i++)
        {
            cout << data[i] << " ";
            if ((i + 1) % width == 0)
                cout << endl;
        }
        cout << endl;
    }

    void printFormattedOutut()
    {
        ofstream outputFile;
        outputFile.open("output.txt", ios_base::app);
        outputFile << endl;
        for (int i = 0; i < width * height; i++)
        {
            char character = (data[i] > 0.9) ? 'X' : (int) ((data[i] + 0.09)*10)%10 + '0';
            outputFile << character;
            if ((i + 1) % width == 0)
                outputFile << endl;
        }
        outputFile << endl;
        outputFile.close();
    }
};

struct threadArgs
{
    Heatmap *heatmapIn;
    Heatmap *heatmapOut;
    pair<int, int> coordinates;
    double result;
};

void *calculateFutureTemperature(void *args)
{
    threadArgs *threadArgs = (struct threadArgs *)args;

    Heatmap *heatmap = threadArgs->heatmapIn;
    Heatmap *output = threadArgs->heatmapOut;
    int x = threadArgs->coordinates.first;
    int y = threadArgs->coordinates.second;

    int width = heatmap->getWidth();
    int height = heatmap->getHeight();

    double sum = 0;
    for (int i = -1; i <= 1; i++)
    {
        for (int j = -1; j <= 1; j++)
        {
            int neighbour_x = x + i;
            int neighbour_y = y + j;
            if (neighbour_x < 0 or neighbour_x >= width or neighbour_y < 0 or neighbour_y >= height)
            {
                sum += 0;
            }
            else
            {
                sum += heatmap->getValue(neighbour_x, neighbour_y);
            }
        }
    }
    double average = sum / 9;

    output->setValue(x, y, average);
    threadArgs->result = average;

    // pthread_exit(result);
    return (void *)threadArgs;
}

void simulateRound(Heatmap &heatmap)
{
    Heatmap futureHeatmap(heatmap.getWidth(), heatmap.getHeight());

    int numberOfBatches = ceil(heatmap.getSize() / NUM_THREADS);
    pthread_t threads[NUM_THREADS];
    int rc;

    threadArgs threadArgs[NUM_THREADS];
    for (int batchID = 0; batchID <= numberOfBatches; batchID++)
    {
        for (int threadID = 0; threadID < NUM_THREADS; threadID++)
        {
            int currentIndex = batchID * NUM_THREADS + threadID;
            if (currentIndex < heatmap.getSize())
            {
                threadArgs[threadID] = {&heatmap, &futureHeatmap, heatmap.getCoordinatesFromIndex(currentIndex)};
                rc = pthread_create(&threads[threadID], NULL, calculateFutureTemperature, (void *)&threadArgs[threadID]);
                if (rc != 0)
                {
                    printf("ERROR; return code from pthread_create() is %d\n", rc);
                    exit(-1);
                }
            }
        }

        for (int threadID = 0; threadID < NUM_THREADS; threadID++)
        {
            int currentIndex = batchID * NUM_THREADS + threadID;
            if (currentIndex < heatmap.getSize())
            {
                rc = pthread_join(threads[threadID], NULL);
                if (rc != 0)
                {
                    printf("ERROR; return code from pthread_join() is %d\n", rc);
                    exit(-1);
                }
            }
        }
    }

    heatmap = futureHeatmap;
    // heatmap.print();
}

void readData(string filename, Lifecycle &lifecycles)
{
    if (filename != "")
    {
        bool flag_header_read = false;

        string line;
        ifstream raw_data_file(filename);
        if (raw_data_file.is_open())
        { // always check whether the file is open
            while (raw_data_file)
            {
                getline(raw_data_file, line);
                if (flag_header_read)
                {
                    vector<int> tokens;
                    string token;
                    istringstream tokenStream(line);
                    if (!line.empty())
                    {
                        while (getline(tokenStream, token, ','))
                        {
                            tokens.push_back(stoi(token));
                        }
                        lifecycles.addValue(pair(tokens[0], tokens[1]), pair(tokens[2], tokens[3]));
                    }
                }
                else
                {
                    flag_header_read = true;
                }
            }
        }
        else
        {
            cout << "Couldn't open file\n";
        }
    }
}

void updateHotspots(Heatmap &heatmap, Lifecycle &lifecycles, int currentRound)
{
    vector<pair<int, int>> activeCells = lifecycles.getCellsByRound(currentRound);
    
    for (auto const &cell : activeCells)
    {
        heatmap.setValue(cell, 1.);
    }
}

int main(int argc, char **argv)
{
    cout << argc << endl;
    if (argc < 3)
    {
        cout << "Error; not enough parameters specified, continuing with default parameters!" << endl;
        // return -2;
    }

    int fieldWidth = 3;
    int fieldHeight = 3;
    int numberOfRounds = 3;
    string hotspotFileName = (argc > 4) ? argv[4] : "";
    if (argc >= 4)
    {
        cout << "argc >=4" << endl;
        fieldWidth = stoi(argv[1]);
        fieldHeight = stoi(argv[2]);
        numberOfRounds = stoi(argv[3]);
        hotspotFileName = argv[4];
    }

    Heatmap heatmap(fieldWidth, fieldHeight);
    heatmap.setValue(1, 1, 1);
    Lifecycle lifecycles = Lifecycle();

    readData(hotspotFileName, lifecycles);
    
    for (int i = 1; i <= numberOfRounds; i++)
    {
        updateHotspots(heatmap, lifecycles, i);
        simulateRound(heatmap);
        heatmap.printFormattedOutut();
        updateHotspots(heatmap, lifecycles, i);
    }

    heatmap.printFormattedOutut();
}