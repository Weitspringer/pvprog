#include <iostream>
#include <cmath>

using namespace std;

const int NUM_THREADS = 8;

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
    sum += heatmap->getValue(x, y);                                                  // self
    sum += (x > 0 && y > 0) ? heatmap->getValue(x - 1, y - 1) : 0.;                  // nordwest
    sum += (x > 0) ? heatmap->getValue(x - 1, y) : 0.;                               // west
    sum += (x > 0 && y < height - 1) ? heatmap->getValue(x - 1, y + 1) : 0.;         // suedwest
    sum += (y < height - 1) ? heatmap->getValue(x, y + 1) : 0.;                      // sued
    sum += (x < width - 1 && y < height - 1) ? heatmap->getValue(x + 1, y + 1) : 0.; // suedost
    sum += (x < width - 1) ? heatmap->getValue(x + 1, y) : 0.;                       // ost
    sum += (x < width - 1 && y > 0) ? heatmap->getValue(x + 1, y - 1) : 0.;          // nordost
    sum += (y > 0) ? heatmap->getValue(x, y - 1) : 0.;                               // nord

    double average = sum / 9;
    double *result = &average;

    output->setValue(x, y, average);
    threadArgs->result = average;

    // pthread_exit(result);
    return (void *)threadArgs;
}

void simulateRound(Heatmap &heatmap)
{
    Heatmap futureHeatmap(heatmap.getWidth(), heatmap.getHeight());

    // int batches = ceil(heatmap.getSize() / NUM_THREADS);
    pthread_t threads[heatmap.getSize()];
    int rc, t;

    threadArgs threadArgs[heatmap.getSize()];
    for (int i = 0; i < heatmap.getSize(); i++)
    {
        threadArgs[i] = {&heatmap, &futureHeatmap, heatmap.getCoordinatesFromIndex(i)};
        rc = pthread_create(&threads[i], NULL, calculateFutureTemperature, (void *)&threadArgs[i]);
        if (rc != 0)
        {
            printf("ERROR; return code from pthread_create() is %d\n", rc);
            exit(-1);
        }
    }

    for (int i = 0; i < heatmap.getSize(); i++)
    {
        rc = pthread_join(threads[i], NULL);
        if (rc != 0)
        {
            printf("ERROR; return code from pthread_join() is %d\n", rc);
            exit(-1);
        }  
    }

    heatmap = futureHeatmap;
    // heatmap.print();
}

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        cout << "Error; not enough parameters specified, continuing with default parameters!" << endl;
        // return -2;
    }

    int fieldWidth = 10;
    int fieldHeight = 10;
    int numberOfRounds = 100;
    string hotspotFileName = (argc > 4) ? argv[4] : "";
    if (argc >= 3)
    {
        fieldWidth = stoi(argv[1]);
        fieldHeight = stoi(argv[2]);
        numberOfRounds = stoi(argv[3]);
    }

    Heatmap heatmap(fieldWidth, fieldHeight);
    heatmap.setValue(1, 1, 1);

    for (int i = 0; i < numberOfRounds; i++)
    {
        simulateRound(heatmap);
    }

    heatmap.print();

    cout << "Done!";
}