#include <iostream>
#include <cstdlib>
#include <cmath>
#include <pthread.h>
#include <vector>

using namespace std;

#define NUM_THREADS 5

struct threadArgs
{
    int x;
    int y;
    vector<vector<double>> heatmap;
};

void *CalculateTemperatureFromNeighborsAndSelf(void *args)
{
    // params int x, int y, vector<vector<double>> &heatmap
    threadArgs *actualArgs = (struct threadArgs *)args;

    vector<vector<double>> heatmap = actualArgs->heatmap;
    int x = actualArgs->x;
    int y = actualArgs->y;

    int height = heatmap.size() - 1;
    int width = heatmap[0].size() - 1;

    // Noch nicht beachtet: Hotspots über Zeit

    double sum = 0;
    sum += heatmap[x][y];                                         // self
    sum += (x > 0 && y > 0) ? heatmap[x - 1][y - 1] : 0;          // nordwest
    sum += (x > 0) ? heatmap[x - 1][y] : 0;                       // west
    sum += (x > 0 && y < height) ? heatmap[x - 1][y + 1] : 0;     // suedwest
    sum += (y < height) ? heatmap[x][y + 1] : 0;                  // sued
    sum += (x < width && y < height) ? heatmap[x + 1][y + 1] : 0; // suedost
    sum += (x < width) ? heatmap[x + 1][y] : 0;                   // ost
    sum += (x < width && y > 0) ? heatmap[x + 1][y - 1] : 0;      // nordost
    sum += (y > 0) ? heatmap[x][y - 1] : 0;                       // nord

    double average = sum / 9;

    return &average;
}

void simulateStep(vector<vector<double>> &heatmap, int width, int height)
{
    vector<vector<double>> nextRoundHeatMap = heatmap;

    int batches = ceil(width * height / NUM_THREADS);
    pthread_t threads[NUM_THREADS];
    int rc;
    int t;

    int x = 0;
    int y = 0;

    for (int i = 1; i <= batches; i++)
    {
        for (t = 0; t < NUM_THREADS; t++)
        {

            if (x + 1 < width)
            {
                x++;
            }
            else
            {
                x = 0;
                y++;
            }

            printf("In main: creating thread %ld\n", t);
            rc = pthread_create(&threads[t], NULL, CalculateTemperatureFromNeighborsAndSelf, (void *)t);
            if (rc != 0)
            {
                printf("ERROR; return code from pthread_create() is %d\n", rc);
                exit(-1);
            }
        }

        for (t = 0; t < NUM_THREADS; t++)
        {
            rc = pthread_join(threads[t], NULL);
            if (rc != 0)
            {
                printf("ERROR; return code from pthread_join() is %d\n", rc);
                exit(-1);
            }
        }
    }
}

void printHeatmap(vector<vector<double>> &heatmap)
{
    for (int i = 0; i < heatmap.size(); i++)
    {
        for (int j = 0; j < heatmap[0].size(); j++)
        {
            cout << heatmap[i][j];
        }
        cout << "\n";
    }
    return;
}

int main(int argc, char **argv)
{
    cout << "Hallo!";
    if (argc < 3)
    {
        cout << "Error; not enough parameters";
    }

    int fieldWidth = stoi(argv[0]);
    int fieldHeight = stoi(argv[1]);
    int numberOfRounds = stoi(argv[2]);
    string hotspotFileName = argv[3];

    vector<vector<double>> heatmap(fieldWidth, vector<double>(fieldHeight, 0));
    vector<vector<double>> testMap{{0.0, 0., 0.}, {0.0, 1., 0.}, {0.0, 0., 0.}};

    for (int roundNum = 1; roundNum <= numberOfRounds; roundNum++)
    {
        simulateStep(heatmap, fieldWidth, fieldHeight);
    }

    printHeatmap(heatmap);
}