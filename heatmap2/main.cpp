// heatmap2.cpp : Defines the entry point for the application.
//

#include "main.h"

using namespace std;

double calculateFutureTemperature(Heatmap &heatmap, int x, int y)
{
	int width = heatmap.getWidth();
    int height = heatmap.getHeight();

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
				sum += heatmap.getValue(neighbour_x, neighbour_y);
			}
		}
	}

    double average = sum / 9;

    return average;
}

void simulateRound(Heatmap& heatmap)
{
	Heatmap futureHeatmap(heatmap.getWidth(), heatmap.getHeight());

	#pragma omp parallel for
	for (int x = 0; x < heatmap.getWidth(); x++)
	{
		for (int y = 0; y < heatmap.getHeight(); y++)
		{
			futureHeatmap.setValue(x, y, calculateFutureTemperature(heatmap, x, y));
		}
	}
	heatmap = futureHeatmap;
}


int main(int argc, char** argv)
{
    if (argc < 3)
    {
        cout << "Error; not enough parameters specified, continuing with default parameters!" << endl;
        // return -2;
    }

    int fieldWidth = 3;
    int fieldHeight = 3;
    int numberOfRounds = 10;
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

    // readData(hotspotFileName, lifecycles);

    for (int i = 1; i <= numberOfRounds; i++)
    {
        //updateHotspots(heatmap, lifecycles, i);
        simulateRound(heatmap);
        //updateHotspots(heatmap, lifecycles, i);
    }

    heatmap.printFormattedOutut();
}