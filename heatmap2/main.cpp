// heatmap2.cpp : Defines the entry point for the application.
//

#include "main.h"

using namespace std;

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
    cout << argc;
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

    cout << "Reading arguments";
    if (argc > 4)
    {
        fieldWidth = stoi(argv[1]);
        fieldHeight = stoi(argv[2]);
        numberOfRounds = stoi(argv[3]);
        hotspotFileName = argv[4];
    }

    if (argc > 5) {
        coordsFileName = argv[5];
    }

    Heatmap heatmap(fieldWidth, fieldHeight);
    Lifecycle lifecycles = Lifecycle();
    vector<pair<int, int>> coords;

    cout << "Hallo!";
    readData(hotspotFileName, lifecycles);
    readData(coordsFileName, coords);

    for (auto const& xy : coords) {
        cout << xy.first << ", " << xy.second << endl;
    }

    for (int i = 0; i < numberOfRounds; i++)
    {
        updateHotspots(heatmap, lifecycles, i);
        cout << "updateHotspots in Round " << i << "/" << numberOfRounds << endl;
        simulateRound(heatmap);
        cout << "simulateRound in Round " << i << "/" << endl;
        updateHotspots(heatmap, lifecycles, i+1);
        cout << "updateHotspots(i+1) in Round " << i << "/" << endl;
    }

    cout << "reachedOutput" << endl;

    if (coords.empty())
    {
        cout << "Print all coordinates" << endl;
        heatmap.printFormattedOutput();
    }
    else
    {
        cout << "Print selected coordinates" << endl;
        heatmap.printAtCoords(coords);
    }

    return 0;
}