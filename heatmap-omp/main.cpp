// heatmap2.cpp : Defines the entry point for the application.
//

#include "main.h"

using namespace std;

void simulateRound(Heatmap& heatmap)
{
	Heatmap futureHeatmap(heatmap.getWidth(), heatmap.getHeight());

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

    int fieldWidth = 20;
    int fieldHeight = 7;
    int numberOfRounds = 17;
    string hotspotFileName = (argc > 4) ? argv[4] : "hotspots.csv";
    string coordsFileName = (argc > 5) ? argv[5] : "";

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

    readData(hotspotFileName, lifecycles);
    readData(coordsFileName, coords);

    for (auto const& xy : coords) {
        cout << xy.first << ", " << xy.second << endl;
    }

    for (int i = 0; i < numberOfRounds; i++)
    {
        updateHotspots(heatmap, lifecycles, i);
        simulateRound(heatmap);
        updateHotspots(heatmap, lifecycles, i+1);
    }

    if (coords.empty())
    {
        heatmap.printFormattedOutput();
    }
    else
    {
        heatmap.printAtCoords(coords);
    }

    return 0;
}