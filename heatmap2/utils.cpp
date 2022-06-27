#include "utils.h"

void readData(string filename, Lifecycle& lifecycles)
{
    if (filename != "")
    {
        bool flag_header_read = false;

        ifstream raw_data_file(filename);
        if (raw_data_file.is_open())
        { // always check whether the file is open
            while (raw_data_file)
            {
                string line;
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

void readData(string filename, vector<pair<int, int>>& coords)
{
    if (filename != "")
    {
        bool flag_header_read = false;

        
        ifstream raw_data_file(filename);
        if (raw_data_file.is_open())
        { // always check whether the file is open
            while (raw_data_file)
            {
                string line;
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
                        coords.push_back(pair(tokens[0], tokens[1]));
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

void updateHotspots(Heatmap& heatmap, Lifecycle& lifecycles, int currentRound)
{
    vector<pair<int, int>> activeCells = lifecycles.getCellsByRound(currentRound);

    for (auto const& cell : activeCells)
    {
        if (cell.first < heatmap.getWidth() && cell.second < heatmap.getHeight())
        {
            heatmap.setValue(cell, 1);
            cout << "hotspot set (Round " << currentRound << " " << cell.first << ", " << cell.second << endl;
        }
    }
}
