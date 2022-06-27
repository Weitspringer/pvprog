#include "lifecycle.h"

void Lifecycle::addValue(pair<int, int> coordinates, pair<int, int> lifespan)
    {
        data.push_back(pair<pair<int, int>, pair<int, int>>(coordinates, lifespan));
    }

vector<pair<int, int>> Lifecycle::getValuesByCoordinates(pair<int, int> coordinates)
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

vector<pair<int, int>> Lifecycle::getCellsByRound(int currentRound)
{
    vector<pair<int, int>> cells;
    for (auto const &entry : data)
    {
        if ((currentRound >= entry.second.first) && (currentRound < entry.second.second))
        {
            cells.push_back(entry.first);
        }
    }
    return cells;
}

void Lifecycle::print()
{
    int i = 0;
    for (auto const &entry : data)
    {
        cout << "Eintrag #" << i++ << ": ";
        pair<int, int> key = entry.first;
        pair<int, int> value = entry.second;
        cout << "zelle (" << key.first << "," << key.second << "): start: " << value.first << " end: " << value.second << endl;
    }
    cout << endl;
}