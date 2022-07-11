#pragma once
#include <vector>
#include <iostream>

using namespace std;

class Lifecycle
{
    vector<pair<pair<int, int>, pair<int, int>>> data;

public:
    void addValue(pair<int, int> coordinates, pair<int, int> lifespan);

    vector<pair<int, int>> getValuesByCoordinates(pair<int, int> coordinates);

    vector<pair<int, int>> getCellsByRound(int round);

    void print();

};