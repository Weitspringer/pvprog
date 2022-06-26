#pragma once
#include <fstream>
#include <iostream>

using namespace std;

class Heatmap
{
    double *data;
    int width;
    int height;

public:
    Heatmap(int w, int h);

    Heatmap(const Heatmap &copyHeatMap);

    double getValue(int x, int y);

    double getValue(pair<int, int> coordinates);

    void setValue(int x, int y, double value);

    void setValue(pair<int, int> coordinates, double value);

    int getWidth();

    int getHeight();

    int getSize();

    pair<int, int> getCoordinatesFromIndex(int index);

    void print();

    void printFormattedOutut();
};