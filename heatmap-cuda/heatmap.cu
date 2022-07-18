#include "heatmap.h"
#include "utils.h"

Heatmap::Heatmap(int w, int h)
{
    data = new double[w * h];
    futureData = new double[w * h];
    width = w;
    height = h;
    for (int i = 0; i < w * h; i++)
    {
        data[i] = 0.0;
    }
}

Heatmap::Heatmap(const Heatmap &copyHeatMap)
{
    data = copyHeatMap.data;
    futureData = copyHeatMap.futureData;
    width = copyHeatMap.width;
    height = copyHeatMap.height;
}

double Heatmap::getValue(int x, int y)
{
    return data[x + y * width];
}

double Heatmap::getValue(pair<int, int> coordinates)
{
    return getValue(coordinates.first, coordinates.second);
}

void Heatmap::setValue(int x, int y, double value)
{
    data[x + y * width] = value;
}

void Heatmap::setValue(pair<int, int> coordinates, double value)
{
    setValue(coordinates.first, coordinates.second, value);
}

double Heatmap::getFutureValue(int x, int y)
{
    return futureData[x + y * width];
}

double Heatmap::getFutureValue(pair<int, int> coordinates)
{
    return getFutureValue(coordinates.first, coordinates.second);
}

void Heatmap::setFutureValue(int x, int y, double value)
{
    futureData[x + y * width] = value;
}

void Heatmap::setFutureValue(pair<int, int> coordinates, double value)
{
    setFutureValue(coordinates.first, coordinates.second, value);
}

void Heatmap::overrideDataWithFutureData()
{
    data = futureData;
}

int Heatmap::getWidth()
{
    return width;
}

int Heatmap::getHeight()
{
    return height;
}

long Heatmap::getSize()
{
    return width * height;
}

pair<int, int> Heatmap::getCoordinatesFromIndex(int index)
{
    return pair<int, int>(index % width, (int)index / width);
}

void Heatmap::print()
{
    for (int i = 0; i < width * height; i++)
    {
        cout << data[i] << " ";
        if ((i + 1) % width == 0)
            cout << endl;
    }
    cout << endl;
}

void Heatmap::printFormattedOutput()
{

    ofstream outputFile;
    outputFile.open("output.txt");
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

void Heatmap::printFormattedOutputCout()
{
    for (int i = 0; i < width * height; i++)
    {
        char character = (data[i] > 0.9) ? 'X' : ((int) ((data[i] + 0.09)*10))%10 + '0';
        cout << character;
        if ((i + 1) % width == 0)
            cout << endl;
    }
}

__device__ void Heatmap::printFormattedOutputDevice()
{
    for (int i = 0; i < width * height; i++)
    {
        char character = (data[i] > 0.9) ? 'X' : ((int) ((data[i] + 0.09)*10))%10 + '0';
        printf("%c", character);
        if ((i + 1) % width == 0)
            printf("\n");
    }
}

void Heatmap::printAtCoords(vector<pair<int, int>> coords) {
    ofstream outputFile;
    outputFile.open("output.txt");
    outputFile << endl;

    for (auto const& coordinate : coords) {
        outputFile << getValue(coordinate) << endl;
    }

    outputFile << endl;
    outputFile.close();
}
