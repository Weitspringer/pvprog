#include <iostream>

using namespace std;

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

    void setValue(int x, int y, double value)
    {
        data[x + y * width] = value;
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
        return (int)(sizeof data / sizeof(data[0]));
    }

    void print()
    {
        for (int i = 0; i < width * height; i++)
        {
            cout << data[i] << " ";
            if ((i + 1) % width == 0)
                cout << endl;
        }
    }
};

double calculateFutureTemperature(Heatmap &heatmap, int x, int y)
{

    int width = heatmap.getWidth();
    int height = heatmap.getHeight();

    double sum = 0;
    sum += heatmap.getValue(x, y);                                                  // self
    sum += (x > 0 && y > 0) ? heatmap.getValue(x - 1, y - 1) : 0.;                  // nordwest
    sum += (x > 0) ? heatmap.getValue(x - 1, y) : 0.;                               // west
    sum += (x > 0 && y < height - 1) ? heatmap.getValue(x - 1, y + 1) : 0.;         // suedwest
    sum += (y < height - 1) ? heatmap.getValue(x, y + 1) : 0.;                      // sued
    sum += (x < width - 1 && y < height - 1) ? heatmap.getValue(x + 1, y + 1) : 0.; // suedost
    sum += (x < width - 1) ? heatmap.getValue(x + 1, y) : 0.;                       // ost
    sum += (x < width - 1 && y > 0) ? heatmap.getValue(x + 1, y - 1) : 0.;          // nordost
    sum += (y > 0) ? heatmap.getValue(x, y - 1) : 0.;                               // nord

    double average = sum / 9;

    return average;
}

void simulateRound(Heatmap &heatmap)
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
    heatmap.print();
    cout << endl;
}

int main(int argc, char **argv)
{
    cout << "Hallo!" << endl;
    if (argc < 3)
    {
        cout << "Error; not enough parameters specified";
        return -1;
    }

    int fieldWidth = stoi(argv[1]);
    int fieldHeight = stoi(argv[2]);
    int numberOfRounds = stoi(argv[3]);
    string hotspotFileName = (argc > 4) ? argv[4] : "";

    Heatmap heatmap(fieldWidth, fieldHeight);
    heatmap.setValue(1, 1, 1);

    for (int i = 0; i < numberOfRounds; i++)
    {
        simulateRound(heatmap);
    }

    cout << "Done!";
}