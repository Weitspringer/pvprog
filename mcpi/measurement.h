#include <vector>

struct MeasurementEntry{
    int strategy;
    int numStartedThreads;
    int64_t numPoints;
    double runtime;
}

class Measurement
{
private:
    /* data */
    vector<MeasurementEntry> data;
public:
    void addMeasurement(int strategy, short nthreads, int64_t npoints, double runtime);
};
