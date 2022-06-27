#include "measurement.h"

void Measurement::addMeasurement(int strategy, int numStartedThreads, int64_t numPoints, double runtime)
{
    data.push_back(MeasurementEntry(strategy, numStartedThreads, numPoints, runtime));
}