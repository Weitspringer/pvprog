#include "measurement.h"

void Measurement::addValue(int strategy, short nthreads, int npoints, double runtime)
{
    data.push_back(pair(strategy, nthreads, npoints, runtime));
}