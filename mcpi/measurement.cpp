#include "measurement.h"

void Measurement::addMeasurement(int strategy, int numStartedThreads, string numPoints, double runtime)
{
    measurementEntry entry = {strategy, numStartedThreads, numPoints, runtime};
    data.push_back(entry);
}


void Measurement::addKey(int strategy, int numStartedThreads, string numPoints)
{
    keys.insert(tuple<int, int, string>(strategy, numStartedThreads, numPoints));
}


void Measurement::addpossibleThreadNum(int numStartedThreads)
{
    possibleThreads.insert(numStartedThreads);
}

set<int>* Measurement::getPossibleThreads(){
    return &possibleThreads;
}

map<tuple<int, int, string>, measurementEntry>* Measurement::getAverages(){
    return &averages;
}


measurementEntry Measurement::calculateAverageRuntime(int strategy, int numStartedThreads, string numPoints){
    measurementEntry returnEntry = {strategy, numStartedThreads, numPoints};
    double runtimeValue = 0;
    int counter = 0;
    for(auto const& entry : data){
        if(entry.strategy == strategy && entry.numStartedThreads == numStartedThreads && entry.numPoints.compare(numPoints)){
            runtimeValue += entry.runtime;
            counter++;
        }
    }
    returnEntry.runtime = runtimeValue / counter;
    return returnEntry;
}

void Measurement::generateMapOfRuntimeAverages(){
    for(auto const& setEntry : keys){
        averages.insert({setEntry, calculateAverageRuntime(get<0>(setEntry), get<1>(setEntry), get<2>(setEntry))});
    };
}

void Measurement::print()
{
    int size = data.size();
    for (int i = 0; i < size; i++)
    {
        cout << "Strategy: " << data[i].strategy << ", No. of started threads: " << data[i].numStartedThreads 
        << ", No. of points: " << data[i].numPoints << ", Runtime: " << data[i].runtime << endl;
    }
}