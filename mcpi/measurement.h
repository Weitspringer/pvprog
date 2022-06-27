#pragma once
#include <vector>
#include <iostream>
#include <tuple>
#include <set>
#include <map>

using namespace std;

struct measurementEntry{
    int strategy;
    int numStartedThreads;
    string numPoints;
    double runtime;
};

class Measurement
{
private:
    /* data */
    vector<measurementEntry> data;
    map<tuple<int, int, string>, measurementEntry> averages;
    set<tuple<int, int, string>> keys;
    set<int> possibleThreads;
    
public:
    set<int>* getPossibleThreads();
    map<tuple<int, int, string>, measurementEntry>* getAverages();
    void addMeasurement(int strategy, int nthreads, string npoints, double runtime);
    measurementEntry calculateAverageRuntime(int strategy, int numStartedThreads, string numPoints);
    void generateMapOfRuntimeAverages();
    void addKey(int strategy, int numStartedThreads, string numPoints);
    void addpossibleThreadNum(int numStartedThreads);
    void print();
};
