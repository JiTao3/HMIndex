#pragma once
#include <chrono>
#include <iostream>

using namespace std;

class ExpRecorder
{
private:
    /* data */
public:
    long pointSearchTime = 0;
    long pointTreeTravelTime = 0;
    long pointModelPreTime = 0;
    long pointBindarySearchTime = 0;

    long rangeTotalTime = 0;
    long rangeLookUpTime = 0;
    long rangeRefinementTime = 0;
    long rangeScanTime = 0;

    void printRangeQuery();
    void cleanRangeQuery();

    ExpRecorder(/* args */);
    ~ExpRecorder();
};

