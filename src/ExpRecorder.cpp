#include "ExpRecorder.h"

ExpRecorder::ExpRecorder(/* args */)
{
    pointSearchTime = 0;
    pointTreeTravelTime = 0;
    pointModelPreTime = 0;
    pointBindarySearchTime = 0;
    knnRangeQueryConterAvg = 0;
}

ExpRecorder::~ExpRecorder()
{
}

void ExpRecorder::printRangeQuery(int rangeQuerySize)
{
    cout << "range query total time： " << this->rangeTotalTime / rangeQuerySize << " ns per query" << endl;
    this->rangeLookUpTime = this->rangeTotalTime - this->rangeRefinementTime - this->rangeScanTime;
    cout << "range query lookup time: " << this->rangeLookUpTime / rangeQuerySize << " ns per query" << endl;
    cout << "range query lookup time: " << this->rangeRefinementTime / rangeQuerySize << " ns per query" << endl;
    cout << "range query lookup time: " << this->rangeScanTime / rangeQuerySize << " ns per query" << endl;
}

void ExpRecorder::cleanRangeQuery()
{
    this->rangeLookUpTime = 0;
    this->rangeRefinementTime = 0;
    this->rangeScanTime = 0;
}