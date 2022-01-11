#include "ExpRecorder.h"

ExpRecorder::ExpRecorder(/* args */)
{
    pointSearchTime = 0;
    pointTreeTravelTime = 0;
    pointModelPreTime = 0;
    pointBindarySearchTime = 0;
}

ExpRecorder::~ExpRecorder()
{
}

void ExpRecorder::printRangeQuery()
{
    cout << "range query total time" << this->rangeTotalTime << "ns" << endl;
    this->rangeLookUpTime = this->rangeTotalTime - this->rangeRefinementTime - this->rangeScanTime;
    cout << "range query lookup time: " << this->rangeLookUpTime << "ns" << endl;
    cout << "range query lookup time: " << this->rangeRefinementTime << "ns" << endl;
    cout << "range query lookup time: " << this->rangeScanTime << "ns" << endl;
}

void ExpRecorder::cleanRangeQuery()
{
    this->rangeLookUpTime = 0;
    this->rangeRefinementTime = 0;
    this->rangeScanTime = 0;
}