#pragma once
#include <chrono>

class ExpRecorder
{
private:
    /* data */
public:
    long pointSearchTime = 0;
    long pointTreeTravelTime = 0;
    long pointModelPreTime = 0;
    long pointBindarySearchTime = 0;

    ExpRecorder(/* args */);
    ~ExpRecorder();
};

