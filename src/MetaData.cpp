#include <iostream>
#include "MetaData.h"

using namespace std;

MetaData::MetaData()
{
    this->data = nullptr;
    this->map_val = 0.0;
}

MetaData::MetaData(array<double, 2>* data)
{
    this->data = data;
}

MetaData::~MetaData() {

}

void MetaData::setMapVal(double map_val)
{
    this->map_val = map_val;
}

void MetaData::setMapVal(vector<double>& cellBound, double cellArea)
{
    vector<double> areaRange;
    double metaArea = 1.0;
    for (int i = 0; i < dim; i++)
    {
        metaArea *= ((*data)[i] - cellBound[i * 2]);
    }
    map_val = metaArea / cellArea;
    this->setMapVal(map_val);
}

void MetaData::print()
{
    for (int i = 0; i < dim; i++)
    {
        cout << (*data)[i] << "\t";
    }
    cout << endl
        << "Map Value: " << this->map_val;
}