#pragma once
#include "ExpRecorder.h"
#include "IndexModel.h"
#include "MetaData.h"
#include <algorithm>
#include <array>
#include <bitset>
#include <iostream>
#include <queue>
#include <vector>

#define DISSOCIATION 0
#define OVERLAP 100
#define INTERSECTION 101
#define CONTAIN 102

using namespace std;

bool equalMetadata(MetaData &me_data1, MetaData &me_data2);
double percentile(vector<double> &vectorIn, double percent, bool is_sorted = false);

vector<int> idx_to_vector_int(int idx, int dim);
bool match_range(vector<double> point_data, vector<double> cell_bound, int dim);

vector<MetaData> splitLeafNodeToLeaf(vector<MetaData> &metadatas, vector<double> split_point, int index);
vector<MetaData> splitLeafNodeToGrid(vector<MetaData> &metadatas, vector<double> range_bound);

vector<double> splitNodeToSplitedRegion(vector<double> cell_range_bound, vector<double> split_point, int index);
vector<double> splitLeafNodeToGridRegion(vector<double> cell_range_bound, vector<double> split_point, int split_dim,
                                         int child_index);

vector<array<double, 2> *> &bindary_search(vector<MetaData> &metadataVec, bitset<BITMAP_SIZE> &bitmap, int begin_idx,
                                           int end_idx, MetaData &meta_key, std::vector<array<double, 2> *> &result,
                                           ExpRecorder &expr);

vector<array<double, 2> *> &bindary_search(array<MetaData, INSERT_BUFFERSIZE> &buffer, int begin_idx, int end_idx,
                                           MetaData &meta_key, std::vector<array<double, 2> *> &result);

int adjustPosition(vector<MetaData> &metadataVec, vector<int> &error_bound, int pre_position, MetaData meta_key,
                   int leftORright);

int knnAdjustPosition(vector<MetaData> &metadataVec, vector<int> &error_bound, int pre_position, MetaData meta_key);

void scan(vector<MetaData> &metadataVec, int begin, int end, double *min_range, double *max_range,
          vector<array<double, 2> *> &result);

void orderMetaData(vector<MetaData> &metadataVec);

bool compareMetadata(MetaData me_data1, MetaData me_data2);

bool insertMetadataInRange(vector<MetaData> &metadataVec, bitset<BITMAP_SIZE> &bitmap, int begin_idx, int end_idx,
                           MetaData &meta_key);

bool deleteMetadataInRange(vector<MetaData> &metadataVec, bitset<BITMAP_SIZE> &bitmap, int begin_idx, int end_idx,
                           MetaData &meta_key);

void scanBuffer(array<MetaData, INSERT_BUFFERSIZE> &insertBuffer, int bufferDataSize, double *min_range,
                double *max_range, vector<array<double, 2> *> &result);

vector<int> getCellIndex(array<double, 2> &raData, vector<double> &initial_partition_bound_x,
                         vector<double> &initial_partition_bound_y);

int queryCellRealtion(vector<double> &rangeBound, vector<double> &query);

template <typename T> std::vector<double> LinSpace(T start_in, T end_in, int num_in)
{

    std::vector<double> linspaced;

    double start = static_cast<double>(start_in);
    double end = static_cast<double>(end_in);
    double num = static_cast<double>(num_in);

    if (num == 0)
    {
        return linspaced;
    }
    if (num == 1)
    {
        linspaced.push_back(start);
        return linspaced;
    }

    double delta = (end - start) / (num - 1);

    for (int i = 0; i < num - 1; ++i)
    {
        linspaced.push_back(start + delta * i);
    }
    linspaced.push_back(end); // I want to ensure that start and end
                              // are exactly the same as the input
    return linspaced;
}

double distFunction(array<double, 2> *, double, double);

class sortForKNN
{
    // array<double, 2> queryPoint;
  public:
    double x;
    double y;
    sortForKNN(array<double, 2> &point)
    {
        // queryPoint = point;
        x = point[0];
        y = point[1];
    }
    bool operator()(array<double, 2> *point1, array<double, 2> *point2)
    {
        // return (distFunction(point1, queryPoint) < distFunction(point2, queryPoint));
        double distance1 = pow(((*point1)[0] - x), 2) + pow(((*point1)[1] - y), 2);
        double distance2 = pow(((*point2)[0] - x), 2) + pow(((*point2)[1] - y), 2);
        return distance1 < distance2;
    }
};
