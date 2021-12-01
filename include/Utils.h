#pragma once
#include <iostream>
#include <vector>
#include <array>
#include <algorithm>
#include <bitset>
#include "MetaData.h"
#include "IndexModel.h"
#include "ExpRecorder.h"

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
vector<double> splitLeafNodeToGridRegion(vector<double> cell_range_bound, vector<double> split_point, int split_dim, int child_index);

vector<array<double, 2> *> &bindary_search(vector<MetaData> &metadataVec, bitset<BITMAP_SIZE>& bitmap, int begin_idx, int end_idx, MetaData &meta_key, std::vector<array<double, 2> *> &result, ExpRecorder &expr);
int adjustPosition(vector<MetaData> &metadataVec, vector<int> error_bound, int pre_position, MetaData meta_key, int leftORright);
void scan(vector<MetaData>& metadataVec, int begin, int end, double *min_range, double *max_range, vector<array<double, 2> *>& result);

void orderMetaData(vector<MetaData> &metadataVec);

bool compareMetadata(MetaData me_data1, MetaData me_data2);

bool deleteMetadataInRange(vector<MetaData> &metadataVec, bitset<BITMAP_SIZE>& bitmap, int begin_idx, int end_idx, MetaData &meta_key);

void scanBuffer(array<MetaData, INSERT_BUFFERSIZE>& insertBuffer, int bufferDataSize, double *min_range, double *max_range, vector<array<double, 2> *>& result);

vector<int> getCellIndex(array<double, 2> &raData, vector<double> &initial_partition_bound_x, vector<double> &initial_partition_bound_y);

int queryCellRealtion(vector<double>& rangeBound, vector<double>&query);

template <typename T>
std::vector<double> LinSpace(T start_in, T end_in, int num_in)
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

double distFunction(array<double, 2>*, array<double, 2>&);

struct sortForKNN
{
    array<double, 2> queryPoint;
    sortForKNN(array<double, 2> &point)
    {
        queryPoint = point;
    }
    bool operator()(array<double, 2>* point1, array<double, 2>* point2)
    {
        return (distFunction(point1, queryPoint) < distFunction(point2, queryPoint));
    }
};

