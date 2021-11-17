#pragma once

#include <vector>
#include <array>
#include "MetaData.h"
#include "IndexModel.h"
#include "Utils.h"
#include "ExpRecorder.h"
//#include "InnerNode.h"

using namespace std;

class GridNode
{
public:
	vector<MetaData> metadataVec;
	int dim = MetaData::dim;
	double mapvalBound[2] = {0, 0};
	vector<double> rangeBound;
	IndexModel *index_model;
	//InnerNode* parent = nullptr;
	vector<double> parent_rangeBound;

	GridNode();
	GridNode(vector<MetaData> metadataVec, vector<double> rangeBound);
	~GridNode();

	vector<array<double, 2> *>& pointSearch(array<double, 2> key, vector<array<double, 2> *>& result, ExpRecorder& exp_Recorder);
	vector<array<double, 2> *>& rangeSearch(vector<double> query_range,  vector<array<double, 2> *> &result);
	double mapValtoScaling(double mapVal);
	double scalingtoMapVal(double scaling);
	double getCellArea();
	double getGridArea();
	void saveMetaDataVectoCSV(string file_path);
private:
};
