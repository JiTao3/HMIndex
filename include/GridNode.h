#pragma once

#include "ExpRecorder.h"
#include "IndexModel.h"
#include "MetaData.h"
#include "Utils.h"
#include <array>
#include <vector>
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
    void *parent = nullptr;
    vector<double> parent_rangeBound;

    // for update
    array<MetaData, INSERT_BUFFERSIZE> insertBuffer;
    int bufferDataSize = 0;
    bitset<BITMAP_SIZE> metadataVecBitMap;

    // bit set for delete?

    GridNode();
    GridNode(vector<MetaData> metadataVec, vector<double> rangeBound);
    ~GridNode();

    vector<array<double, 2> *> &pointSearch(array<double, 2> key, vector<array<double, 2> *> &result,
                                            ExpRecorder &exp_Recorder);
    vector<array<double, 2> *> &rangeSearch(vector<double> query_range, vector<array<double, 2> *> &result,
                                            ExpRecorder &exp_Recorder);
    double mapValtoScaling(double mapVal);
    double scalingtoMapVal(double scaling);
    double getCellArea();
    double getGridArea();
    void saveMetaDataVectoCSV(string file_path);

    void initialBitMap();

    bool insert(array<double, 2> &point);
    bool remove(array<double, 2> &point);

    int getKeysNum();

  private:
};
