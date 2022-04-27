#pragma once
#include "ExpRecorder.h"
#include "IndexModel.h"
#include "MetaData.h"
#include "Utils.h"
#include <algorithm>
#include <bitset>
#include <limits>
#include <vector>

using namespace std;

class LeafNode
{
  public:
    std::vector<MetaData> metadataVec;

    int dim = 2;
    double mapvalBound[2] = {0, 0};
    std::vector<double> range_bound;
    double cell_area = 1.0;
    IndexModel *index_model;
    // int key_couter = 0;
    bool mergeDelete = false;
    bool splitFlag = false;
    void *parent = nullptr;

    // for update
    array<MetaData, INSERT_BUFFERSIZE> insertBuffer;
    int bufferDataSize = 0;
    bitset<BITMAP_SIZE> metadataVecBitMap;
    bool bufferOrdered = true;

    LeafNode();
    LeafNode(std::vector<MetaData> &_metadatas, std::vector<double> _range_bound);
    ~LeafNode();

    std::vector<double> getRangeBound();
    double getCellArea();
    void setMapVals();

    vector<array<double, 2> *> &pointSearch(array<double, 2> key, vector<array<double, 2> *> &result,
                                            ExpRecorder &exp_Recorder);
    vector<array<double, 2> *> &rangeSearch(std::vector<double> query_range, vector<array<double, 2> *> &result,
                                            ExpRecorder &exp_Recorder);
    void saveMetaDataVectoCSV(string file_path);

    void initialBitMap();

    bool insert(array<double, 2> &point);
    bool remove(array<double, 2> &point);

    int getKeysNum();

    void retrainModel();

    void kNNInNode(std::vector<double> query_range,
                   priority_queue<array<double, 2> *, vector<array<double, 2> *>, sortForKNN> &temp_result);

  private:
};
