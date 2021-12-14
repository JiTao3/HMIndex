#pragma once
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

#include <boost/variant.hpp>
#include <boost/variant/get.hpp>
#include <torch/torch.h>

#include "ExpRecorder.h"
#include "FileReader.h"
#include "GridNode.h"
#include "IndexModel.h"
#include "InnerNode.h"
#include "LeafNode.h"
#include "MetaData.h"
#include "Utils.h"

void getAllMetaData(boost::variant<InnerNode *, LeafNode *, GridNode *, int> root, vector<MetaData> &all_medata);
void getAllData(boost::variant<InnerNode *, LeafNode *, GridNode *, int> root, vector<array<double, 2> *> &result);

class CellTree
{
  public:
    vector<array<double, 2>> raw_data;
    vector<vector<vector<MetaData>>> initial_partition_data;
    int split_num = 100;
    std::vector<double> init_partion_bound_x;
    std::vector<double> init_partion_bound_y;
    std::vector<double> data_space_bound; // initial root node
    std::vector<std::vector<int>> cell_bound_idx;

    InnerNode root;

    int cellSplitTh = 50000;
    int gridSplitTh = 15000;
    int mergeTh = 5000;
	int removeTh = 4000;

    int modelCapability = 10000;

    string saveSplitPath;
    string modelParamPath;

    // hist info for knn

    vector<double> hist_info_x;
    vector<double> hist_info_y;
    int bin_num = 500;

    // global insert buffer

  public:
    CellTree();
    ~CellTree();
    CellTree(int split_num, vector<double> data_space_bound, string raw_path);
    void loadRawData(string file_path);
    void loadSampleData(string file_path);
    void initialPartionBound(int n);
    void initialPartionData();
    void buildTree(std::vector<std::vector<int>> cell_bound_idx, InnerNode *root_node, int child_idx);
    void saveSplitData(boost::variant<InnerNode *, LeafNode *, GridNode *, int> root);
    void buildCheck(boost::variant<InnerNode *, LeafNode *, GridNode *, int> root, int child_index,
                    bool openTraining = false);
    vector<array<double, 2> *> &pointSearch(array<double, 2> &query, vector<array<double, 2> *> &result,
                                            ExpRecorder &exp_Recorder);
    vector<array<double, 2> *> &rangeSearch(vector<double> &query, vector<array<double, 2> *> &result,
                                            ExpRecorder &exp_Recorder);
    vector<array<double, 2> *> &kNNSearch(array<double, 2> &query, int k, vector<array<double, 2> *> &result);
    void DFSCelltree(vector<double> &query, vector<array<double, 2> *> &result,
                     boost::variant<InnerNode *, LeafNode *, GridNode *, int> root);
    void train(boost::variant<InnerNode *, LeafNode *, GridNode *, int> root);

    void insert(array<double, 2> &point);
    void remove(array<double, 2> &point);
};