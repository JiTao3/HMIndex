#include <iostream>
// #include <Eigen/Dense>
// #include <Eigen/Core>
#include <vector>
#include "MetaClient.h"
// #include <IndexModel.h>
// #include <CellTree.h>

using namespace std;

// void testLeafNode()
// {
//   FileReader file_reader;
//   vector<array<double, 2>> points = file_reader.get_array_points("/data/jitao/dataset/OSM/split2/9387.csv", ",");
//   vector<double> mapvalVec = file_reader.get_mapval("/data/jitao/dataset/OSM/split2/9387.csv", ",");
//   vector<MetaData> metadataVec;
//   for (int i = 0; i < points.size(); i++)
//   {
//     MetaData metadata(&points[i]);
//     metadata.setMapVal(mapvalVec[i]);
//     metadataVec.push_back(metadata);
//   }
//   vector<double> range_bound = {126.543, 126.74, 44.2496, 44.3546}; // 126.543,44.2756
//   LeafNode *leafnode = new LeafNode(metadataVec, range_bound);
//   leafnode->index_model->loadParameter("/data/jitao/dataset/OSM/new_trained_model_param_for_split2/9387.csv");
//   leafnode->index_model->getErrorBound();
//   vector<double> query = {126.543 - 0.01, 126.543 + 0.01, 44.2756 - 0.01, 44.2756 + 0.01};
//   vector<array<double, 2> *> result;
//   double min_range[2] = {126.543 - 0.01, 44.2756 - 0.01};
//   double max_range[2] = {126.543 + 0.01, 44.2756 + 0.01};
//   // scan(leafnode->metadataVec, 0, leafnode->metadataVec.size() - 1, min_range, max_range, result);
//   leafnode->rangeSearch(query, result);
//   cout << result.size() << endl;
// }

int main()
{
  // testLeafNode();
  connectMetaServer(12333);
}