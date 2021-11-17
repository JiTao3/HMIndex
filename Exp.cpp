#include "CellTree.h"
// #include "Core/Matrix.h"
// #include <chrono>
using namespace std;

// void genCellTree(string, string);
void expSplitDataSave(string, string);

void expPointSearch(string, string);

void expRangeSearch(string, string);

int main(int argc, char *argv[])
{
	cout << argv[1] << endl;
	cout << argv[2] << endl;
	// cout << argv[3] << endl;
	// string s1 = "/data/jitao/dataset/OSM/osm.csv";
	// string s2 = "/data/jitao/dataset/OSM/new_trained_model_param_for_split2/";
	// expSplitDataSave(argv[1], argv[2]);
	// expSplitDataSave(argv[1], argv[2]);
	// expPointSearch(argv[1], argv[2]);
	// expPointSearch(s1, s2);
	expRangeSearch(argv[1], argv[2]);

	// expRangeSearch(s1, s2);

	// testLeafNode();
	// testGridNode();
	cout << "finish!" << endl;
}

void expSplitDataSave(string csv_path, string save_path)
{
	// 1000000 116309831
	int split_num = 100;

	// vector<double> data_space_bound = {-81.79535869999985, -65.27891709999955, 38.43836500000005, 45.98917950000055};
	vector<double> data_space_bound = {70.9825433, 142.2560836, 4.999728700, 54.35880621};

	string raw_data_path = csv_path;
	cout << "raw data path: " << raw_data_path << endl;
	cout << "save data path: " << save_path << endl;
	CellTree *cell_tree = new CellTree(split_num, data_space_bound, raw_data_path);
	cell_tree->saveSplitPath = save_path;
	cout << "raw data size : " << cell_tree->raw_data.size() << endl;

	cout << "build begin" << endl;
	cell_tree->buildTree(cell_tree->cell_bound_idx, &cell_tree->root, 0);
	cout << "build end" << endl;

	cout << "build check begin" << endl;
	cell_tree->buildCheck(&cell_tree->root, 0);
	cout << "build check end" << endl;

	cout << "save begin" << endl;
	cell_tree->saveSplitData(&cell_tree->root);
	cout << "save end" << endl;
}

void expPointSearch(string csv_path, string model_param_path)
{
	int split_num = 100;

	vector<double> data_space_bound = {70.9825433, 142.2560836, 4.999728700, 54.35880621};
	string raw_data_path = csv_path;
	cout << "raw data path: " << raw_data_path << endl;
	CellTree *cell_tree = new CellTree(split_num, data_space_bound, raw_data_path);
	cout << "raw data size : " << cell_tree->raw_data.size() << endl;

	cout << "build begin" << endl;
	cell_tree->buildTree(cell_tree->cell_bound_idx, &cell_tree->root, 0);
	cout << "build end" << endl;

	cout << "build check begin" << endl;
	cell_tree->buildCheck(&cell_tree->root, 0);
	cout << "build check end" << endl;

	cout << "load model parameter begin" << endl;
	cell_tree->modelParamPath = model_param_path;
	cell_tree->train(&cell_tree->root);
	cout << "train fiish" << endl;

	// cell_tree->pointSearch();
	cout << "read query data" << endl;
	FileReader *pointQueryFileReader = new FileReader();
	vector<array<double, 2>> queryPoints = pointQueryFileReader->get_array_points("/data/jitao/dataset/OSM/point_query_sample_10w.csv", ",");
	cout << "read finish： " << queryPoints.size() << endl;
	long timeconsume = 0;
	ExpRecorder *exp_Recorder = new ExpRecorder();
	for (auto &query_point : queryPoints)
	{
		vector<array<double, 2> *> result;
		auto start_t = chrono::high_resolution_clock::now();
		cell_tree->pointSearch(query_point, result, *exp_Recorder);
		auto end_t = chrono::high_resolution_clock::now();
		timeconsume += chrono::duration_cast<chrono::nanoseconds>(end_t - start_t).count();
	}
	cout << "time consumption : " << timeconsume / queryPoints.size() << "ns per point query" << endl;
	cout << "tree travel :" << exp_Recorder->pointTreeTravelTime / queryPoints.size() << "ns per point" << endl;
	cout << "positition pre :" << exp_Recorder->pointModelPreTime / queryPoints.size() << "ns per point" << endl;
	cout << "bindary search :" << exp_Recorder->pointBindarySearchTime / queryPoints.size() << "ns per point" << endl;
}

void expRangeSearch(string csv_path, string model_param_path)
{
	int split_num = 100;

	vector<double> data_space_bound = {70.9825433, 142.2560836, 4.999728700, 54.35880621};
	string raw_data_path = csv_path;
	cout << "raw data path: " << raw_data_path << endl;
	CellTree *cell_tree = new CellTree(split_num, data_space_bound, raw_data_path);
	cout << "raw data size : " << cell_tree->raw_data.size() << endl;

	cout << "build begin" << endl;
	cell_tree->buildTree(cell_tree->cell_bound_idx, &cell_tree->root, 0);
	cout << "build end" << endl;

	cout << "build check begin" << endl;
	cell_tree->buildCheck(&cell_tree->root, 0);
	cout << "build check end" << endl;

	cout << "load model parameter begin" << endl;
	cell_tree->modelParamPath = model_param_path;
	cell_tree->train(&cell_tree->root);
	cout << "train fiish" << endl;

	cout << "read range query data: " << endl;
	FileReader *rangeQueryFileReader = new FileReader();

	vector<vector<double>> range_query = rangeQueryFileReader->getRangePoints("/data/jitao/dataset/OSM/rangequery_osm_1000_0.003_1.csv", ",");
	// x_min x_max y_min y_max
	cout << "range query size:" << range_query.size() << endl;

	long timeconsume = 0;
	ExpRecorder *exp_Recorder = new ExpRecorder();

	for (auto &query : range_query)
	{
		vector<array<double, 2> *> result;
		auto start_t = chrono::high_resolution_clock::now();
		cell_tree->rangeSearch(query, result, *exp_Recorder);
		auto end_t = chrono::high_resolution_clock::now();
		timeconsume += chrono::duration_cast<chrono::nanoseconds>(end_t - start_t).count();
		cout << "result size:" << result.size() << endl;
	}
	cout << "time consumption : " << timeconsume / range_query.size() << "ns per point query" << endl;
}

// void testLeafNode()
// {
// 	FileReader file_reader;
// 	vector<array<double, 2>> points = file_reader.get_array_points("/data/jitao/dataset/OSM/split1/0.csv", ",");
// 	vector<double> mapvalVec = file_reader.get_mapval("/data/jitao/dataset/OSM/split1/0.csv", ",");
// 	vector<MetaData> metadataVec;
// 	for (int i = 0; i < points.size(); i++)
// 	{
// 		MetaData metadata(&points[i]);
// 		metadata.setMapVal(mapvalVec[i]);
// 		metadataVec.push_back(metadata);
// 	}
// 	vector<double> range_bound = {86.0087, 99.6185, 20.1372, 22.1056};
// 	LeafNode *leafnode = new LeafNode(metadataVec, range_bound);
// 	leafnode->index_model->loadParameter("/data/jitao/dataset/OSM/trained_modelParam/0.csv");
// 	leafnode->index_model->getErrorBound();
// 	array<double, 2> query = {86.0849, 22.0527};
// 	vector<array<double, 2> *> result;
// 	leafnode->pointSearch(query, result);
// }

// void testGridNode()
// {
// 	FileReader file_reader;
// 	vector<array<double, 2>> points = file_reader.get_array_points("/data/jitao/dataset/OSM/split1/1.csv", ",");
// 	vector<double> mapvalVec = file_reader.get_mapval("/data/jitao/dataset/OSM/split1/1.csv", ",");
// 	vector<MetaData> metadataVec;
// 	for (int i = 0; i < points.size(); i++)
// 	{
// 		MetaData metadata(&points[i]);
// 		metadata.setMapVal(mapvalVec[i]);
// 		metadataVec.push_back(metadata);
// 	}
// 	vector<double> range_bound = {99.6186, 99.9054, 21.6027, 22.1056};
// 	LeafNode *leafnode = new LeafNode(metadataVec, range_bound);
// 	leafnode->index_model->loadParameter("/data/jitao/dataset/OSM/trained_modelParam/1.csv");
// 	leafnode->index_model->getErrorBound();
// 	array<double, 2> query = {99.6188, 21.9764};
// 	vector<array<double, 2> *> result;
// 	leafnode->pointSearch(query, result);
// }