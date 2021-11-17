#include "CellTree.h"
#include <chrono>
using namespace std;

// void genCellTree(string, string);
void expSplitDataSave(string, string);

void expPointSearch(string, string);

void testLeafNode();

void testGridNode();

int main(int argc, char *argv[])
{
	cout << argv[1] << endl;
	cout << argv[2] << endl;
	// cout << argv[3] << endl;
	expSplitDataSave(argv[1], argv[2]);
	// expSplitDataSave(argv[1], argv[2]);
	// expPointSearch(argv[1], argv[2]);
	// expPointSearch(s1, s2);

	// testLeafNode();
	// testGridNode();
	cout << "finish" << endl;
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
	for (auto &query_point : queryPoints)
	{
		vector<array<double, 2> *> result;
		auto start_t = chrono::high_resolution_clock::now();
		cell_tree->pointSearch(query_point, result);
		auto end_t = chrono::high_resolution_clock::now();
		timeconsume += chrono::duration_cast<chrono::nanoseconds>(end_t - start_t).count();
	}
	// average = x / 10^4*10^3 us;
	cout << "time consumption : " << timeconsume << "ns" << endl;
}

