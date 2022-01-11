#include "CellTree.h"
using namespace std;


int split_num = 100;// For OSM_CN OSM_NE_US
// int split_num = 40 ;// for tiger
// vector<double> data_space_bound = {70.9825433, 142.2560836, 4.999728700, 54.35880621};// OSM_CN
// vector<double> data_space_bound = {-81.79535869999985, -65.27891709999955, 38.43836500000005, 45.98917950000055}; // OSM_NE_US
// vector<double> data_space_bound = {-90.3100275, -64.566563, 17.627786999999994, 47.457235};//Tiger
vector<double> data_space_bound = {0, 1, 0, 1}; //


// void genCellTree(string, string);
void expSplitDataSave(string, string);

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

