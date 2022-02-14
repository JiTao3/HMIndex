#include "CellTree.h"
// #include "Core/Matrix.h"
// #include <chrono>
using namespace std;

int split_num = 100;
// int split_num = 40 ;// for tiger

// vector<double> data_space_bound = {70.9825433, 142.2560836, 4.999728700, 54.35880621};
// vector<double> data_space_bound = {-81.79535869999985, -65.27891709999955, 38.43836500000005, 45.98917950000055};
// vector<double> data_space_bound = {-90.3100275, -64.566563, 17.627786999999994, 47.457235};//Tiger

vector<double> data_space_bound = {0, 1, 0, 1}; //
string knnN[5] = {"1", "5", "25", "125", "625"};

// void genCellTree(string, string);
void expSplitDataSave(string, string);

void expPointSearch(string, string, string);

void expRangeSearch(string, string, string);

void expkNNSearch(string);

int main(int argc, char *argv[])
{
    cout << argv[1] << endl;
    // cout << argv[2] << endl;
    // cout << argv[3] << endl;
    // string s1 = "/data/jitao/dataset/OSM/osm.csv";
    // string s2 = "/data/jitao/dataset/OSM/new_trained_model_param_for_split2/";
    // expSplitDataSave(argv[1], argv[2]);
    // expSplitDataSave(argv[1], argv[2]);

    // expPointSearch(argv[1], argv[2], argv[3]);
    // expPointSearch(s1, s2);
    // string s1 = "osm_ne_us";
    // string s2 = "0.0001";
    // string s3 = "0.25";
    // expRangeSearch(argv[1], argv[2], argv[3]);
    // expRangeSearch(s1, s2, s3);

    expkNNSearch(argv[1]);
    // cout << "finish!" << endl;
}

void expSplitDataSave(string csv_path, string save_path)
{
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

void expPointSearch(string csv_path, string query_path, string model_param_path)
{

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
    cout << "read query data:" << query_path << endl;
    FileReader *pointQueryFileReader = new FileReader();
    vector<array<double, 2>> queryPoints = pointQueryFileReader->get_array_points(query_path, ",");
    cout << "read finish： " << queryPoints.size() << endl;
    ExpRecorder *exp_Recorder = new ExpRecorder();

    long timeconsume = 0;
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

void expRangeSearch(string dataset, string windowSize, string aspectRadio)
{
    int range_split_num = 100;
    vector<double> data_space_bound;
    string csv_path, model_param_path;
    vector<vector<double>> range_query;
    FileReader *rangeQueryFileReader = new FileReader();
    if (dataset == "uniform")
    {
        data_space_bound = {0, 1, 0, 1};
        csv_path = "/data/jitao/dataset/uniform/2d_len_1e8_seed_1.csv";
        model_param_path = "/data/jitao/dataset/uniform/trained_modelParam_for_split/";
        range_query = rangeQueryFileReader->getRangePoints(
            "/data/jitao/dataset/uniform/range_query/2d_len_1e8_seed_1_1000_" + windowSize + "_" + aspectRadio + ".csv",
            ",");
    }
    else if (dataset == "skewed")
    {
        data_space_bound = {0, 1, 0, 1};
        csv_path = "/data/jitao/dataset/skewed/2d_len_1e8_seed_1.csv";
        model_param_path = "/data/jitao/dataset/skewed/trained_modelParam_for_split/";
        range_query = rangeQueryFileReader->getRangePoints(
            "/data/jitao/dataset/skewed/range_query/2d_len_1e8_seed_1_1000_" + windowSize + "_" + aspectRadio + ".csv",
            ",");
    }
    else if (dataset == "osm_cn")
    {
        data_space_bound = {70.9825433, 142.2560836, 4.999728700, 54.35880621};
        csv_path = "/data/jitao/dataset/OSM/osm.csv";
        model_param_path = "/data/jitao/dataset/OSM/trained_modelParam_for_split2/";
        range_query = rangeQueryFileReader->getRangePoints(
            "/data/jitao/dataset/OSM/range_query/osm_1000_" + windowSize + "_" + aspectRadio + ".csv", ",");
    }
    else if (dataset == "osm_ne_us")
    {
        data_space_bound = {-81.79535869999985, -65.27891709999955, 38.43836500000005, 45.98917950000055};
        csv_path = "/data/jitao/dataset/OSM_US_NE/20_outliers_lon_lat.csv";
        model_param_path = "/data/jitao/dataset/OSM_US_NE/trained_modelParam_for_split/";
        range_query =
            rangeQueryFileReader->getRangePoints("/data/jitao/dataset/OSM_US_NE/range_query/20_outliers_lon_lat_1000_" +
                                                     windowSize + "_" + aspectRadio + ".csv",
                                                 ",");
    }
    else if (dataset == "tiger")
    {
        range_split_num = 40;
        data_space_bound = {-90.3100275, -64.566563, 17.627786999999994, 47.457235};
        csv_path = "/data/jitao/dataset/Tiger/center_tiger_east_17m.txt";
        model_param_path = "/data/jitao/dataset/Tiger/trained_modelParam_for_split/";
        range_query =
            rangeQueryFileReader->getRangePoints("/data/jitao/dataset/Tiger/range_query/center_tiger_east_17m_1000_" +
                                                     windowSize + "_" + aspectRadio + ".csv",
                                                 ",");
    }
    else
    {
        cout << dataset << ": error distribution";
        return;
    }
    string raw_data_path = csv_path;
    cout << "raw data path: " << raw_data_path << endl;
    CellTree *cell_tree = new CellTree(range_split_num, data_space_bound, raw_data_path);
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

    cout << "range query size:" << range_query.size() << "  window size: " << windowSize
         << "  window aspect: " << aspectRadio << endl;

    long timeconsume = 0;
    ExpRecorder *exp_Recorder = new ExpRecorder();

    for (auto &query : range_query)
    {
        vector<array<double, 2> *> result;
        auto start_t = chrono::high_resolution_clock::now();
        cell_tree->rangeSearch(query, result, *exp_Recorder);
        auto end_t = chrono::high_resolution_clock::now();
        timeconsume += chrono::duration_cast<chrono::nanoseconds>(end_t - start_t).count();
        // cout << "result size:" << result.size() << endl;
    }
    cout << "time consumption : " << timeconsume / range_query.size() << " ns per point query" << endl;
    exp_Recorder->printRangeQuery(range_query.size());
    cout << "--------------end------------------" << endl << endl << endl;
}

void expkNNSearch(string dataset)
{
    string csv_path;
    string model_param_path;
    vector<array<double, 2>> queryPoints;
    FileReader *knnQueryFileReader = new FileReader();
    int range_split_num = 100;
    if (dataset == "uniform")
    {
        data_space_bound = {0, 1, 0, 1};
        csv_path = "/data/jitao/dataset/uniform/2d_len_1e8_seed_1.csv";
        model_param_path = "/data/jitao/dataset/uniform/trained_modelParam_for_split/";
        queryPoints =
            knnQueryFileReader->get_array_points("/data/jitao/dataset/uniform/point_query_sample_10w.csv", ",");
    }
    else if (dataset == "skewed")
    {
        data_space_bound = {0, 1, 0, 1};
        csv_path = "/data/jitao/dataset/skewed/2d_len_1e8_seed_1.csv";
        model_param_path = "/data/jitao/dataset/skewed/trained_modelParam_for_split/";
        queryPoints =
            knnQueryFileReader->get_array_points("/data/jitao/dataset/skewed/point_query_sample_10w.csv", ",");
    }
    else if (dataset == "osm_cn")
    {
        data_space_bound = {70.9825433, 142.2560836, 4.999728700, 54.35880621};
        csv_path = "/data/jitao/dataset/OSM/osm.csv";
        model_param_path = "/data/jitao/dataset/OSM/trained_modelParam_for_split2/";
        queryPoints = knnQueryFileReader->get_array_points("/data/jitao/dataset/OSM/point_query_sample_10w.csv", ",");
    }
    else if (dataset == "osm_ne_us")
    {
        data_space_bound = {-81.79535869999985, -65.27891709999955, 38.43836500000005, 45.98917950000055};
        csv_path = "/data/jitao/dataset/OSM_US_NE/20_outliers_lon_lat.csv";
        model_param_path = "/data/jitao/dataset/OSM_US_NE/trained_modelParam_for_split/";
        queryPoints =
            knnQueryFileReader->get_array_points("/data/jitao/dataset/OSM_US_NE/point_query_sample_10w.csv", ",");
    }
    else if (dataset == "tiger")
    {
        range_split_num = 40;
        data_space_bound = {-90.3100275, -64.566563, 17.627786999999994, 47.457235};
        csv_path = "/data/jitao/dataset/Tiger/center_tiger_east_17m.txt";
        model_param_path = "/data/jitao/dataset/Tiger/trained_modelParam_for_split/";
        queryPoints = knnQueryFileReader->get_array_points("/data/jitao/dataset/Tiger/point_query_sample_10w.csv", ",");
    }
    else
    {
        cout << dataset << ": error distribution";
        return;
    }

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

    // cout << k << "NN search" << endl;
    ExpRecorder *exp_Recorder = new ExpRecorder();
    for (auto &n : knnN)
    {
        long timeconsume = 0;
        int k = atoi(n.c_str());
        cout << "-------------------- start query * " << k << " * ---------------------" << endl;
        vector<array<double, 2> *> result;
        for (auto queryPoint : queryPoints)
        {
            auto start_t = chrono::high_resolution_clock::now();
            cell_tree->kNNSearch(queryPoint, k, result, *exp_Recorder);
            auto end_t = chrono::high_resolution_clock::now();
            timeconsume += chrono::duration_cast<chrono::nanoseconds>(end_t - start_t).count();
        }
        cout << "KNN K: " << k << "time consumption : " << timeconsume / queryPoints.size() << "ns per point query"
             << endl;
        cout << "average range query times: " << exp_Recorder->knnRangeQueryConterAvg / queryPoints.size() << endl;
        cout << "-------------------- end query * " << k << " * ---------------------" << endl;
    }
}