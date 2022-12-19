#include "CellTree.h"
// #include "Core/Matrix.h"
// #include <chrono>
using namespace std;

int split_num = 100;
// int split_num = 40 ;// for tiger

// vector<double> data_space_bound = {70.9825433, 142.2560836, 4.999728700, 54.35880621};
// vector<double> data_space_bound = {-81.79535869999985, -65.27891709999955, 38.43836500000005, 45.98917950000055};
// vector<double> data_space_bound = {-90.3100275, -64.566563, 17.627786999999994, 47.457235}; // Tiger

// vector<double> data_space_bound = {0, 1, 0, 1}; //
string knnN[6] = {"1", "1", "5", "25", "125", "625"};
// string knnN[3] = {"25", "125", "625"};
// string knnN[1] = {"25"};
string insertSizeArray[4] = {"0.1", "0.2", "0.3", "0.4"};

// string windowSizeArray[5] = {"0.001", "0.005", "0.01", "0.04", "0.16"};
// string windowSizeArray[2] = {"0.001", "0.005"};
string windowSizeArray[1] = {"0.01"};
// string aspectRadioArray[5] = {"0.25", "0.5", "1", "2", "4"};
string aspectRadioArray[1] = {"1"};

// void genCellTree(string, string);
void expSplitDataSave(string, string);

void expPointSearch(string);

void expRangeSearch(string);

void expkNNSearch(string);

void expInsert();

void expAvgHeightPartition(string);

int main(int argc, char *argv[])
{

    // expPointSearch(argv[1]);

    // expRangeSearch(argv[1]);

    // expkNNSearch(argv[1]);
    cout << "start!" << endl;

    expAvgHeightPartition(argv[1]);

    // expSplitDataSave(argv[1], argv[2]);

    cout << "finish!" << endl;
}

void expSplitDataSave(string csv_path, string save_path)
{
    vector<double> data_space_bound;

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

void expPointSearch(string dataset)
{
    int range_split_num = 100;
    vector<double> data_space_bound;
    string csv_path, model_param_path, query_path;

    FileReader *rangeQueryFileReader = new FileReader();
    if (dataset == "uniform")
    {
        data_space_bound = {0, 1, 0, 1};
        csv_path = "";
        model_param_path = "";
        query_path = "";
    }
    else if (dataset == "skewed")
    {
        data_space_bound = {0, 1, 0, 1};
        csv_path = "";
        model_param_path = "";
        query_path = "";
    }
    else if (dataset == "osm_cn")
    {
        data_space_bound = {70.9825433, 142.2560836, 4.999728700, 54.35880621};
        csv_path = "";
        model_param_path = "";
    }
    else if (dataset == "osm_ne_us")
    {
        data_space_bound = {-81.79535869999985, -65.27891709999955, 38.43836500000005, 45.98917950000055};
        csv_path = "";
        model_param_path = "";
        query_path = "";
    }
    else if (dataset == "tiger")
    {
        range_split_num = 40;
        data_space_bound = {-90.3100275, -64.566563, 17.627786999999994, 47.457235};
        csv_path = "";
        model_param_path = "";
        query_path = "";
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

void expRangeSearch(string dataset)
{
    int range_split_num = 100;
    vector<double> data_space_bound;
    string csv_path, model_param_path;

    string rangeQueryPrefix;

    FileReader *rangeQueryFileReader = new FileReader();
    if (dataset == "uniform")
    {
        data_space_bound = {0, 1, 0, 1};
        csv_path = "";
        model_param_path = "";

        rangeQueryPrefix = "";
    }
    else if (dataset == "skewed")
    {
        data_space_bound = {0, 1, 0, 1};
        csv_path = "";
        model_param_path = "";
        rangeQueryPrefix = "";
    }
    else if (dataset == "osm_cn")
    {
        data_space_bound = {70.9825433, 142.2560836, 4.999728700, 54.35880621};
        csv_path = "";
        model_param_path = "";
        rangeQueryPrefix = "";
    }
    else if (dataset == "osm_ne_us")
    {
        data_space_bound = {-81.79535869999985, -65.27891709999955, 38.43836500000005, 45.98917950000055};
        csv_path = "";
        model_param_path = "";
        rangeQueryPrefix = "";
    }
    else if (dataset == "tiger")
    {
        range_split_num = 40;
        data_space_bound = {-90.3100275, -64.566563, 17.627786999999994, 47.457235};
        csv_path = "";
        model_param_path = "";
        rangeQueryPrefix = "";
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

    for (auto windowSize : windowSizeArray)
    {
        for (auto aspectRadio : aspectRadioArray)
        {
            vector<vector<double>> range_query =
                rangeQueryFileReader->getRangePoints(rangeQueryPrefix + windowSize + "_" + aspectRadio + ".csv", ",");

            cout << "range query size:" << range_query.size() << "  window size: " << windowSize
                 << "  window aspect: " << aspectRadio << endl;
            for (int cache = 0; cache < 2; cache++)
            {
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
                cout << "--------------end------------------" << endl
                     << endl
                     << endl;
            }
        }
    }
}

void expkNNSearch(string dataset)
{
    string csv_path;
    string model_param_path;
    vector<array<double, 2>> queryPoints;
    vector<double> data_space_bound;

    FileReader *knnQueryFileReader = new FileReader();
    int range_split_num = 100;
    if (dataset == "uniform")
    {
        data_space_bound = {0, 1, 0, 1};
        csv_path = "";
        model_param_path = "";
        queryPoints =
            knnQueryFileReader->get_array_points("", ",");
    }
    else if (dataset == "skewed")
    {
        data_space_bound = {0, 1, 0, 1};
        csv_path = "";
        model_param_path = "";
        queryPoints =
            knnQueryFileReader->get_array_points("", ",");
    }
    else if (dataset == "osm_cn")
    {
        data_space_bound = {70.9825433, 142.2560836, 4.999728700, 54.35880621};
        csv_path = "";
        model_param_path = "";
        queryPoints = knnQueryFileReader->get_array_points("", ",");
    }
    else if (dataset == "osm_ne_us")
    {
        data_space_bound = {-81.79535869999985, -65.27891709999955, 38.43836500000005, 45.98917950000055};
        csv_path = "";
        model_param_path = "";
        queryPoints =
            knnQueryFileReader->get_array_points("", ",");
    }
    else if (dataset == "tiger")
    {
        range_split_num = 40;
        data_space_bound = {-90.3100275, -64.566563, 17.627786999999994, 47.457235};
        csv_path = "";
        model_param_path = "";
        queryPoints = knnQueryFileReader->get_array_points("", ",");
    }
    else
    {
        cout << dataset << ": error distribution";
        return;
    }
    for (auto space_bound : data_space_bound)
    {
        cout << space_bound << ", ";
    }
    cout << range_split_num << endl;
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

    cout << "load model parameter begin : " << model_param_path << endl;
    cell_tree->modelParamPath = model_param_path;
    cell_tree->train(&cell_tree->root);
    cout << "train fiish" << endl;

    vector<array<double, 2>> knnQueryPoints;
    for (int i = 0; i < 1000; i++)
    {
        knnQueryPoints.push_back(queryPoints[i]);
    }

    // cout << k << "NN search" << endl;
    for (auto &n : knnN)
    {
        ExpRecorder *exp_Recorder = new ExpRecorder();
        long timeconsume = 0;
        int k = atoi(n.c_str());
        cout << "-------------------- start query * " << k << " * ---------------------" << endl;
        for (auto queryPoint : knnQueryPoints)
        {
            vector<array<double, 2> *> result;
            auto start_t = chrono::high_resolution_clock::now();
            cell_tree->kNNSearch(queryPoint, k, result, *exp_Recorder);
            auto end_t = chrono::high_resolution_clock::now();
            timeconsume += chrono::duration_cast<chrono::nanoseconds>(end_t - start_t).count();
        }
        cout << "KNN K: " << k << " time consumption : " << timeconsume / knnQueryPoints.size() << " ns per point query"
             << endl;
        cout << "average range query times: " << exp_Recorder->knnRangeQueryConterAvg / knnQueryPoints.size() << endl;
        cout << "-------------------- end query * " << k << " * ---------------------" << endl;
    }
}

void expInsert()
{
    string csv_path;
    string model_param_path;
    vector<array<double, 2>> insertPoints;
    FileReader *insertFileReader = new FileReader();
    int range_split_num = 100;
    vector<double> data_space_bound;

    data_space_bound = {0, 1, 0, 1};
    csv_path = "";
    model_param_path = "";

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

    long insertTimeconsume = 0;

    insertPoints = insertFileReader->get_array_points("", ",");
    cout << "-------------------- start query insert size * " << 0.4 << " * ---------------------" << endl;
    vector<array<double, 2> *> result;
    int insertidx = 0;

    FileReader *pointQueryFileReader = new FileReader();
    vector<array<double, 2>> queryPoints =
        pointQueryFileReader->get_array_points("", ",");
    cout << "read finish： " << queryPoints.size() << endl;

    string rangeQueryPrefix = "";
    string windowSize = "0.01";
    string aspectRatio = "1";

    vector<vector<double>> range_query =
        pointQueryFileReader->getRangePoints(rangeQueryPrefix + windowSize + "_" + aspectRatio + ".csv", ",");

    for (auto insertPoint : insertPoints)
    {

        auto start_t = chrono::high_resolution_clock::now();
        cell_tree->insert(insertPoint);
        auto end_t = chrono::high_resolution_clock::now();
        insertTimeconsume += chrono::duration_cast<chrono::nanoseconds>(end_t - start_t).count();
        // cout << "insert idx: " << insertidx++ << endl;
        insertidx++;
        if ((insertidx % 10000000) == 0)
        {
            long pointTimeconsume = 0;
            cout << "-------------------- end insert * " << insertidx << " * ---------------------" << endl;

            for (auto &query_point : queryPoints)
            {
                vector<array<double, 2> *> result;
                auto startp_t = chrono::high_resolution_clock::now();
                cell_tree->pointSearch(query_point, result, *exp_Recorder);
                auto endp_t = chrono::high_resolution_clock::now();
                pointTimeconsume += chrono::duration_cast<chrono::nanoseconds>(endp_t - startp_t).count();
            }
            cout << "Point query time consumption : " << pointTimeconsume / queryPoints.size() << "ns per point query"
                 << endl;
            cout << "Insert size: " << insertidx << "time consumption : " << insertTimeconsume / insertidx
                 << " ns per point query" << endl;
            cout << "-------------------- end insert * " << insertidx << " * ---------------------" << endl;

            long rangeTimeconsume = 0;
            cout << "-------------------- end insert * " << insertidx << " * ---------------------" << endl;

            for (auto &rangeQ : range_query)
            {
                vector<array<double, 2> *> result;
                auto startp_t = chrono::high_resolution_clock::now();
                cell_tree->rangeSearch(rangeQ, result, *exp_Recorder);
                auto endp_t = chrono::high_resolution_clock::now();
                rangeTimeconsume += chrono::duration_cast<chrono::nanoseconds>(endp_t - startp_t).count();
            }
            cout << "range query time consumption : " << rangeTimeconsume / range_query.size() << "ns per range query"
                 << endl;
            cout << "Insert size: " << insertidx << "time consumption : " << insertTimeconsume / insertidx
                 << " ns per point query" << endl;
            cout << "-------------------- end insert * " << insertidx << " * ---------------------" << endl;

            // long rangeTimeconsume = 0;
            // cout << "-------------------- end insert * " << insertidx << " * ---------------------" << endl;

            // for (auto &rangeQ : range_query)
            // {
            //     vector<array<double, 2> *> result;
            //     auto startp_t = chrono::high_resolution_clock::now();
            //     cell_tree->rangeSearch(rangeQ, result, *exp_Recorder);
            //     auto endp_t = chrono::high_resolution_clock::now();
            //     rangeTimeconsume += chrono::duration_cast<chrono::nanoseconds>(endp_t - startp_t).count();
            // }
            // cout << "range query time consumption : " << rangeTimeconsume / queryPoints.size() << "ns per range
            // query"
            //      << endl;
            // cout << "Insert size: " << insertidx << "time consumption : " << insertTimeconsume / insertidx
            //      << "ns per point query" << endl;
            // cout << "-------------------- end insert * " << insertidx << " * ---------------------" << endl;
        }
    }

    // cout << "read query data:" << query_path << endl;

    cout << "-------------------- end insert * " << 0.4 << " * ---------------------" << endl;
}

void expRemove()
{
    string csv_path;
    string model_param_path;
    vector<array<double, 2>> removePoints;
    FileReader *insertFileReader = new FileReader();
    int range_split_num = 100;
    vector<double> data_space_bound;

    data_space_bound = {0, 1, 0, 1};
    csv_path = "";
    model_param_path = "";

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

    long insertTimeconsume = 0;
    long pointTimeconsume = 0;

    removePoints = insertFileReader->get_array_points("", ",");
    cout << "-------------------- start query insert size * " << 0.4 << " * ---------------------" << endl;
    vector<array<double, 2> *> result;
    int removeidx = 0;

    FileReader *pointQueryFileReader = new FileReader();
    vector<array<double, 2>> queryPoints =
        pointQueryFileReader->get_array_points("", ",");
    cout << "read finish： " << queryPoints.size() << endl;

    for (auto removePoint : removePoints)
    {

        auto start_t = chrono::high_resolution_clock::now();
        cell_tree->remove(removePoint);
        auto end_t = chrono::high_resolution_clock::now();
        insertTimeconsume += chrono::duration_cast<chrono::nanoseconds>(end_t - start_t).count();
        // cout << "insert idx: " << insertidx++ << endl;
        removeidx++;
        if ((removeidx % 10000000) == 0)
        {
            cout << "-------------------- end insert * " << removeidx << " * ---------------------" << endl;

            for (auto &query_point : queryPoints)
            {
                vector<array<double, 2> *> result;
                auto start_t = chrono::high_resolution_clock::now();
                cell_tree->pointSearch(query_point, result, *exp_Recorder);
                auto end_t = chrono::high_resolution_clock::now();
                pointTimeconsume += chrono::duration_cast<chrono::nanoseconds>(end_t - start_t).count();
            }
            cout << "Point query time consumption : " << pointTimeconsume / queryPoints.size() << "ns per point query"
                 << endl;
            cout << "Insert size: " << removeidx << "time consumption : " << insertTimeconsume / removePoints.size()
                 << "ns per point query" << endl;
            cout << "-------------------- end insert * " << removeidx << " * ---------------------" << endl;
        }
    }

    // cout << "read query data:" << query_path << endl;

    cout << "-------------------- end insert * " << 0.4 << " * ---------------------" << endl;
}

void expAvgHeightPartition(string dataset)
{

    cout << dataset << endl;
    int range_split_num = 100;
    vector<double> data_space_bound;
    string csv_path = "";
    string model_param_path = "";

    if (dataset == "uniform")
    {
        data_space_bound = {0, 1, 0, 1};
        csv_path = "/data/jitao/dataset/uniform/2d_len_1e8_seed_1.csv";
        model_param_path = "/data/jitao/dataset/uniform/initial_param_mono/";
    }
    else if (dataset == "skewed")
    {
        data_space_bound = {0, 1, 0, 1};
        csv_path = "/data/jitao/dataset/skewed/2d_len_1e8_seed_1.csv";
        model_param_path = "/data/jitao/dataset/skewed/initial_param_mono/";
    }
    else if (dataset == "osm_cn")
    {
        data_space_bound = {70.9825433, 142.2560836, 4.999728700, 54.35880621};
        csv_path = "";
        model_param_path = "";
    }
    else if (dataset == "osm_ne_us")
    {
        data_space_bound = {-81.79535869999985, -65.27891709999955, 38.43836500000005, 45.98917950000055};
        csv_path = "/data/jitao/dataset/OSM_US_NE/20_outliers_lon_lat.csv";
        model_param_path = "/data/jitao/dataset/OSM_US_NE/initial_param_mono/";
    }
    else if (dataset == "tiger")
    {
        range_split_num = 40;
        data_space_bound = {-90.3100275, -64.566563, 17.627786999999994, 47.457235};
        csv_path = "/data/jitao/dataset/Tiger/center_tiger_east_17m.txt";
        model_param_path = "/data/jitao/dataset/Tiger/initial_param_mono/";
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
    auto start_time = chrono::high_resolution_clock::now();

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
    long cost_time = chrono::duration_cast<chrono::seconds>(chrono::high_resolution_clock::now() - start_time).count();
    cout << "---time cost of building----" << cost_time << endl;

    double avg = cell_tree->travleAverageHeight();
    cout << "avg: " << avg << endl;
}