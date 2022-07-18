#include "CellTree.h"
// #include "Core/Matrix.h"
// #include <chrono>
using namespace std;

int split_num = 100;

string windowSizeArray[5] = {"0.0001", "0.0025", "0.01", "0.04", "0.16"};
string aspectRadioArray[5] = {"0.25", "0.5", "1", "2", "4"};

string insertSizeArray[4] = {"0.1", "0.2", "0.3", "0.4"};

void expRemove();
void expInsert();

int main(int argc, char *argv[])
{

    expInsert();
    cout << "finish!" << endl;
}

void expRemove()
{
    string csv_path;
    string model_param_path;
    vector<array<double, 2>> removePoints;
    FileReader *insertFileReader = new FileReader();
    int range_split_num = 100;

    vector<double> data_space_bound = {0, 1, 0, 1};
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

    long removeTimeconsume = 0;

    removePoints = insertFileReader->get_array_points("", ",");
    cout << "-------------------- start query remove size * " << 0.4 << " * ---------------------" << endl;
    cout << "-------------------- start query remove size * " << removePoints.size() << " * ---------------------"
         << endl;
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
        removeTimeconsume += chrono::duration_cast<chrono::nanoseconds>(end_t - start_t).count();
        // cout << "insert idx: " << insertidx++ << endl;
        removeidx++;
        if ((removeidx % 10000000) == 0)
        {
            cout << "-------------------- end remove * " << removeidx << " * ---------------------" << endl;

            long pointTimeconsume = 0;
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
            cout << "remove size: " << removeidx << "time consumption : " << removeTimeconsume / removeidx
                 << "ns per point query" << endl;
            cout << "-------------------- end remove * " << removeidx << " * ---------------------" << endl;
        }
    }

    // cout << "read query data:" << query_path << endl;

    cout << "-------------------- end remove * " << 0.4 << " * ---------------------" << endl;
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

    long insertTimeconsume = 0;

    insertPoints = insertFileReader->get_array_points("/data/jitao/dataset/skewed/insert/2d_len_0.4_seed_1.csv", ",");
    // insertPoints =
    // insertFileReader->get_array_points("/data/jitao/dataset/skewed/insert/uniform_2d_len_4e7_seed_12333.csv", ",");

    cout << "-------------------- start query insert size * " << 0.4 << " * ---------------------" << endl;
    int insertidx = 0;

    FileReader *pointQueryFileReader = new FileReader();
    vector<array<double, 2>> queryPoints =
        pointQueryFileReader->get_array_points("/data/jitao/dataset/skewed/point_query_sample_10w.csv", ",");
    cout << "read finish： " << queryPoints.size() << endl;

    string rangeQueryPrefix = "/data/jitao/dataset/skewed/range_query/2d_len_1e8_seed_1_1000_";
    string windowSize = "0.01";
    string aspectRatio = "1";

    vector<vector<double>> range_query =
        pointQueryFileReader->getRangePoints(rangeQueryPrefix + windowSize + "_" + aspectRatio + ".csv", ",");

    vector<array<double, 2>> knnQueryPoints;
    for (int i = 0; i < 1000; i++)
    {
        knnQueryPoints.push_back(queryPoints[i]);
    }

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
            for (int cache_miss = 0; cache_miss < 2; cache_miss++)
            {
                ExpRecorder *exp_Recorder = new ExpRecorder();

                for (auto &query_point : queryPoints)
                {
                    vector<array<double, 2> *> result;
                    auto startp_t = chrono::high_resolution_clock::now();
                    cell_tree->pointSearch(query_point, result, *exp_Recorder);
                    auto endp_t = chrono::high_resolution_clock::now();
                    pointTimeconsume += chrono::duration_cast<chrono::nanoseconds>(endp_t - startp_t).count();
                }
                cout << "Point query time consumption : " << pointTimeconsume / queryPoints.size()
                     << " ns per point query " << endl;
                cout << "Insert size: " << insertidx << "time consumption : " << insertTimeconsume / insertidx
                     << " ns per point query" << endl;
                cout << "-------------------- end insert * " << insertidx << " * ---------------------" << endl;
            }

            for (int cache_miss = 0; cache_miss < 2; cache_miss++)
            {
                long rangeTimeconsume = 0;
                cout << "-------------------- end insert * " << insertidx << " * ---------------------" << endl;
                ExpRecorder *exp_Recorder = new ExpRecorder();

                for (auto &rangeQ : range_query)
                {
                    vector<array<double, 2> *> result;
                    auto startp_t = chrono::high_resolution_clock::now();
                    cell_tree->rangeSearch(rangeQ, result, *exp_Recorder);
                    auto endp_t = chrono::high_resolution_clock::now();
                    rangeTimeconsume += chrono::duration_cast<chrono::nanoseconds>(endp_t - startp_t).count();
                }
                cout << "range query time consumption : " << rangeTimeconsume / range_query.size()
                     << " ns per range query " << endl;
                cout << "Insert size: " << insertidx << "time consumption : " << insertTimeconsume / insertidx
                     << " ns per point query" << endl;
                cout << "-------------------- end insert * " << insertidx << " * ---------------------" << endl;
            }
            for (int cache_miss = 0; cache_miss < 2; cache_miss++)
            {
                ExpRecorder *exp_Recorder = new ExpRecorder();

                long knnTimeconsume = 0;
                exp_Recorder->knnRangeQueryConterAvg = 0;
                for (auto queryPoint : knnQueryPoints)
                {
                    vector<array<double, 2> *> result;
                    auto start_t = chrono::high_resolution_clock::now();
                    cell_tree->kNNSearch(queryPoint, 25, result, *exp_Recorder);
                    auto end_t = chrono::high_resolution_clock::now();
                    knnTimeconsume += chrono::duration_cast<chrono::nanoseconds>(end_t - start_t).count();
                }
                cout << "KNN K: " << 25 << " time consumption : " << knnTimeconsume / knnQueryPoints.size()
                     << " ns per knn query" << endl;
                cout << "average knn query times: " << exp_Recorder->knnRangeQueryConterAvg / knnQueryPoints.size()
                     << endl;
                cout << "-------------------- end query * " << 25 << " * ---------------------" << endl;
            }

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
