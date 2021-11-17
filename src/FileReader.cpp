#include "FileReader.h"

#include <iostream>
#include <fstream>
#include <boost/algorithm/string.hpp>
#include <vector>
#include <iterator>
#include <string>
#include <algorithm>

#include "rapidcsv.h"

// #include "../entities/Point.cpp"
// #include "../entities/Mbr.h"
using namespace std;

FileReader::FileReader()
{
}

FileReader::FileReader(string filename, string delimiter)
{
    this->filename = filename;
    this->delimiter = delimiter;
}

vector<vector<string>> FileReader::get_data(string path)
{
    ifstream file(path);

    vector<vector<string>> data_list;

    string line = "";
    // Iterate through each line and split the content using delimiter
    while (getline(file, line))
    {
        vector<string> vec;
        boost::algorithm::split(vec, line, boost::is_any_of(delimiter));
        data_list.push_back(vec);
    }
    // Close the File
    file.close();

    return data_list;
}

vector<vector<string>> FileReader::get_data()
{
    return get_data(this->filename);
}

vector<vector<double>> FileReader::get_points()
{
    /*  ifstream file(filename);
    vector<vector<double>> points;
    string line = "";
    while (getline(file, line))
    {
        vector<string> vec;
        boost::algorithm::split(vec, line, boost::is_any_of(delimiter));
        if (vec.size() > 1)
        {
            vector<double> point;
            point.push_back(stod(vec[0]));
            point.push_back(stod(vec[1]));
            points.push_back(point);
        }
        cout << "\r point size : " << points.size();
    }*/
    // Close the File
    // file.close();

    vector<vector<double>> points;
    rapidcsv::Document doc(filename, rapidcsv::LabelParams(-1, -1));
    vector<double> x = doc.GetColumn<double>(0);
    // cout << "x size: " << x.size() << endl;
    vector<double> y = doc.GetColumn<double>(1);
    cout << "x size : " << x.size() << " y size : " << y.size() << endl;
    for (int i = 0; i < x.size(); i++)
    {
        vector<double> point = {x[i], y[i]};
        points.push_back(point);
        cout << "\r point size : " << points.size();
    }
    // cout<<endl;
    return points;
}
/*
vector<Mbr> FileReader::get_mbrs()
{
    ifstream file(filename);

    vector<Mbr> mbrs;

    string line = "";
    while (getline(file, line))
    {
        vector<string> vec;
        boost::algorithm::split(vec, line, boost::is_any_of(delimiter));
        Mbr mbr(stod(vec[0]), stod(vec[1]), stod(vec[2]), stod(vec[3]));
        mbrs.push_back(mbr);
    }

    file.close();

    return mbrs;
}
*/
vector<vector<double>> FileReader::get_points(string filename, string delimiter)
{
    ifstream file(filename);
    vector<vector<double>> points;
    string line = "";
    while (getline(file, line))
    {
        vector<string> vec;
        boost::algorithm::split(vec, line, boost::is_any_of(delimiter));
        if (vec.size() > 1)
        {
            vector<double> point;
            point.push_back((double)stod(vec[0]));
            point.push_back((double)stod(vec[1]));
            points.push_back(point);
            // cout << "\r point size : " << points.size();
        }
    }
    // Close the File
    file.close();

    return points;
}

vector<array<double, 2>> FileReader::get_array_points(string filename, string delimiter)
{
    ifstream file(filename);
    vector<array<double, 2>> points;
    string line = "";
    while (getline(file, line))
    {
        vector<string> vec;
        boost::algorithm::split(vec, line, boost::is_any_of(delimiter));
        if (vec.size() > 1)
        {
            array<double, 2> point;
            point[0] = ((double)stod(vec[0]));
            point[1] = ((double)stod(vec[1]));
            points.push_back(point);
            // cout << "\r point size : " << points.size();
        }
    }
    // Close the File
    file.close();

    return points;
}

vector<double> FileReader::get_mapval(string filename, string delimiter)
{
    ifstream file(filename);
    vector<double> mapvalVec;
    string line = "";
    while (getline(file, line))
    {
        vector<string> vec;
        boost::algorithm::split(vec, line, boost::is_any_of(delimiter));
        if (vec.size() > 1)
        {
            mapvalVec.push_back((double)stod(vec[2]));
        }
    }
    // Close the File
    file.close();

    return mapvalVec;
}

vector<vector<double>> FileReader::getRangePoints(string filename, string delimiter)
{
    ifstream file(filename);

    vector<vector<double>> range_query;

    string line = "";
    while (getline(file, line))
    {
        vector<string> vec;
        boost::algorithm::split(vec, line, boost::is_any_of(delimiter));
        vector<double> query = {stod(vec[0]), stod(vec[1]), stod(vec[2]), stod(vec[3])};
        range_query.push_back(query);
    }

    file.close();

    return range_query;
}


vector<double> FileReader::getModelParameter(string paramPath)
{
    // rapidcsv::Document doc(paramPath, rapidcsv::LabelParams(-1, -1));
    // vector<double> paramVec = doc.GetColumn<double>(0);
    vector<double> paramVec;
    ifstream file(paramPath);
    string line = "";
    while (getline(file, line))
    {
        paramVec.push_back((double)stod(line));
    }
    return paramVec;
}