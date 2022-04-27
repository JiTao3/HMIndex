#include "Utils.h"

using namespace std;

bool equalMetadata(MetaData &me_data1, MetaData &me_data2)
{
    if (me_data1.map_val != me_data2.map_val)
    {
        return false;
    }
    else
    {
        for (int i = 0; i < MetaData::dim; i++)
        {
            if (me_data1.data[i] != me_data2.data[i])
            {
                return false;
            }
        }
    }
    return true;
}

double percentile(vector<double> &vectorIn, double percent, bool is_sorted)
{
    if (!is_sorted)
        sort(vectorIn.begin(), vectorIn.end());
    double res = 0.0;
    double x = (vectorIn.size() - 1) * percent;
    int i = (int)x;
    double j = i - x;
    if (i == (vectorIn.size() - 1))
    {
        return vectorIn[i];
    }
    res = (1 - j) * vectorIn[i] + j * vectorIn[i + 1];
    return res;
}

// template std::vector<double> LinSpace(double start_in, double end_in, int num_in);

vector<int> idx_to_vector_int(int idx, int dim)
{
    vector<int> vector_int;
    for (int i = 0; i < dim; i++)
    {
        if (idx % 2 == 0)
        {
            vector_int.push_back(0);
        }
        else
        {
            vector_int.push_back(1);
        }
        idx /= 2;
    }
    return vector_int;
}

bool match_range(array<double, 2> point_data, vector<double> cell_bound, int dim)
{
    for (int i = 0; i < dim; i++)
    {
        if (point_data[i] < cell_bound[i * 2] || point_data[i] >= cell_bound[i * 2 + 1])
        {
            return false;
        }
    }
    return true;
}

bool match_region(const MetaData &medata, vector<double> split_point, vector<int> left_or_right)
{
    bool match_flag = true;
    for (int i = 0; i < MetaData::dim; i++)
    {
        if (left_or_right[i] == 0)
        {
            match_flag = (match_flag && ((*medata.data)[i] <= split_point[i]));
        }
        else if (left_or_right[i] == 1)
        {
            match_flag = (match_flag && ((*medata.data)[i] > split_point[i]));
        }
        if (!match_flag)
        {
            return false;
        }
    }
    return true;
}

vector<MetaData> splitLeafNodeToLeaf(vector<MetaData> &metadatas, vector<double> split_point, int index)
{
    vector<int> data_left_or_right = idx_to_vector_int(index, split_point.size());
    vector<MetaData> medataVec;
    for (auto &medata : metadatas)
    {
        if (match_region(medata, split_point, data_left_or_right))
        {
            medataVec.push_back(medata);
        }
    }
    return medataVec;
}

vector<MetaData> splitLeafNodeToGrid(vector<MetaData> &metadatas, vector<double> range_bound)
{
    vector<MetaData> metadataVec;
    for (auto &medata : metadatas)
    {
        if (match_range(*medata.data, range_bound, MetaData::dim))
        {
            metadataVec.push_back(medata);
        }
    }
    return metadataVec;
}

vector<double> splitNodeToSplitedRegion(vector<double> cell_range_bound, vector<double> split_point, int index)
{
    vector<int> data_left_or_right = idx_to_vector_int(index, split_point.size());
    vector<double> split_region;
    for (int i = 0; i < MetaData::dim; i++)
    {
        double min_b = cell_range_bound[i * 2];
        double max_b = cell_range_bound[i * 2 + 1];
        if (data_left_or_right[i] == 0)
        {
            split_region.push_back(min_b);
            split_region.push_back(split_point[i]);
        }
        else
        {
            split_region.push_back(split_point[i]);
            split_region.push_back(max_b);
        }
    }
    return split_region;
}

vector<double> splitLeafNodeToGridRegion(vector<double> cell_range_bound, vector<double> split_point, int split_dim,
                                         int child_index)
{
    vector<double> split_region;
    for (int i = 0; i < MetaData::dim; i++)
    {
        if (i == split_dim)
        {
            split_region.push_back(split_point[child_index]);
            split_region.push_back(split_point[child_index + 1]);
        }
        else
        {
            split_region.push_back(cell_range_bound[i * 2]);
            split_region.push_back(cell_range_bound[i * 2 + 1]);
        }
    }
    return split_region;
}

vector<array<double, 2> *> &bindary_search(vector<MetaData> &metadataVec, bitset<BITMAP_SIZE> &bitmap, int begin_idx,
                                           int end_idx, MetaData &meta_key, std::vector<array<double, 2> *> &result,
                                           ExpRecorder &expr)
{
    // auto start_bin = chrono::high_resolution_clock::now();

    int start = begin_idx;
    int end = end_idx;
    int mid = -1;
    while (start <= end)
    {
        mid = (end - start) / 2 + start;
        if (equalMetadata(metadataVec[mid], meta_key))
        {
            if (bitmap[mid])
                result.push_back(metadataVec[mid].data);
            break;
        }
        else if (meta_key.map_val < metadataVec[mid].map_val)
        {
            end = mid - 1;
        }
        else
        {
            start = mid + 1;
        }
    }

    int left = mid - 1;
    int right = mid + 1;
    while (left >= begin_idx && metadataVec[left].map_val == meta_key.map_val)
    {
        if (equalMetadata(metadataVec[left], meta_key) && bitmap[left])
        {
            result.push_back(metadataVec[left].data);
        }
        left--;
    }
    while (right <= end_idx && metadataVec[right].map_val == meta_key.map_val)
    {
        if (equalMetadata(metadataVec[right], meta_key) && bitmap[right])
        {
            result.push_back(metadataVec[right].data);
        }
        right++;
    }
    // auto end_bin = chrono::high_resolution_clock::now();
    // expr.pointBindarySearchTime +=chrono::duration_cast<chrono::nanoseconds>(end_bin - start_bin).count();

    return result;
}

vector<array<double, 2> *> &bindary_search(array<MetaData, INSERT_BUFFERSIZE> &buffer, int begin_idx, int end_idx,
                                           MetaData &meta_key, std::vector<array<double, 2> *> &result)
{
    int start = begin_idx;
    int end = end_idx;
    int mid = -1;
    while (start <= end)
    {
        mid = (end - start) / 2 + start;

        if (equalMetadata(buffer[mid], meta_key))
        {
            result.push_back(buffer[mid].data);
        }
        else if (meta_key.map_val < buffer[mid].map_val)
        {
            end = mid - 1;
        }
        else
        {
            start = mid + 1;
        }
    }

    int left = mid - 1;
    int right = mid + 1;
    while (left >= begin_idx && buffer[left].map_val == meta_key.map_val)
    {
        if (equalMetadata(buffer[left], meta_key))
        {
            result.push_back(buffer[left].data);
        }
        left--;
    }
    while (right <= end_idx && buffer[right].map_val == meta_key.map_val)
    {
        if (equalMetadata(buffer[right], meta_key))
        {
            result.push_back(buffer[right].data);
        }
        right++;
    }
    // auto end_bin = chrono::high_resolution_clock::now();
    // expr.pointBindarySearchTime +=chrono::duration_cast<chrono::nanoseconds>(end_bin - start_bin).count();

    return result;
}

int bindarySearchAdjustPrePos(vector<MetaData> &metadataVec, int beginIndex, int endIndex, MetaData &meta_key,
                              int LeftORRight)
{
    // leftORRight > 0 👉
    // leftORRight < 0 👈
    // int adjustPrePos = -1;

    int begin = beginIndex;
    int end = endIndex;
    if (beginIndex == endIndex)
        return beginIndex;
    int mid = -1;
    while (begin <= end)
    {
        mid = (end - begin) / 2 + begin;
        if (metadataVec[mid].map_val == meta_key.map_val)
            break;
        else if (meta_key.map_val < metadataVec[mid].map_val)
            end = mid - 1;
        else
            begin = mid + 1;
    }
    if (LeftORRight < 0)
    {
        while (mid > beginIndex && metadataVec[mid].map_val == meta_key.map_val)
        {
            mid--;
        }
        return mid;
    }
    else
    {
        while (mid < endIndex && metadataVec[mid].map_val == meta_key.map_val)
        {
            mid++;
        }
        return mid;
    }
}

int adjustPosition(vector<MetaData> &metadataVec, vector<int> &error_bound, int pre_position, MetaData meta_key,
                   int leftORright)
{
    if (metadataVec[pre_position].map_val == meta_key.map_val)
    {
        // > 0 👉
        // < 0 👈
        // upper bound move left
        // lower bound move right
        if (leftORright > 0)
        {
            while (metadataVec[pre_position].map_val == meta_key.map_val && pre_position < (metadataVec.size() - 1))
                pre_position++;
        }
        else
        {
            while (metadataVec[pre_position].map_val == meta_key.map_val && pre_position > 0)
                pre_position--;
        }
        return pre_position;
    }
    else if (metadataVec[pre_position].map_val > meta_key.map_val)
    {
        int min_position = pre_position + error_bound[0] < 0 ? 0 : pre_position + error_bound[0];
        return bindarySearchAdjustPrePos(metadataVec, min_position, pre_position, meta_key, leftORright);
    }
    else
    {
        int max_position = pre_position + error_bound[1] > metadataVec.size() - 1 ? metadataVec.size() - 1
                                                                                  : pre_position + error_bound[1];
        return bindarySearchAdjustPrePos(metadataVec, pre_position, max_position, meta_key, leftORright);
    }
    // return pre_position;
}

int knnbindarySearchAdjustPrePos(vector<MetaData> &metadataVec, int beginIndex, int endIndex, MetaData &meta_key)
{
    int begin = beginIndex;
    int end = endIndex;
    int mid = -1;
    if (begin == end)
        return beginIndex;
    while (begin <= end)
    {
        mid = (end - begin) / 2 + begin;

        if (metadataVec[mid].map_val == meta_key.map_val)
            break;
        else if (meta_key.map_val < metadataVec[mid].map_val)
            end = mid - 1;
        else
            begin = mid + 1;
    }
    return mid;
}

int knnAdjustPosition(vector<MetaData> &metadataVec, vector<int> &error_bound, int pre_position, MetaData meta_key)
{
    if (metadataVec[pre_position].map_val == meta_key.map_val)
    {
        // > 0 👉
        // < 0 👈
        return pre_position;
    }
    else if (metadataVec[pre_position].map_val > meta_key.map_val)
    {
        // std::vector<int> offsets = bindary_search(metadataVec, pre_position + error_bound[0], pre_position,
        // meta_key); pre_position = *std::max_element(offsets.begin(), offsets.end());
        int min_position = pre_position + error_bound[0] < 0 ? 0 : pre_position + error_bound[0];
        // if (min_position == pre_position)
        //     return pre_position;
        // else
        return knnbindarySearchAdjustPrePos(metadataVec, min_position, pre_position, meta_key);
    }
    else
    {
        // std::vector<int> offsets = bindary_search(metadataVec, pre_position, pre_position + error_bound[0],
        // meta_key); pre_position = *std::min_element(offsets.begin(), offsets.end());
        int max_position = pre_position + error_bound[1] > metadataVec.size() - 1 ? metadataVec.size() - 1
                                                                                  : pre_position + error_bound[1];
        // if (max_position == pre_position)
        //     return pre_position;
        // else
        return knnbindarySearchAdjustPrePos(metadataVec, pre_position, max_position, meta_key);
    }
}
void scan(vector<MetaData> &metadataVec, int begin, int end, double *min_range, double *max_range,
          vector<array<double, 2> *> &result)
{
    for (int i = begin; i <= end; i++)
    {
        if (min_range[0] > (*(metadataVec[i].data))[0] || (*(metadataVec[i].data))[0] > max_range[0] ||
            min_range[1] > (*(metadataVec[i].data))[1] || (*(metadataVec[i].data))[1] > max_range[1])
        {
            continue;
        }
        else
        {
            result.push_back(metadataVec[i].data);
        }

        // bool is_in = true;
        // for (int idx = 0; idx < MetaData::dim; idx++)
        // {
        //     if (!((*(metadataVec[i].data))[idx] >= min_range[idx] && (*(metadataVec[i].data))[idx] <=
        //     max_range[idx]))
        //     {
        //         is_in = false;
        //         break;
        //     }
        // }
        // if (is_in)
        // {
        //     result.push_back(metadataVec[i].data);
        // }
    }
}

vector<int> getCellIndex(array<double, 2> &raData, vector<double> &initial_partition_bound_x,
                         vector<double> &initial_partition_bound_y)
{
    vector<int> cell_index;
    for (int j = 0; j < initial_partition_bound_x.size(); j++)
    {
        if (initial_partition_bound_x[j] > raData[0])
        {
            cell_index.push_back(j - 1);
            break;
        }
    }
    for (int j = 0; j < initial_partition_bound_y.size(); j++)
    {
        if (initial_partition_bound_y[j] > raData[1])
        {
            cell_index.push_back(j - 1);
            break;
        }
    }

    return cell_index;
}

bool compareMetadata(MetaData me_data1, MetaData me_data2)
{
    return me_data1.map_val < me_data2.map_val;
}

void orderMetaData(vector<MetaData> &metadataVec)
{
    std::sort(metadataVec.begin(), metadataVec.end(), compareMetadata);
}

int queryCellRealtion(vector<double> &rangeBound, vector<double> &query)
{
    // bool overlapFlag = OVERLAP;
    float dimxMax = rangeBound[1] > query[1] ? query[1] : rangeBound[1];
    float dimxMin = rangeBound[0] > query[0] ? rangeBound[0] : query[0];

    float dimyMax = rangeBound[3] > query[3] ? query[3] : rangeBound[3];
    float dimyMin = rangeBound[2] > query[2] ? rangeBound[2] : query[2];

    if (dimxMax < dimxMin || dimyMax < dimyMin)
        return DISSOCIATION;

    else if ((query[0] < rangeBound[0] && query[1] > rangeBound[1]) &&
             (query[2] < rangeBound[2] && query[3] > rangeBound[3]))
        return CONTAIN;
    else
        return INTERSECTION;
    // for (int i = 0; i < MetaData::dim; i++)
    // {
    //     // min(node max and range max)
    //     float dimMax = rangeBound[2 * i + 1] > query[2 * i + 1] ? query[2 * i + 1] : rangeBound[2 * i + 1];
    //     // max(node min and range min)
    //     float dimMin = rangeBound[2 * i] > query[2 * i] ? rangeBound[2 * i] : query[2 * i];
    //     if (dimMax < dimMin)
    //         return DISSOCIATION;
    // }
    //     for (int i = 0; i < MetaData::dim; i++)
    //     {
    //         if (!(query[i * 2] < rangeBound[i * 2] && query[i * 2 + 1] > rangeBound[i * 2 + 1]))
    //             return INTERSECTION;
    //     }
    // return CONTAIN;
}

double distFunction(array<double, 2> *point1, double x, double y)
{
    return pow(((*point1)[0] - x), 2) + pow(((*point1)[1] - y), 2);
}

bool insertMetadataInRange(vector<MetaData> &metadataVec, bitset<BITMAP_SIZE> &bitmap, int begin_idx, int end_idx,
                           MetaData &meta_key)
{
    int idx = begin_idx;
    while (metadataVec[idx].map_val < meta_key.map_val && idx <= end_idx)
        idx++;
    idx--;
    if (idx < 0)
        return false;
    if (bitmap[idx] == false)
    {
        metadataVec[idx] = meta_key;
        bitmap[idx] = 1;
        return true;
    }
    return false;
}

bool deleteMetadataInRange(vector<MetaData> &metadataVec, bitset<BITMAP_SIZE> &bitmap, int begin_idx, int end_idx,
                           MetaData &deleteMetadata)
{
    int start = begin_idx;
    int end = end_idx;
    int mid = -1;
    bool deleteFlag = false;
    while (start <= end)
    {
        mid = (end - start) / 2 + start;
        if (equalMetadata(metadataVec[mid], deleteMetadata))
        {
            bitmap[mid] = 0;
            deleteFlag = true;
            break;
        }
        else if (deleteMetadata.map_val < metadataVec[mid].map_val)
        {
            end = mid - 1;
        }
        else
        {
            start = mid + 1;
        }
    }

    int left = mid - 1;
    int right = mid + 1;
    while (left >= begin_idx && metadataVec[left].map_val == deleteMetadata.map_val)
    {
        if (equalMetadata(metadataVec[left], deleteMetadata))
        {
            bitmap[mid] = 0;
            deleteFlag = true;
        }
        left--;
    }
    while (right <= end_idx && metadataVec[right].map_val == deleteMetadata.map_val)
    {
        if (equalMetadata(metadataVec[right], deleteMetadata))
        {
            bitmap[mid] = 0;
            deleteFlag = true;
        }
        right++;
    }
    return deleteFlag;
}

void scanBuffer(array<MetaData, INSERT_BUFFERSIZE> &insertBuffer, int bufferDataSize, double *min_range,
                double *max_range, vector<array<double, 2> *> &result)
{
    for (int i = 0; i < bufferDataSize; i++)
    {
        if (min_range[0] > (*(insertBuffer[i].data))[0] || (*(insertBuffer[i].data))[0] > max_range[0] ||
            min_range[1] > (*(insertBuffer[i].data))[1] || (*(insertBuffer[i].data))[1] > max_range[1])
        {
            continue;
        }
        else
        {
            result.push_back(insertBuffer[i].data);
        }
        // bool is_in = true;
        // for (int idx = 0; idx < MetaData::dim; idx++)
        // {
        //     if (!((*(insertBuffer[i].data))[idx] >= min_range[idx] && (*(insertBuffer[i].data))[idx] <=
        //     max_range[idx]))
        //     {
        //         is_in = false;
        //         break;
        //     }
        // }
        // if (is_in)
        // {
        //     result.push_back(insertBuffer[i].data);
        // }
    }
}
