//#include <vector>
#include "LeafNode.h"

using namespace std;

LeafNode::LeafNode()
{
}

LeafNode::LeafNode(std::vector<MetaData> &_metadatas, std::vector<double> _range_bound)
{
    // memset(modelError, 0, sizeof(double) * 2);
    metadataVec = _metadatas;
    dim = MetaData::dim;
    range_bound = _range_bound;
    getCellArea();
    setMapVals();
    orderMetaData(this->metadataVec);
    if (_metadatas.size() > 0)
    {
        mapvalBound[0] = metadataVec[0].map_val;
        mapvalBound[1] = metadataVec[metadataVec.size() - 1].map_val;
    }
    else
    {
        mapvalBound[0] = 0.0;
        mapvalBound[1] = 0.0;
    }
    this->initialBitMap();
    index_model = new IndexModel(this->metadataVec);
    // index_model->buildModel();
}

LeafNode::~LeafNode()
{
    if (index_model != nullptr)
    {
        delete index_model;
    }
}

std::vector<double> LeafNode::getRangeBound()
{
    // wast of mem
    // inaccurate?
    for (int dim_idx = 0; dim_idx < dim; dim_idx++)
    {
        std::vector<double> dim_data;
        for (auto &me_data : metadataVec)
        {
            dim_data.push_back((*me_data.data)[dim_idx]);
        }
        double min_d = *std::min_element(dim_data.begin(), dim_data.end());
        double max_d = *std::max_element(dim_data.begin(), dim_data.end());
        range_bound.push_back(min_d);
        range_bound.push_back(max_d);
    }
    return range_bound;
}

double LeafNode::getCellArea()
{
    this->cell_area = 1.0;
    for (int i = 0; i < dim; i++)
    {
        cell_area *= (range_bound[i * 2 + 1] - range_bound[i * 2]);
    }
    return cell_area;
}

void LeafNode::setMapVals()
{
    if (cell_area <= 0)
    {
        cout << "cell area error!" << cell_area << endl;
    }
    for (auto &me_data : metadataVec)
    {
        me_data.setMapVal(range_bound, cell_area);
    }
}

vector<array<double, 2> *> &LeafNode::pointSearch(array<double, 2> key, std::vector<array<double, 2> *> &result,
                                                  ExpRecorder &exp_Recorder)
{

    if (this->getKeysNum() <= 1)
        return result;
    MetaData meta_key(&key);
    meta_key.setMapVal(range_bound, cell_area);
    int pre_position = index_model->preFastPosition(meta_key.map_val);

    pre_position = pre_position > metadataVec.size() - 1 ? metadataVec.size() - 1 : pre_position;
    pre_position = pre_position > 0 ? pre_position : 0;

    double pre_map_val = metadataVec[pre_position].map_val;
    if (pre_map_val > meta_key.map_val)
    {
        int min_search_index =
            pre_position + index_model->error_bound[0] > 0 ? pre_position + index_model->error_bound[0] : 0;
        bindary_search(metadataVec, metadataVecBitMap, min_search_index, pre_position, meta_key, result, exp_Recorder);
        if (bufferDataSize > 0)
            bindary_search(this->insertBuffer, 0, bufferDataSize, meta_key, result);
        return result;
    }
    else
    {
        int max_search_index = pre_position + index_model->error_bound[1] > metadataVec.size() - 1
                                   ? metadataVec.size() - 1
                                   : pre_position + index_model->error_bound[1];
        bindary_search(metadataVec, metadataVecBitMap, pre_position, max_search_index, meta_key, result, exp_Recorder);
        if (bufferDataSize > 0)
            bindary_search(this->insertBuffer, 0, bufferDataSize, meta_key, result);
        return result;
    }
}

vector<array<double, 2> *> &LeafNode::rangeSearch(std::vector<double> query_range, vector<array<double, 2> *> &result,
                                                  ExpRecorder &exp_Recorder)
{
    if (this->getKeysNum() <= 2)
        return result;
    auto start_refine = chrono::high_resolution_clock::now();

    double min_range[2] = {query_range[0], query_range[2]};
    double max_range[2] = {query_range[1], query_range[3]};

    double node_min[2] = {range_bound[0], range_bound[2]};
    double node_max[2] = {range_bound[1], range_bound[3]};

    array<double, MetaData::dim> overlap_min;
    array<double, MetaData::dim> overlap_max;
    for (int i = 0; i < MetaData::dim; i++)
    {
        if (node_min[i] > min_range[i] && node_max[i] < max_range[i])
        {
            overlap_min[i] = node_min[i];
            overlap_max[i] = node_max[i];
        }
        else if (node_min[i] < min_range[i] && node_max[i] > max_range[i])
        {
            overlap_min[i] = min_range[i];
            overlap_max[i] = max_range[i];
        }
        else if (node_min[i] > min_range[i])
        {
            overlap_min[i] = node_min[i];
            overlap_max[i] = max_range[i];
        }
        else if (node_max[i] > min_range[i])
        {
            overlap_min[i] = min_range[i];
            overlap_max[i] = node_max[i];
        }
    }
    MetaData meta_min(&overlap_min);
    meta_min.setMapVal(range_bound, cell_area);
    MetaData meta_max(&overlap_max);
    meta_max.setMapVal(range_bound, cell_area);
    int pre_min_position = index_model->preFastPosition(meta_min.map_val);
    int pre_max_position = index_model->preFastPosition(meta_max.map_val);

    pre_min_position = pre_min_position > 0 ? pre_min_position : 0;
    pre_max_position = pre_max_position > 0 ? pre_max_position : 0;

    pre_min_position = pre_min_position > metadataVec.size() - 1 ? metadataVec.size() - 1 : pre_min_position;
    pre_max_position = pre_max_position > metadataVec.size() - 1 ? metadataVec.size() - 1 : pre_max_position;

    pre_min_position = adjustPosition(metadataVec, index_model->error_bound, pre_min_position, meta_min, -1);
    pre_max_position = adjustPosition(metadataVec, index_model->error_bound, pre_max_position, meta_max, 1);

    auto end_refine = chrono::high_resolution_clock::now();
    exp_Recorder.rangeRefinementTime += chrono::duration_cast<chrono::nanoseconds>(end_refine - start_refine).count();

    auto start_scan = chrono::high_resolution_clock::now();
    scan(metadataVec, pre_min_position, pre_max_position, min_range, max_range, result);
    // ! scanBuffer
    if (bufferDataSize > 0)
    {
        if (!this->bufferOrdered)
        {
            std::sort(insertBuffer.begin(), insertBuffer.begin() + bufferDataSize, compareMetadata);
            this->bufferOrdered = true;
        }

        scanBuffer(insertBuffer, bufferDataSize, min_range, max_range, result);
    }
    auto end_scan = chrono::high_resolution_clock::now();
    exp_Recorder.rangeScanTime += chrono::duration_cast<chrono::nanoseconds>(end_scan - start_scan).count();

    return result;
}

void LeafNode::saveMetaDataVectoCSV(string file_path)
{
    std::ofstream outfile(file_path);
    for (auto &medata : this->metadataVec)
    {
        for (auto i : *medata.data)
        {
            outfile << i << ',';
        }
        outfile << medata.map_val;
        outfile << '\n';
    }
}

void LeafNode::initialBitMap()
{
    this->metadataVecBitMap.reset();
    for (int i = 0; i < metadataVec.size() && i < BITMAP_SIZE; i++)
    {
        this->metadataVecBitMap[i] = 1;
    }
}

bool LeafNode::insert(array<double, 2> &point)
{
    // return ture if merge then in cell tree check the leaf node
    // return false if not merge.
    // // TODO : check can insert into the origin vector?

    MetaData insertMetadata(&point);
    insertMetadata.setMapVal(this->range_bound, this->cell_area);

    int pre_position = index_model->preFastPosition(insertMetadata.map_val);

    pre_position = pre_position > metadataVec.size() - 1 ? metadataVec.size() - 1 : pre_position;
    pre_position = pre_position > 0 ? pre_position : 0;

    double pre_map_val = metadataVec[pre_position].map_val;

    bool bindaryInsert = false;

    if (pre_map_val > insertMetadata.map_val)
    {
        int min_search_index =
            pre_position + index_model->error_bound[0] > 0 ? pre_position + index_model->error_bound[0] : 0;
        bindaryInsert =
            insertMetadataInRange(metadataVec, metadataVecBitMap, min_search_index, pre_position, insertMetadata);
    }
    else
    {
        int max_search_position = pre_position + index_model->error_bound[1] > metadataVec.size() - 1
                                      ? metadataVec.size() - 1
                                      : pre_position + index_model->error_bound[1];
        // return bindary_search(metadataVec, metadataVecBitMap, pre_position, max_search_position, meta_key, result,
        //                       insertMetadata);
        bindaryInsert =
            insertMetadataInRange(metadataVec, metadataVecBitMap, pre_position, max_search_position, insertMetadata);
    }
    bool mergeFlag = false;
    if (!bindaryInsert)
    {
        if (bufferDataSize < INSERT_BUFFERSIZE)
        {
            insertBuffer[bufferDataSize] = insertMetadata;
            bufferDataSize++;
            this->bufferOrdered = false;
            // std::sort(insertBuffer.begin(), insertBuffer.begin() + bufferDataSize, compareMetadata);
        }
        else
        {
            // merge the data into metadataVec
            // cout << "before merge : " << metadataVec.size() << " ";
            for (int i = 0; i < bufferDataSize; i++)
            {
                metadataVec.push_back(insertBuffer[i]);
            }
            std::sort(metadataVec.begin(), metadataVec.end(), compareMetadata);
            // cout << "after merge : " << metadataVec.size() << endl;

            mergeFlag = true;
            bufferDataSize = 0;
        }
    }
    return mergeFlag;
}

bool LeafNode::remove(array<double, 2> &point)
{
    bool deleteFlag = false;
    MetaData deleteMetadata(&point);
    deleteMetadata.setMapVal(this->range_bound, this->cell_area);
    int prePosition = this->index_model->preFastPosition(deleteMetadata.map_val);

    if (metadataVec[prePosition].map_val > deleteMetadata.map_val)
    {
        int min_search_index =
            prePosition + index_model->error_bound[0] > 0 ? prePosition + index_model->error_bound[0] : 0;
        deleteFlag =
            deleteMetadataInRange(metadataVec, metadataVecBitMap, min_search_index, prePosition, deleteMetadata);
    }
    else
    {
        int max_search_position = prePosition + index_model->error_bound[1] > metadataVec.size() - 1
                                      ? metadataVec.size() - 1
                                      : prePosition + index_model->error_bound[1];
        deleteFlag =
            deleteMetadataInRange(metadataVec, metadataVecBitMap, prePosition, max_search_position, deleteMetadata);
    }
    int deleteNumInBuffer = 0;
    for (int i = 0; i < bufferDataSize; i++)
    {
        if (equalMetadata(insertBuffer[i], deleteMetadata))
        {
            insertBuffer[i].map_val = numeric_limits<double>::max();
            insertBuffer[i].data = nullptr;
            deleteNumInBuffer++;
            deleteFlag = true;
        }
    }
    if (deleteNumInBuffer > 0)
    {
        std::sort(insertBuffer.begin(), insertBuffer.begin() + bufferDataSize, compareMetadata);
        this->bufferDataSize -= deleteNumInBuffer;
    }
    return deleteFlag;
}

int LeafNode::getKeysNum()
{
    return this->metadataVecBitMap.count() + bufferDataSize;
}

void LeafNode::retrainModel()
{
    // vector<int> preErrorBound = this->index_model->error_bound;
    int pre_lowerror = this->index_model->error_bound[0];
    int pre_upperror = this->index_model->error_bound[1];

    this->index_model->refreshMetaDataVec(this->metadataVec);
    this->index_model->getErrorBound();

    int low_error_change = std::abs(this->index_model->error_bound[0] - pre_lowerror);
    int upper_error_change = std::abs(this->index_model->error_bound[1] - pre_upperror);

    // cout << "befor merge: " << pre_lowerror << " --- " << pre_upperror << endl;
    // cout << "after merge: " << this->index_model->error_bound[0] << " --- " << this->index_model->error_bound[1]
    //      << endl;
    //! error bound reach the setting threshold, then retrain the model
    if (low_error_change > 500 && upper_error_change > 500)
        this->index_model->getParamFromScoket(12333, this->metadataVec);
    // cout << "after training: " << this->index_model->error_bound[0] << " --- " << this->index_model->error_bound[1]
    //      << endl
    //      << endl;
}

void LeafNode::kNNInNode(std::vector<double> query_range,
                         priority_queue<array<double, 2> *, vector<array<double, 2> *>, sortForKNN> &temp_result)
{

    if (this->getKeysNum() <= 1)
        return;

    double min_range[2] = {query_range[0], query_range[2]};
    double max_range[2] = {query_range[1], query_range[3]};

    double node_min[2] = {range_bound[0], range_bound[2]};
    double node_max[2] = {range_bound[1], range_bound[3]};

    array<double, MetaData::dim> overlap_min;
    array<double, MetaData::dim> overlap_max;
    for (int i = 0; i < MetaData::dim; i++)
    {
        if (node_min[i] > min_range[i] && node_max[i] < max_range[i])
        {
            overlap_min[i] = node_min[i];
            overlap_max[i] = node_max[i];
        }
        else if (node_min[i] < min_range[i] && node_max[i] > max_range[i])
        {
            overlap_min[i] = min_range[i];
            overlap_max[i] = max_range[i];
        }
        else if (node_min[i] > min_range[i])
        {
            overlap_min[i] = node_min[i];
            overlap_max[i] = max_range[i];
        }
        else if (node_max[i] > min_range[i])
        {
            overlap_min[i] = min_range[i];
            overlap_max[i] = node_max[i];
        }
    }
    MetaData meta_min(&overlap_min);
    meta_min.setMapVal(range_bound, cell_area);
    MetaData meta_max(&overlap_max);
    meta_max.setMapVal(range_bound, cell_area);
    int pre_min_position = index_model->preFastPosition(meta_min.map_val);
    int pre_max_position = index_model->preFastPosition(meta_max.map_val);

    pre_min_position = pre_min_position > 0 ? pre_min_position : 0;
    pre_max_position = pre_max_position > 0 ? pre_max_position : 0;

    pre_min_position = pre_min_position > metadataVec.size() - 1 ? metadataVec.size() - 1 : pre_min_position;
    pre_max_position = pre_max_position > metadataVec.size() - 1 ? metadataVec.size() - 1 : pre_max_position;

    pre_min_position = knnAdjustPosition(metadataVec, index_model->error_bound, pre_min_position, meta_min);
    pre_max_position = knnAdjustPosition(metadataVec, index_model->error_bound, pre_max_position, meta_max);

    // pre_min_position = adjustPosition(metadataVec, index_model->error_bound, pre_min_position, meta_min, -1);
    // pre_max_position = adjustPosition(metadataVec, index_model->error_bound, pre_max_position, meta_max, 1);

    for (int i = pre_min_position; i <= pre_max_position; i++)
    {
        if (min_range[0] > (*(metadataVec[i].data))[0] || (*(metadataVec[i].data))[0] > max_range[0] ||
            min_range[1] > (*(metadataVec[i].data))[1] || (*(metadataVec[i].data))[1] > max_range[1])
        {
            continue;
        }
        else
        {
            temp_result.push(metadataVec[i].data);
        }
    }
}