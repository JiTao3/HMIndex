#include "CellTree.h"
// #include "Point.h"
// #include <time.h>

int SPLIT_LEAF_NODE_NUM = 0;
int TRAIN_LEAF_NODE_NUM = 0;

CellTree::CellTree()
{
}
CellTree::~CellTree()
{
}

CellTree::CellTree(int split_num, vector<double> data_space_bound, string raw_path)
{
    this->split_num = split_num;
    this->data_space_bound = data_space_bound;
    this->root.range_bound = data_space_bound;
    this->loadRawData(raw_path);
    this->initialPartionBound(split_num + 1);
    this->initialPartionData();
}

void CellTree::loadRawData(string file_path)
{
    FileReader filereader(file_path, ",");
    raw_data = filereader.get_array_points(file_path, ",");
    cout << "raw data size: " << raw_data.size() << endl;
}

void CellTree::initialPartionBound(int n)
{
    vector<double> lin = LinSpace(0.0, 1.0, n);
    vector<double> x;
    vector<double> y;

    for (int i = 0; i < raw_data.size(); i++)
    {
        x.push_back(raw_data[i][0]);
        y.push_back(raw_data[i][1]);
    }
    sort(y.begin(), y.end());
    sort(x.begin(), x.end());
    // push back cell bound
    init_partion_bound_x.push_back(data_space_bound[0]);
    init_partion_bound_y.push_back(data_space_bound[2]);

    for (int i = 1; i < lin.size() - 1; i++)
    {
        init_partion_bound_x.push_back(percentile(x, lin[i], true));
        init_partion_bound_y.push_back(percentile(y, lin[i], true));
    }
    init_partion_bound_x.push_back(data_space_bound[1]);
    init_partion_bound_y.push_back(data_space_bound[3]);

    for (int i = 0; i < MetaData::dim; i++)
    {
        std::vector<int> dim_split_idx;
        for (int j = 0; j < split_num + 1; j++)
        {
            dim_split_idx.push_back(j);
        }
        this->cell_bound_idx.push_back(dim_split_idx);
    }

    // generate hist info for knn
    vector<double> hist_lin = LinSpace(0.0, 1.0, this->bin_num + 1);
    hist_info_x.push_back(data_space_bound[0]);
    hist_info_y.push_back(data_space_bound[2]);

    for (int i = 1; i < hist_lin.size() - 1; i++)
    {
        hist_info_x.push_back(percentile(x, hist_lin[i], true));
        hist_info_y.push_back(percentile(y, hist_lin[i], true));
    }
    hist_info_x.push_back(data_space_bound[1]);
    hist_info_y.push_back(data_space_bound[3]);
}

void CellTree::initialPartionData()
{
    for (int i = 0; i < split_num; i++)
    {
        vector<vector<MetaData>> raw_cell_medata;
        for (int j = 0; j < split_num; j++)
        {
            vector<MetaData> cell_medata;
            raw_cell_medata.push_back(cell_medata);
        }
        this->initial_partition_data.push_back(raw_cell_medata);
    }
    for (int i = 0; i < raw_data.size(); i++)
    {
        vector<int> cell_index = getCellIndex(raw_data[i], this->init_partion_bound_x, this->init_partion_bound_y);
        MetaData *me_data = new MetaData(&raw_data[i]);
        // me_data->setMapVal()
        this->initial_partition_data[cell_index[0]][cell_index[1]].push_back(*me_data);
        // cout << "\r initial pattion data: " << i;
    }
    // cout << endl;
}

void CellTree::buildTree(std::vector<std::vector<int>> cell_bound_idx, InnerNode *root_node, int child_idx)
{
    bool is_single_cell = true;
    for (int i = 0; i < MetaData::dim; i++)
    {
        if (cell_bound_idx[i].size() != 2)
        {
            is_single_cell = false;
            break;
        }
        else if (cell_bound_idx[i][0] == cell_bound_idx[i][1])
        {
            return;
        }
    }
    // if is a leaf node
    if (is_single_cell)
    {
        vector<double> cell_bound = root_node->range_bound;
        vector<MetaData> medata_vec = this->initial_partition_data[cell_bound_idx[0][0]][cell_bound_idx[1][0]];
        // cout << "metadata num: " << medata_vec.size() << endl;
        InnerNode *parent_node = root_node->parent;

        LeafNode *leaf_node = new LeafNode(medata_vec, cell_bound);
        leaf_node->parent = parent_node;
        parent_node->children.at(child_idx) = leaf_node;
        // train_model
    }

    // else if still can split
    else
    {
        std::vector<std::vector<int>> left_splited, right_splited;
        std::vector<double> split_point;
        // get split point and leaf&right partion of each dimention
        for (int i = 0; i < MetaData::dim; i++)
        {
            std::vector<int> dim_split_index = cell_bound_idx[i];
            // split from mid index
            int mid = (*dim_split_index.begin() + *(dim_split_index.end() - 1)) / 2;
            // get split point
            if (i == 0)
                split_point.push_back(init_partion_bound_x[mid]);
            else if (i == 1)
                split_point.push_back(init_partion_bound_y[mid]);
            else
                cout << "error for multi dim" << endl;

            std::vector<int> left, right;
            if (dim_split_index.size() == 2)
            {
                left.push_back(mid);
                left.push_back(mid);
                right.push_back(mid);
                right.push_back(mid + 1);
            }
            else
            {
                int mid_idx = (dim_split_index.size() + 1) / 2;
                left.insert(left.end(), dim_split_index.begin(), dim_split_index.begin() + mid_idx);
                right.insert(right.end(), dim_split_index.begin() + mid_idx - 1, dim_split_index.end());
            }
            // Can be optimized
            left_splited.push_back(left);
            right_splited.push_back(right);
        }
        root_node->split_point = split_point;
        for (int i = 0; i < pow(2, MetaData::dim); i++)
        {
            std::vector<std::vector<int>> split;

            std::vector<int> data_left_or_right = idx_to_vector_int(i, MetaData::dim);
            for (int k = 0; k < MetaData::dim; k++)
            {
                if (data_left_or_right[k] == 0)
                {
                    split.push_back(left_splited[k]);
                }
                else
                {
                    split.push_back(right_splited[k]);
                }
            }
            // build()
            vector<double> child_range_bound =
                splitNodeToSplitedRegion(root_node->range_bound, root_node->split_point, i);
            InnerNode *new_in_node = new InnerNode();
            new_in_node->range_bound = child_range_bound;
            new_in_node->parent = root_node;
            root_node->children[i] = new_in_node;
            buildTree(split, new_in_node, i);
        }
    }
}

void CellTree::buildCheck(boost::variant<InnerNode *, LeafNode *, GridNode *, int> root, int child_index,
                          bool openTraining)
{
    if (root.type() == typeid(InnerNode *))
    {
        // child node has already merge and split?
        InnerNode *node = boost::get<InnerNode *>(root);
        for (int i = 0; i < node->children.size(); i++)
        {
            if (node->children[i].type() == typeid(LeafNode *))
            {
                LeafNode *leaf_node = boost::get<LeafNode *>(node->children[i]);
                if (leaf_node->mergeDelete)
                    return;
            }
            else if (node->children[i].type() == typeid(InnerNode *))
            {
                InnerNode *inner_node = boost::get<InnerNode *>(node->children[i]);
                if (inner_node->mergeDelete)
                    return;
            }
            buildCheck(node->children[i], i);
        }
    }
    else if (root.type() == typeid(LeafNode *))
    {
        LeafNode *leaf_node = boost::get<LeafNode *>(root);
        InnerNode *parent_node = (InnerNode *)leaf_node->parent;
        if ((!leaf_node->splitFlag) && (leaf_node->getKeysNum() < mergeTh))
        {
            // merge and split and set merge flag true
            vector<double> split_point;
            vector<double> x;
            vector<double> y;

            vector<MetaData> all_metadata;
            getAllMetaData(parent_node, all_metadata);

            InnerNode *parent_parent_node = parent_node->parent;
            LeafNode *merge_node = new LeafNode(all_metadata, parent_node->range_bound);
            merge_node->parent = parent_parent_node;

            // delete the brother node to avoid the recursion.
            for (auto &child : parent_node->children)
            {
                if (child.type() == typeid(LeafNode *))
                {
                    LeafNode *leaf_node = boost::get<LeafNode *>(child);
                    leaf_node->mergeDelete = true;
                }
                else if (child.type() == typeid(InnerNode *))
                {
                    InnerNode *inner_node = boost::get<InnerNode *>(child);
                    inner_node->mergeDelete = true;
                }
            }
            // find merge node index in ppnode
            int p_index;
            for (p_index = 0; p_index < parent_parent_node->children.size(); p_index++)
            {
                if (parent_parent_node->children[p_index].type() == typeid(InnerNode *))
                {
                    InnerNode *ppchild_pointer = boost::get<InnerNode *>(parent_parent_node->children[p_index]);
                    if (ppchild_pointer == parent_node)
                    {
                        break;
                    }
                    else if (p_index == 3)
                    {
                        p_index = -1;
                        cout << "find error!" << endl;
                        break;
                    }
                }
            }
            parent_parent_node->children[p_index] = merge_node;
            buildCheck(merge_node, p_index);
            // set flag
        }
        else if (leaf_node->getKeysNum() > cellSplitTh)
        {
            // split leaf node into leaf node
            // split from point
            /*vector<double> split_point;
                            for (int j = 0; j < leaf_node.range_bound.size(); j++) {
                                    split_point.push_back(leaf_node.range_bound[j] /
               2);
                            }*/

            vector<double> split_point;
            vector<double> x;
            vector<double> y;

            for (int j = 0; j < leaf_node->metadataVec.size(); j++)
            {
                MetaData medata = leaf_node->metadataVec[j];
                x.push_back((*medata.data)[0]);
                y.push_back((*medata.data)[1]);
            }

            split_point.push_back(percentile(x, 0.5));
            split_point.push_back(percentile(y, 0.5));

            InnerNode *replace_node = new InnerNode();
            parent_node->children[child_index] = replace_node;
            replace_node->split_point = split_point;
            replace_node->range_bound = leaf_node->range_bound;
            replace_node->parent = (InnerNode *)leaf_node->parent;
            // **** info of replace node *****

            for (int j = 0; j < replace_node->children.size(); j++)
            {
                vector<MetaData> medataVec = splitLeafNodeToLeaf(leaf_node->metadataVec, split_point, j);
                vector<double> range_bound = splitNodeToSplitedRegion(leaf_node->range_bound, split_point, j);
                LeafNode *child_node = new LeafNode(medataVec, range_bound);
                child_node->parent = replace_node;
                child_node->splitFlag = true;

                replace_node->children[j] = child_node;
                buildCheck(replace_node->children[j], j);
            }

            // remember to delete &leaf_node;
        }
        else if (leaf_node->getKeysNum() > gridSplitTh)
        {
            // split leaf node into grid
            // split into grids
            // split dim select
            int split_dim = 0;
            int split_num = leaf_node->getKeysNum() / modelCapability;
            vector<double> split_point;
            vector<double> lin = LinSpace(0.0, 1.0, split_num + 1);
            vector<double> x;
            // vector<double> y;
            for (int j = 0; j < leaf_node->metadataVec.size(); j++)
            {
                MetaData medata = leaf_node->metadataVec[j];
                x.push_back((*medata.data)[0]);
                // y.push_back((*medata.data)[1]);
            }
            for (int j = 0; j < lin.size(); j++)
            {
                // split from x
                if (lin[j] == 0.0)
                {
                    split_point.push_back(leaf_node->range_bound[0]);
                }
                else if (lin[j] == 1.0)
                {
                    split_point.push_back(leaf_node->range_bound[1]);
                }
                else
                {
                    split_point.push_back(percentile(x, lin[j]));
                }
            }

            InnerNode *replace_node = new InnerNode();
            parent_node->children[child_index] = replace_node;
            replace_node->split_point = split_point;
            replace_node->range_bound = leaf_node->range_bound;
            replace_node->parent = (InnerNode *)leaf_node->parent;

            if (replace_node->children.size() < split_num)
            {
                for (int j = 0; j < replace_node->children.size() - split_num; j++)
                {
                    replace_node->children.push_back(-11);
                }
            }
            // **** info of replace node ****

            for (int j = 0; j < split_num; j++)
            {
                vector<double> range_bound =
                    splitLeafNodeToGridRegion(leaf_node->range_bound, split_point, split_dim, j);
                vector<MetaData> metadataVec = splitLeafNodeToGrid(leaf_node->metadataVec, range_bound);
                GridNode *grid_node = new GridNode(metadataVec, range_bound);
                grid_node->parent = replace_node;
                grid_node->parent_rangeBound = replace_node->range_bound;
                replace_node->children[j] = grid_node;
            }
        }
    }
    else if (openTraining && root.type() == typeid(GridNode *))
    {
        GridNode *grid_node = boost::get<GridNode *>(root);
        InnerNode *parent_node = (InnerNode *)grid_node->parent;

        // ! if all grid node keys > split th
        //		! merge all grid node into a leafnode and split
        // ! else grid can only slice into 2 grid
        // ! also grid can only merge with anther grid
        // 		! may be split again

        int metadataNumAllChildGridNode = 0;

        for (auto &child : parent_node->children)
        {
            if (child.type() == typeid(InnerNode *))
                continue;
            else if (child.type() == typeid(LeafNode *))
            {
                LeafNode *child_leaf_node = boost::get<LeafNode *>(child);
                metadataNumAllChildGridNode += child_leaf_node->getKeysNum();
            }
            else if (child.type() == typeid(GridNode *))
            {
                GridNode *child_grid_node = boost::get<GridNode *>(child);
                metadataNumAllChildGridNode += child_grid_node->getKeysNum();
            }
        }
        if (metadataNumAllChildGridNode > cellSplitTh)
        {
            vector<double> split_point;
            vector<double> x;
            vector<double> y;

            vector<MetaData> all_metadata;
            getAllMetaData(parent_node, all_metadata);

            InnerNode *parent_parent_node = parent_node->parent;
            LeafNode *merge_node = new LeafNode(all_metadata, parent_node->range_bound);
            merge_node->parent = parent_parent_node;

            // * delete the GridNode
            for (auto &child : parent_node->children)
            {
                if (child.type() == typeid(GridNode *))
                {
                    GridNode *grid_child = boost::get<GridNode *>(child);
                    delete grid_node;
                }
            }

            // * find parent parent node index
            int p_index;
            for (p_index = 0; p_index < parent_parent_node->children.size(); p_index++)
            {
                if (parent_parent_node->children[p_index].type() == typeid(InnerNode *))
                {
                    InnerNode *ppchild_pointer = boost::get<InnerNode *>(parent_parent_node->children[p_index]);
                    if (ppchild_pointer == parent_node)
                    {
                        break;
                    }
                    else if (p_index == 3)
                    {
                        p_index = -1;
                        cout << "find error!" << endl;
                        break;
                    }
                }
            }
            parent_parent_node->children[p_index] = merge_node;
            buildCheck(merge_node, p_index);
        }
        else if (grid_node->getKeysNum() > gridSplitTh)
        {
            int split_dim = 0;
            int split_num = 2;
            vector<double> split_point;
            vector<double> lin = LinSpace(0.0, 1.0, split_num + 1);
            vector<double> x;
            // vector<double> y;
            for (int j = 0; j < grid_node->metadataVec.size(); j++)
            {
                MetaData medata = grid_node->metadataVec[j];
                x.push_back((*medata.data)[0]);
                // y.push_back((*medata.data)[1]);
            }
            for (int j = 0; j < lin.size(); j++)
            {
                // split from x
                if (lin[j] == 0.0)
                    split_point.push_back(grid_node->rangeBound[0]);
                else if (lin[j] == 1.0)
                    split_point.push_back(grid_node->rangeBound[1]);
                else
                    split_point.push_back(percentile(x, lin[j]));
            }

            for (int j = 0; j < split_num; j++)
            {
                vector<double> range_bound =
                    splitLeafNodeToGridRegion(grid_node->rangeBound, split_point, split_dim, j);
                vector<MetaData> metadataVec = splitLeafNodeToGrid(grid_node->metadataVec, range_bound);
                GridNode *new_grid_node = new GridNode(metadataVec, range_bound);
                new_grid_node->parent = parent_node;
                new_grid_node->parent_rangeBound = grid_node->parent_rangeBound;
                if (j == 0)
                    parent_node->children[child_index] = new_grid_node;
                else
                    parent_node->children.insert(parent_node->children.begin() + child_index + j, new_grid_node);
            }
        }
        else if (grid_node->getKeysNum() < mergeTh)
        {
            int childGridNodeNum = 0;
            int selectedGridNodeIdx = 0;
            for (auto &child : parent_node->children)
            {
                if (child.type() == typeid(GridNode *))
                    childGridNodeNum++;
            }
            if (childGridNodeNum >= 2)
            {
                if (child_index == 0)
                    selectedGridNodeIdx = 1;
                else
                    selectedGridNodeIdx = child_index - 1;
            }
            else
                return;
            GridNode *selectedMergeNode = boost::get<GridNode *>(parent_node->children[selectedGridNodeIdx]);
            vector<MetaData> mergeMetaData;
            vector<double> mergeRangeBound = {0.0, 0.0, 0.0, 0.0};
            for (int i = 0; i < grid_node->metadataVec.size(); i++)
                mergeMetaData.push_back(grid_node->metadataVec[i]);
            for (int i = 0; i < selectedMergeNode->metadataVec.size(); i++)
                mergeMetaData.push_back(selectedMergeNode->metadataVec[i]);
            if (grid_node->bufferDataSize > 0)
            {
                for (int i = 0; i < grid_node->bufferDataSize; i++)
                    mergeMetaData.push_back(grid_node->insertBuffer[i]);
            }
            if (selectedMergeNode->bufferDataSize > 0)
            {
                for (int i = 0; i < selectedMergeNode->bufferDataSize; i++)
                    mergeMetaData.push_back(selectedMergeNode->insertBuffer[i]);
            }

            mergeRangeBound[0] = std::min(selectedMergeNode->rangeBound[0], grid_node->rangeBound[0]);
            mergeRangeBound[1] = std::max(selectedMergeNode->rangeBound[1], grid_node->rangeBound[1]);
            mergeRangeBound[2] = std::min(selectedMergeNode->rangeBound[2], grid_node->rangeBound[2]);
            mergeRangeBound[3] = std::max(selectedMergeNode->rangeBound[3], grid_node->rangeBound[3]);

            GridNode *replaceGridNode = new GridNode(mergeMetaData, mergeRangeBound);

            parent_node->children[std::min(child_index, selectedGridNodeIdx)] = replaceGridNode;
            parent_node->children.erase(parent_node->children.begin() + std::max(selectedGridNodeIdx, child_index));
            parent_node->split_point.erase(parent_node->split_point.begin() +
                                           std::max(selectedGridNodeIdx, child_index));
        }
    }
    else
    {
        return;
    }
}

void CellTree::saveSplitData(boost::variant<InnerNode *, LeafNode *, GridNode *, int> root)
{
    if (root.type() == typeid(InnerNode *))
    {
        InnerNode *node = boost::get<InnerNode *>(root);
        for (auto &child : node->children)
        {
            saveSplitData(child);
        }
    }
    else if (root.type() == typeid(LeafNode *))
    {
        LeafNode *node = boost::get<LeafNode *>(root);
        node->saveMetaDataVectoCSV(this->saveSplitPath + to_string(SPLIT_LEAF_NODE_NUM++) + ".csv");
    }
    else if (root.type() == typeid(GridNode *))
    {
        GridNode *node = boost::get<GridNode *>(root);
        node->saveMetaDataVectoCSV(this->saveSplitPath + to_string(SPLIT_LEAF_NODE_NUM++) + ".csv");
    }
    else
    {
        return;
    }
}

vector<array<double, 2> *> &CellTree::pointSearch(array<double, 2> &query, vector<array<double, 2> *> &result,
                                                  ExpRecorder &exp_Recorder)
{

    boost::variant<InnerNode *, LeafNode *, GridNode *, int> node = &this->root;

    while (node.type() == typeid(InnerNode *))
    {
        InnerNode *innernode = boost::get<InnerNode *>(node);
        int child_index = innernode->child_index(query);
        node = innernode->children[child_index];
    }
    if (node.type() == typeid(LeafNode *))
    {
        LeafNode *leaf_node = boost::get<LeafNode *>(node);
        return leaf_node->pointSearch(query, result, exp_Recorder);
    }
    else if (node.type() == typeid(GridNode *))
    {
        GridNode *grid_node = boost::get<GridNode *>(node);
        return grid_node->pointSearch(query, result, exp_Recorder);
    }
}

vector<array<double, 2> *> &CellTree::rangeSearch(vector<double> &query, vector<array<double, 2> *> &result,
                                                  ExpRecorder &exp_Recorder)
{
    auto start_range = chrono::high_resolution_clock::now();
    this->DFSCelltree(query, result, &this->root, exp_Recorder);
    auto end_range = chrono::high_resolution_clock::now();
    exp_Recorder.rangeTotalTime += chrono::duration_cast<chrono::nanoseconds>(end_range - start_range).count();
    return result;
}

vector<array<double, 2> *> &CellTree::kNNSearch(array<double, 2> &query, int k, vector<array<double, 2> *> &result,
                                                ExpRecorder &exp_Recorder)
{
    boost::variant<InnerNode *, LeafNode *, GridNode *, int> node = &this->root;
    double cellArea = 0.0;
    double cellKeyNum = 0.0;
    vector<double> *t_rangeBound;

    while (node.type() == typeid(InnerNode *))
    {
        InnerNode *innernode = boost::get<InnerNode *>(node);
        int child_index = innernode->child_index(query);
        node = innernode->children[child_index];
    }
    if (node.type() == typeid(LeafNode *))
    {
        LeafNode *leaf_node = boost::get<LeafNode *>(node);
        cellArea = leaf_node->cell_area;
        cellKeyNum = (double)leaf_node->metadataVec.size();
        t_rangeBound = &(leaf_node->range_bound);
    }
    else if (node.type() == typeid(GridNode *))
    {
        GridNode *grid_node = boost::get<GridNode *>(node);
        cellArea = grid_node->getGridArea();
        cellKeyNum = (double)grid_node->metadataVec.size();
        t_rangeBound = &(grid_node->rangeBound);
    }
    double R_bar = sqrt((cellArea * k) / (M_PI * cellKeyNum));

    // for each dimension use r_bar and eq-depth histgoram to estimate the
    // cardinality of r_bar
    //
    array<double, MetaData::dim> p_u;
    array<double, MetaData::dim> p_i;
    array<double, MetaData::dim> r_i;
    array<int, MetaData::dim> hist_index;
    for (int i = 0; i < MetaData::dim; i++)
    {
        // p_u
        // p_i
        p_u[i] = cellKeyNum / (((*t_rangeBound)[2 * i + 1] - (*t_rangeBound)[2 * i]) * (double)raw_data.size());

        vector<double> *dim_hist_info = i == 0 ? &this->hist_info_x : &this->hist_info_y;
        int start = 0;
        int end = dim_hist_info->size() - 1;
        while (start < end && (end - start) > 1)
        {
            int mid = (start + end) / 2;
            if ((*dim_hist_info)[mid] <= query[i])
                start = mid;
            else
                end = mid;
        }
        hist_index[i] = start;
        // !!! p_i
        p_i[i] = 1 / (((*dim_hist_info).size() - 1) *
                      ((*dim_hist_info)[hist_index[i] + 1] - (*dim_hist_info)[hist_index[i]]));
        r_i[i] = p_u[i] / p_i[i] * R_bar;
    }

    double range_R = r_i[0] > r_i[1] ? r_i[0] : r_i[1];
    vector<array<double, 2> *> temp_result;

    int num_range_query = 0;

    while (true)
    {
        vector<double> range_query = {query[0] - range_R, query[0] + range_R, query[1] - range_R, query[1] + range_R};
        temp_result.clear();
        this->DFSCelltree(range_query, temp_result, &this->root, exp_Recorder);
        if (temp_result.size() > k)
        {
            sort(temp_result.begin(), temp_result.end(), sortForKNN(query));
            for (int i = 0; i < k; i++)
            {
                result.push_back(temp_result[i]);
            }

            break;
        }
        num_range_query++;
        // cout << "range query times: " << num_range_query << endl;
        range_R *= 2;
    }
    exp_Recorder.knnRangeQueryConterAvg += (double)num_range_query;
    // cout << "query" << num_range_query << endl;
    return result;
}

void CellTree::DFSCelltree(vector<double> &query, vector<array<double, 2> *> &result,
                           boost::variant<InnerNode *, LeafNode *, GridNode *, int> root, ExpRecorder &exp_Recorder)
{
    /*
            get overlap flag
            dissociate return
            contain get alldata
                    innernode
                    leafnode
                    gridnode
            intersection
                    inner node
                            dfs child
                    leaf node and grid node
                            range search
    */
    int overlapFlag = -2;
    if (root.type() == typeid(InnerNode *))
    {
        InnerNode *innernode = boost::get<InnerNode *>(root);
        overlapFlag = queryCellRealtion(innernode->range_bound, query);
        if (overlapFlag == DISSOCIATION)
            return;
        else if (overlapFlag == CONTAIN)
        {
            getAllData(innernode, result);
            return;
        }
        else if (overlapFlag == INTERSECTION)
        {
            for (auto child : innernode->children)
                this->DFSCelltree(query, result, child, exp_Recorder);
        }
    }
    else if (root.type() == typeid(LeafNode *))
    {
        LeafNode *leafnode = boost::get<LeafNode *>(root);
        overlapFlag = queryCellRealtion(leafnode->range_bound, query);
        if (overlapFlag == DISSOCIATION)
            return;
        else if (overlapFlag == CONTAIN)
        {
            for (auto metadata : leafnode->metadataVec)
                result.push_back(metadata.data);
            return;
        }
        else if (overlapFlag == INTERSECTION)
        {
            leafnode->rangeSearch(query, result, exp_Recorder);
            return;
        }
    }
    else if (root.type() == typeid(GridNode *))
    {
        GridNode *gridnode = boost::get<GridNode *>(root);
        overlapFlag = queryCellRealtion(gridnode->rangeBound, query);
        if (overlapFlag == DISSOCIATION)
            return;
        else if (overlapFlag == CONTAIN)
        {
            for (auto metadata : gridnode->metadataVec)
                result.push_back(metadata.data);
            return;
        }
        else if (overlapFlag == INTERSECTION)
        {
            gridnode->rangeSearch(query, result, exp_Recorder);
            return;
        }
    }
}

void CellTree::train(boost::variant<InnerNode *, LeafNode *, GridNode *, int> root)
{
    if (root.type() == typeid(InnerNode *))
    {
        InnerNode *node = boost::get<InnerNode *>(root);
        for (auto &child : node->children)
        {
            train(child);
        }
    }
    else if (root.type() == typeid(LeafNode *))
    {
        // load model initial parameter
        LeafNode *node = boost::get<LeafNode *>(root);
        // cout << "leaf node index :" << TRAIN_LEAF_NODE_NUM;
        node->index_model->loadParameter(this->modelParamPath + to_string(TRAIN_LEAF_NODE_NUM++) + ".csv");
        // this->train_pool->submit(node->index_model->buildModel(), );
        // node->index_model->buildModel();
        node->index_model->getErrorBound();
        // cout << "; node error bound: " << node->index_model->error_bound[0] << ", " <<
        // node->index_model->error_bound[1]
        //      << " ; node key conter :" << node->getKeysNum() << endl;
    }
    else if (root.type() == typeid(GridNode *))
    {
        // load model initial parameter
        GridNode *node = boost::get<GridNode *>(root);
        // cout << "grid node index :" << TRAIN_LEAF_NODE_NUM;
        node->index_model->loadParameter(this->modelParamPath + to_string(TRAIN_LEAF_NODE_NUM++) + ".csv");
        // node->index_model->buildModel();zaCVb
        node->index_model->getErrorBound();
        // cout << "; node error bound: " << node->index_model->error_bound[0] << ", " <<
        // node->index_model->error_bound[1]
        //      << " ; node key conter :" << node->getKeysNum() << endl;
    }
    else
    {
        return;
    }
}

void CellTree::insert(array<double, 2> &point)
{
    bool buildCheck = false;
    boost::variant<InnerNode *, LeafNode *, GridNode *, int> node = &this->root;
    int child_index = 0;
    while (node.type() == typeid(InnerNode *))
    {
        InnerNode *innernode = boost::get<InnerNode *>(node);
        child_index = innernode->child_index(point);
        node = innernode->children[child_index];
    }
    if (node.type() == typeid(LeafNode *))
    {
        LeafNode *leaf_node = boost::get<LeafNode *>(node);
        buildCheck = leaf_node->insert(point);
        if (buildCheck)
        {
            this->buildCheck(leaf_node, child_index, true);
        }
    }
    else if (node.type() == typeid(GridNode *))
    {
        GridNode *grid_node = boost::get<GridNode *>(node);
        buildCheck = grid_node->insert(point);
        if (buildCheck)
        {
            this->buildCheck(grid_node, child_index, true);
        }
    }
}

void CellTree::remove(array<double, 2> &point)
{
    bool buildCheck = false;
    boost::variant<InnerNode *, LeafNode *, GridNode *, int> node = &this->root;
    int child_index = 0;
    while (node.type() == typeid(InnerNode *))
    {
        InnerNode *innernode = boost::get<InnerNode *>(node);
        child_index = innernode->child_index(point);
        node = innernode->children[child_index];
    }
    if (node.type() == typeid(LeafNode *))
    {
        LeafNode *leaf_node = boost::get<LeafNode *>(node);
        leaf_node->remove(point);
        if (leaf_node->getKeysNum() < removeTh)
            this->buildCheck(leaf_node, child_index);
    }
    else if (node.type() == typeid(GridNode *))
    {
        GridNode *grid_node = boost::get<GridNode *>(node);
        grid_node->remove(point);
        if (grid_node->getKeysNum() < removeTh)
            this->buildCheck(grid_node, child_index);
    }
}

void getAllMetaData(boost::variant<InnerNode *, LeafNode *, GridNode *, int> root, vector<MetaData> &all_medata)
{
    if (root.type() == typeid(InnerNode *))
    {
        InnerNode *innernode = boost::get<InnerNode *>(root);
        for (int i = 0; i < innernode->children.size(); i++)
        {
            getAllMetaData(innernode->children[i], all_medata);
        }
    }
    else if (root.type() == typeid(LeafNode *))
    {
        LeafNode *leaf_node = boost::get<LeafNode *>(root);
        all_medata.insert(all_medata.end(), leaf_node->metadataVec.begin(), leaf_node->metadataVec.end());
    }
    else if (root.type() == typeid(GridNode *))
    {
        GridNode *grid_node = boost::get<GridNode *>(root);
        all_medata.insert(all_medata.end(), grid_node->metadataVec.begin(), grid_node->metadataVec.end());
    }
    else
    {
        return;
    }
}

void getAllData(boost::variant<InnerNode *, LeafNode *, GridNode *, int> root, vector<array<double, 2> *> &result)
{
    if (root.type() == typeid(InnerNode *))
    {
        InnerNode *innernode = boost::get<InnerNode *>(root);
        for (int i = 0; i < innernode->children.size(); i++)
        {
            getAllData(innernode->children[i], result);
        }
    }
    else if (root.type() == typeid(LeafNode *))
    {
        LeafNode *leaf_node = boost::get<LeafNode *>(root);
        for (auto &metadata : leaf_node->metadataVec)
        {
            result.push_back(metadata.data);
        }
    }
    else if (root.type() == typeid(GridNode *))
    {
        GridNode *grid_node = boost::get<GridNode *>(root);
        for (auto &metadata : grid_node->metadataVec)
        {
            result.push_back(metadata.data);
        }
    }
    else
    {
        return;
    }
}