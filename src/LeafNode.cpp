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
	index_model = new IndexModel(&metadataVec);
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

vector<array<double, 2> *> &LeafNode::pointSearch(array<double, 2> key, std::vector<array<double, 2> *> &result, ExpRecorder &exp_Recorder)
{
	// auto start_prePos = chrono::high_resolution_clock::now();

	MetaData meta_key(&key);
	meta_key.setMapVal(range_bound, cell_area);
	int pre_position = index_model->preFastPosition(meta_key.map_val);
	// pre_position = std::min(pre_position, (int)(metadataVec.size() - 1));
	// pre_position = std::max(pre_position, 0);
	pre_position = pre_position > metadataVec.size() - 1 ? metadataVec.size() - 1 : pre_position;
	pre_position = pre_position > 0 ? pre_position : 0;
	// auto end_prePos = chrono::high_resolution_clock::now();
	// exp_Recorder.pointModelPreTime +=chrono::duration_cast<chrono::nanoseconds>(end_prePos - start_prePos).count();
	// auto start_BinSearch = chrono::high_resolution_clock::now();

	double pre_map_val = metadataVec[pre_position].map_val;
	// wast time?
	if (pre_map_val > meta_key.map_val)
	{
		// int min_search_index = std::max(pre_position + index_model->error_bound[1], 0);
		int min_search_index = pre_position + index_model->error_bound[0] > 0 ? pre_position + index_model->error_bound[0] : 0;
		return bindary_search(metadataVec, metadataVecBitMap, min_search_index, pre_position, meta_key, result, exp_Recorder);
	}
	else
	{
		// int max_search_position = std::min(pre_position + index_model->error_bound[0], (int)(metadataVec.size() - 1));
		int max_search_position = pre_position + index_model->error_bound[1] > metadataVec.size() - 1 ? metadataVec.size() - 1 : pre_position + index_model->error_bound[1];
		return bindary_search(metadataVec, metadataVecBitMap, pre_position, max_search_position, meta_key, result, exp_Recorder);
	}
	// auto end_BinSearch = chrono::high_resolution_clock::now();
	// exp_Recorder.pointBindarySearchTime +=chrono::duration_cast<chrono::nanoseconds>(end_BinSearch - start_BinSearch).count();
}

vector<array<double, 2> *> &LeafNode::rangeSearch(std::vector<double> query_range, vector<array<double, 2> *> &result)
{
	double *min_range = new double[MetaData::dim];
	double *max_range = new double[MetaData::dim];
	for (int i = 0; i < MetaData::dim; i++)
	{
		min_range[i] = query_range[i * 2];
		max_range[i] = query_range[i * 2 + 1];
	}

	double *node_min = new double[MetaData::dim];
	double *node_max = new double[MetaData::dim];

	for (int i = 0; i < MetaData::dim; i++)
	{
		node_min[i] = range_bound[i * 2];
		node_max[i] = range_bound[i * 2 + 1];
	}
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

	// pre_min_position = std::max(pre_min_position, 0);
	// pre_max_position = std::max(pre_max_position, 0);
	pre_min_position = pre_min_position > 0 ? pre_min_position : 0;
	pre_max_position = pre_max_position > 0 ? pre_max_position : 0;

	// pre_min_position = std::min(pre_min_position, (int)(metadataVec.size() - 1));
	// pre_max_position = std::min(pre_max_position, (int)(metadataVec.size() - 1));
	pre_min_position = pre_min_position > metadataVec.size() - 1 ? metadataVec.size() - 1 : pre_min_position;
	pre_max_position = pre_max_position > metadataVec.size() - 1 ? metadataVec.size() - 1 : pre_max_position;

	pre_min_position = adjustPosition(metadataVec, index_model->error_bound, pre_min_position, meta_min, -1);
	pre_max_position = adjustPosition(metadataVec, index_model->error_bound, pre_max_position, meta_max, 1);
	scan(metadataVec, pre_min_position, pre_max_position, min_range, max_range, result);
	// ! scanBuffer
	scanBuffer(insertBuffer, bufferDataSize, min_range, max_range, result);

	delete[] min_range;
	delete[] max_range;
	delete[] node_min;
	delete[] node_max;

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
	for (int i = 0; i < metadataVec.size(); i++)
	{
		this->metadataVecBitMap[i] = 1;
	}
}

bool LeafNode::insert(array<double, 2> &point)
{
	// return ture if merge then in cell tree check the leaf node
	// return false if not merge.

	MetaData insertMetadata(&point);
	insertMetadata.setMapVal(this->range_bound, this->cell_area);
	bool mergeFlag = false;
	if (bufferDataSize < INSERT_BUFFERSIZE)
	{
		insertBuffer[bufferDataSize] = insertMetadata;
		bufferDataSize++;
		std::sort(insertBuffer.begin(), insertBuffer.begin() + bufferDataSize, compareMetadata);
	}
	else
	{
		// merge the data into metadataVec
		for (int i = 0; i < bufferDataSize; i++)
		{
			metadataVec.push_back(insertBuffer[i]);
		}
		mergeFlag = true;
		bufferDataSize = 0;
	}
	return mergeFlag;
}

bool LeafNode::remove(array<double, 2> &point)
{

	MetaData deleteMetadata(&point);
	deleteMetadata.setMapVal(this->range_bound, this->cell_area);
	int prePosition = this->index_model->preFastPosition(deleteMetadata.map_val);

	if (metadataVec[prePosition].map_val > deleteMetadata.map_val)
	{
		// int min_search_index = std::max(pre_position + index_model->error_bound[1], 0);
		int min_search_index = prePosition + index_model->error_bound[0] > 0 ? prePosition + index_model->error_bound[0] : 0;
		deleteMetadataInRange(metadataVec, metadataVecBitMap, min_search_index, prePosition, deleteMetadata);
	}
	else
	{
		// int max_search_position = std::min(pre_position + index_model->error_bound[0], (int)(metadataVec.size() - 1));
		int max_search_position = prePosition + index_model->error_bound[1] > metadataVec.size() - 1 ? metadataVec.size() - 1 : prePosition + index_model->error_bound[1];
		deleteMetadataInRange(metadataVec, metadataVecBitMap, prePosition, max_search_position, deleteMetadata);
	}
	int deleteNumInBuffer = 0;
	for (int i = 0; i < bufferDataSize; i++)
	{
		if (compareMetadata(insertBuffer[i], deleteMetadata))
		{
			insertBuffer[i].map_val = numeric_limits<double>::max();
			insertBuffer[i].data = nullptr;
			deleteNumInBuffer++;
		}
	}
	if (deleteNumInBuffer > 0)
	{
		std::sort(insertBuffer.begin(), insertBuffer.begin() + bufferDataSize, compareMetadata);
		this->bufferDataSize -= deleteNumInBuffer;
	}

	// build check this node in cell tree by checking num.
}

int LeafNode::getKeysNum()
{
	return this->metadataVecBitMap.count() + bufferDataSize;
}
