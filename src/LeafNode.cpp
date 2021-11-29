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
	getKeyCounter();
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

void LeafNode::getKeyCounter()
{
	key_couter = metadataVec.size();
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
		return bindary_search(metadataVec, min_search_index, pre_position, meta_key, result, exp_Recorder);
	}
	else
	{
		// int max_search_position = std::min(pre_position + index_model->error_bound[0], (int)(metadataVec.size() - 1));
		int max_search_position = pre_position + index_model->error_bound[1] > metadataVec.size() - 1 ? metadataVec.size() - 1 : pre_position + index_model->error_bound[1];
		return bindary_search(metadataVec, pre_position, max_search_position, meta_key, result, exp_Recorder);
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

void LeafNode::setGappedArray()
{
	double insertGap = (double)this->metadataVec.size() / initialGapSize;
	this->bitMap.set();

	for (int i = 0; i < initialGapSize; i++)
	{
		MetaData *tmpMedata = new MetaData();
		metadataVec.insert(metadataVec.begin() + (int)(insertGap * i) + i, *tmpMedata);
		this->bitMap[(int)(insertGap * i) + i] = 0;
	}

	for (int i = metadataVec.size(); i < bitMap.size(); i++)
	{
		this->bitMap[i] = 0;
	}
}

void LeafNode::insert(array<double, 2> &point)
{
	MetaData insertMetadata(&point);
	insertMetadata.setMapVal(this->range_bound, this->cell_area);
	int insertPosition = this->index_model->preFastPosition(insertMetadata.map_val);
	bool binaryInsert = false;

	while (!this->bitMap[insertPosition--])
		;
	if (this->metadataVec[insertPosition].map_val == insertMetadata.map_val)
	{
		// when equality find a position on the right to insert
		int maxPosition = insertPosition + this->index_model->error_bound[1];
		binaryInsert = insertInBound(metadataVec, bitMap, insertMetadata, insertPosition, maxPosition);
	}
	else if (this->metadataVec[insertPosition].map_val > insertMetadata.map_val)
	{
		int minPosition = insertPosition + this->index_model->error_bound[0];
		binaryInsert = insertInBound(metadataVec, bitMap, insertMetadata, minPosition, insertPosition);
	}
	else
	{
		int maxPosition = insertPosition + this->index_model->error_bound[1];
		binaryInsert = insertInBound(metadataVec, bitMap, insertMetadata, insertPosition, maxPosition);
	}
	if (!binaryInsert)
	{
		// exponential search to find a position
		this->expSearchInUse = true;
		if (this->metadataVec[insertPosition].map_val > insertMetadata.map_val)
		{
			// 👈 find a gap position
			// get insert position 
			// move data in interval [gap_position, insert position]
			// insert
			
		}
		else 
		{
			// 👉 find a gap
			// get insert position 
			// move data in interval [insert position, gap_position]
			// insert
		}

	}
}