//#include <vector>
//#include <torch/torch.h>
//#include "MetaData.h"
//#include "Utils.h"
#include "IndexModel.h"

using namespace std;

NNDataSet::NNDataSet(std::vector<double> _map_vals)
{
	map_vals = _map_vals;
	length = map_vals.size();
}

torch::data::Example<> NNDataSet::get(size_t index)
{
	torch::Tensor map_v = torch::tensor({map_vals[index]}, {torch::kFloat32}).view({-1, 1});
	torch::Tensor pos_v = torch::tensor({(double)index / (double)(length - 1)}, {torch::kFloat32}).view({-1, 1});
	return {map_v.clone(), pos_v.clone()};
}

torch::optional<size_t> NNDataSet::size() const
{
	return length;
};

IndexModel::IndexModel(std::vector<MetaData> *metadataVec)
{
	for (auto meta_data : *metadataVec)
	{
		this->mapValVec.push_back(meta_data.map_val);
	}

	nnModel = new NNModel(1, 100, 1);
	// orderMetaData();
}

IndexModel::IndexModel(std::vector<double> &mapvalvec)
{
	this->mapValVec = mapvalvec;
	nnModel = new NNModel(1, 100, 1);
}

IndexModel::~IndexModel()
{
	if (nnModel != nullptr)
	{
		delete nnModel;
	}
}

void IndexModel::buildModel()
{
	auto data_set = NNDataSet(this->mapValVec).map(torch::data::transforms::Stack<>());
	auto data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(data_set), torch::data::DataLoaderOptions().batch_size(128));

	torch::optim::Adam optimizer(nnModel->parameters(), 0.001);

	for (size_t epoch = 0; epoch < 200; epoch++)
	{
		size_t batch_index = 0;
		double loss_item = 0.0;
		for (auto &batch : *data_loader)
		{
			optimizer.zero_grad();
			torch::Tensor pre = nnModel->forward(batch.data);
			torch::Tensor loss = torch::mse_loss(pre, batch.target);
			loss.backward();
			optimizer.step();
			// std::cout << "Epoch: " << epoch << "batch:"<< batch_index++ << "loss: " << std::setprecision(3) << loss.item<double>() << std::endl;
			loss_item += loss.item<double>();
		}
		std::cout << "epoch: " << epoch << "loss:" << std::setprecision(5) << loss_item << std::endl;
	}

	torch::NoGradGuard no_grad;

	double up_error = 0.0;
	double down_error = 0.0;
	for (auto &batch : *data_loader)
	{
		torch::Tensor pre = nnModel->forward(batch.data);
		torch::Tensor target = batch.target;
		const size_t train_dataset_size = data_set.size().value();
		torch::Tensor error = (target - pre) * (double)train_dataset_size;

		for (int i = 0; i < error.sizes()[0]; i++)
		{
			double error_i = error[i].item<double>();
			if (error_i < down_error && error_i < 0)
			{
				down_error = error_i;
			}
			else if (error_i > up_error && error_i > 0)
			{
				up_error = error_i;
			}
		}
	}
	error_bound.push_back((int)down_error - 1);
	error_bound.push_back((int)up_error + 1);
}

void IndexModel::getErrorBound()
{
	torch::NoGradGuard no_grad;

	auto data_set = NNDataSet(this->mapValVec).map(torch::data::transforms::Stack<>());
	auto data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(data_set), torch::data::DataLoaderOptions().batch_size(5000));

	double up_error = 0.0;
	double down_error = 0.0;
	for (auto &batch : *data_loader)
	{
		torch::Tensor pre = nnModel->forward(batch.data);
		torch::Tensor target = batch.target;
		const size_t train_dataset_size = data_set.size().value();
		torch::Tensor error = (target - pre) * (double)train_dataset_size;

		for (int i = 0; i < error.sizes()[0]; i++)
		{
			double error_i = error[i].item<double>();
			if (error_i < down_error && error_i < 0)
			{
				down_error = error_i;
			}
			else if (error_i > up_error && error_i > 0)
			{
				up_error = error_i;
			}
		}
	}
	error_bound.push_back((int)down_error - 1);
	error_bound.push_back((int)up_error + 1);
}

int IndexModel::prePosition(double meta_key_map_val)
{
	torch::NoGradGuard no_grad;
	torch::Tensor map_val_tensor = torch::tensor({meta_key_map_val}, {torch::kFloat32}).view({1, -1});
	torch::Tensor pre_pos = nnModel->forward(map_val_tensor);
	// cout << pre_pos.item<double>() << endl;
	int pre_position = (int)(pre_pos.item<double>() * this->mapValVec.size()) + 1;
	return pre_position;
}

void IndexModel::loadParameter(string paramPath)
{
	FileReader *file_reader = new FileReader();
	vector<double> modelParam = file_reader->getModelParameter(paramPath);
	vector<float> floatParam;
	for (auto param : modelParam)
	{
		floatParam.push_back((float)param);
	}
	// nnModel->input.
	torch::Tensor modelTensorParam = torch::tensor(modelParam, torch::dtype(torch::kFloat32));
	torch::Tensor inputLayerWeight = modelTensorParam.index({torch::indexing::Slice(torch::indexing::None, 100, torch::indexing::None)}).view({-1, 1});
	torch::Tensor inputLayerBias = modelTensorParam.index({torch::indexing::Slice(100, 200, torch::indexing::None)}).view({-1});
	torch::Tensor hidenLayerWeight = modelTensorParam.index({torch::indexing::Slice(200, 300, torch::indexing::None)}).view({1, -1});
	torch::Tensor hidenLayerBias = modelTensorParam.index({torch::indexing::Slice(300, 301, torch::indexing::None)}).view({-1});
	nnModel->input->weight.set_data(inputLayerWeight);
	nnModel->input->bias.set_data(inputLayerBias);
	nnModel->hiden->weight.set_data(hidenLayerWeight);
	nnModel->hiden->bias.set_data(hidenLayerBias);

	Eigen::MatrixXf modelParamVec = Eigen::Map<Eigen::Matrix<float, 301, 1>>(floatParam.data());
	this->input_weight = modelParamVec.block(0, 0, 100, 1);
	this->input_bias = modelParamVec.block(100, 0, 100, 1).transpose();
	this->hiden_weight = modelParamVec.block(200, 0, 100, 1).transpose();
	this->hiden_bias = modelParamVec.block(300, 0, 1, 1);
}

int IndexModel::preFastPosition(double map_val)
{
	Eigen::Matrix<float, 1, 1> input{{(float)map_val}};
	// Eigen::MatrixXf result1;
	result = input * input_weight.transpose();
	result = result + input_bias;
	result = result.cwiseMax(hidenActive1) * hiden_weight.transpose();
	result = result + hiden_bias;
	result = result.cwiseMax(hidenActive2);
	double pre_pos = result(0, 0);
	// cout << pre_pos << endl;
	return (int)(pre_pos * this->mapValVec.size()) + 1;
}
