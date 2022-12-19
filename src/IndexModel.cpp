// #include <vector>
// #include <torch/torch.h>
// #include "MetaData.h"
// #include "Utils.h"
#include "IndexModel.h"

using namespace std;

torch::Device device = torch::cuda::is_available() ? torch::Device(torch::kCUDA, 3) : torch::kCPU;

NNDataSet::NNDataSet(std::vector<float> &_map_vals, std::vector<float> &_pos_vals)
{
    map_vals = &_map_vals;
    pos_vals = &_pos_vals;
    length = map_vals->size();
}

torch::data::Example<> NNDataSet::get(size_t index)
{
    torch::Tensor map_v = torch::tensor({(*map_vals)[index]}, {torch::kFloat32}).view({-1, 1});
    torch::Tensor pos_v = torch::tensor({(*pos_vals)[index]}, {torch::kFloat32}).view({-1, 1});
    return {map_v.clone(), pos_v.clone()};
}

torch::optional<size_t> NNDataSet::size() const
{
    return length;
};

IndexModel::IndexModel(std::vector<MetaData> *metadataVec)
{
    double idx = 0;
    for (auto meta_data : *metadataVec)
    {
        this->mapValVec.push_back(meta_data.map_val);
        this->positionVec.push_back(idx / (metadataVec->size() - 1));
        idx++;
    }

    nnModel = new NNModel(1, 50, 1);
    // orderMetaData();
}

IndexModel::IndexModel(std::vector<MetaData> &metadataVec)
{
    double idx = 0;

    for (auto &meta_data : metadataVec)
    {
        this->mapValVec.push_back(meta_data.map_val);
        this->positionVec.push_back(idx / (metadataVec.size() - 1));
        idx++;
    }

    nnModel = new NNModel(1, 50, 1);
}

IndexModel::IndexModel(std::vector<double> &mapvalvec)
{
    // this->mapValVec = mapvalvec;
    double idx = 0;

    for (auto &map_val : mapvalvec)
    {
        this->mapValVec.push_back(map_val);
        this->positionVec.push_back(idx / (mapvalvec.size() - 1));

        idx++;
    }
    nnModel = new NNModel(1, 50, 1);
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
    torch::Tensor x = torch::tensor(this->mapValVec).reshape({(long)this->mapValVec.size(), 1}).to(device);
    torch::Tensor y = torch::tensor(this->positionVec).reshape({(long)this->positionVec.size(), 1}).to(device);

    torch::optim::SGD optimizer(nnModel->parameters(), 0.005);
    this->nnModel->to(device);
    size_t batch_index = 0;
    for (size_t epoch = 0; epoch < 200; epoch++)
    {
        // double loss_item = 0.0;dasda
        // for (auto &batch : *data_loader)
        // {
        // batch_index++;
        optimizer.zero_grad();
        // cout << " idx : " << batch_index << " " << batch.data.sizes() << " "
        //      << batch.data.index({torch::indexing::Slice(60, 70, torch::indexing::None)}) << endl;
        torch::Tensor pre = nnModel->forward(x);
        torch::Tensor loss = torch::mse_loss(pre, y);
        loss.backward();
        optimizer.step();
        // std::cout << "Epoch: " << epoch << "batch:"<< batch_index++ << "loss: " << std::setprecision(5) <<
        // loss.item<double>() << std::endl; loss_item += loss.item<double>();
        // }
        // std::cout << "epoch: " << epoch << " loss:" << std::setprecision(5) << loss_item << std::endl;
    }

    nnModel->to(torch::kCPU);
    this->getErrorBound();
    this->initialIndexModelTrainedParam();
}

void IndexModel::getErrorBound()
{
    torch::NoGradGuard no_grad;

    // auto data_set = NNDataSet(this->mapValVec).map(torch::data::transforms::Stack<>());
    // auto data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
    //     std::move(data_set), torch::data::DataLoaderOptions().batch_size(5000).workers(4));

    double up_error = 0.0;
    double down_error = 0.0;
    // for (auto &batch : *data_loader)
    // {
    torch::Tensor map_v = torch::tensor(this->mapValVec, at::kCPU).reshape({-1, 1});
    torch::Tensor pos_v = torch::tensor(this->positionVec, at::kCPU).reshape({-1, 1});
    torch::Tensor pre = nnModel->forward(map_v);

    torch::Tensor error = (pos_v - pre) * (double)this->mapValVec.size();

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
    // }
    this->error_bound.clear();

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
    if (floatParam.size() != 151)
        return;
    this->initialIndexModelParam(floatParam);
    this->initialIndexModelTrainedParam();
}

int IndexModel::preFastPosition(double map_val)
{
    Eigen::Matrix<float, 1, 1> input{{(float)map_val}};
    result = input * input_weight.transpose();
    result = result + input_bias;
    result = result.cwiseMax(0);
    result = result * hiden_weight.transpose();
    result = result + hiden_bias;
    // result = result.cwiseMax(0) + 0.01 * result.cwiseMin(0);
    double pre_pos = result(0, 0);
    return (int)(pre_pos * this->mapValVec.size());
}

void IndexModel::getParamFromScoket(int serverPort, vector<MetaData> &metadataVec)
{
    // means retrain
    // this->refreshMetaDataVec(metadataVec);
    vector<float> floatParam;
    vector<double> sampleData;
    vector<int> keyIdx;

    int sampleBatchSize = 1024;

    for (int i = 0; i < this->mapValVec.size(); i++)
        keyIdx.push_back(i);

    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(keyIdx.begin(), keyIdx.end(), g);

    for (int i = 0; i < sampleBatchSize; i++)
        sampleData.push_back(this->mapValVec[keyIdx[i]]);

    sort(sampleData.begin(), sampleData.end());

    connectMetaServer(floatParam, sampleData, serverPort, 151);

    this->initialIndexModelParam(floatParam);
    this->buildModel();
    // this->trainModel();
    this->initialIndexModelTrainedParam();
}

void IndexModel::initialIndexModelParam(vector<float> &floatParam)
{
    torch::Tensor modelTensorParam = torch::tensor(floatParam, torch::dtype(torch::kFloat32));
    torch::Tensor inputLayerWeight =
        modelTensorParam.index({torch::indexing::Slice(torch::indexing::None, 50, torch::indexing::None)})
            .view({-1, 1});
    torch::Tensor inputLayerBias =
        modelTensorParam.index({torch::indexing::Slice(50, 100, torch::indexing::None)}).view({-1});
    torch::Tensor hidenLayerWeight =
        modelTensorParam.index({torch::indexing::Slice(100, 150, torch::indexing::None)}).view({1, -1});
    torch::Tensor hidenLayerBias =
        modelTensorParam.index({torch::indexing::Slice(150, 151, torch::indexing::None)}).view({-1});
    nnModel->input->weight.set_data(inputLayerWeight);
    nnModel->input->bias.set_data(inputLayerBias);
    nnModel->hiden->weight.set_data(hidenLayerWeight);
    nnModel->hiden->bias.set_data(hidenLayerBias);
}

void IndexModel::initialIndexModelTrainedParam()
{
    vector<float> inputLayerWeight_v(this->nnModel->input->weight.data_ptr<float>(),
                                     this->nnModel->input->weight.data_ptr<float>() +
                                         this->nnModel->input->weight.numel());
    vector<float> inputLayerBias_v(this->nnModel->input->bias.data_ptr<float>(),
                                   this->nnModel->input->bias.data_ptr<float>() + this->nnModel->input->bias.numel());
    vector<float> hidenLayerWeight_v(this->nnModel->hiden->weight.data_ptr<float>(),
                                     this->nnModel->hiden->weight.data_ptr<float>() +
                                         this->nnModel->hiden->weight.numel());
    vector<float> hidenLayerBias_v(this->nnModel->hiden->bias.data_ptr<float>(),
                                   this->nnModel->hiden->bias.data_ptr<float>() + this->nnModel->hiden->bias.numel());

    vector<float> floatParam;
    floatParam.insert(floatParam.end(), inputLayerWeight_v.begin(), inputLayerWeight_v.end());
    floatParam.insert(floatParam.end(), inputLayerBias_v.begin(), inputLayerBias_v.end());
    floatParam.insert(floatParam.end(), hidenLayerWeight_v.begin(), hidenLayerWeight_v.end());
    floatParam.insert(floatParam.end(), hidenLayerBias_v.begin(), hidenLayerBias_v.end());

    Eigen::MatrixXf modelParamVec = Eigen::Map<Eigen::Matrix<float, 151, 1>>(floatParam.data());
    this->input_weight = modelParamVec.block(0, 0, 50, 1);
    this->input_bias = modelParamVec.block(50, 0, 50, 1).transpose();
    this->hiden_weight = modelParamVec.block(100, 0, 50, 1).transpose();
    this->hiden_bias = modelParamVec.block(150, 0, 1, 1);
}

void IndexModel::refreshMetaDataVec(vector<MetaData> &metadataVec)
{
    this->mapValVec.clear();
    this->positionVec.clear();
    double idx = 0;
    for (auto &meta_data : metadataVec)
    {
        this->mapValVec.push_back(meta_data.map_val);
        this->positionVec.push_back(idx / metadataVec.size());
        idx++;
    }
}

void IndexModel::trainModel()
{
    torch::Tensor map_v = torch::tensor(this->mapValVec, at::kCUDA).reshape({-1, 1});
    torch::Tensor pos_v = torch::tensor(this->positionVec, at::kCUDA).reshape({-1, 1});
    torch::optim::SGD optimizer(nnModel->parameters(), 0.005);
    this->nnModel->to(device);
    for (size_t epoch = 0; epoch < 200; epoch++)
    {
        // size_t batch_index = 0;
        optimizer.zero_grad();
        torch::Tensor loss = torch::mse_loss(this->nnModel->forward(map_v), pos_v);
        loss.to(at::kCUDA);
        loss.backward();
        optimizer.step();
        // std::cout << "Epoch: " << epoch << "batch:"<< batch_index++ << "loss: " << std::setprecision(5) <<
        // loss.item<double>() << std::endl; loss_item += loss.item<double>();

        // std::cout << "epoch: " << epoch << " loss:" << std::setprecision(5) << loss_item << std::endl;
    }

    nnModel->to(torch::kCPU);
    this->getErrorBound();
}
