#pragma once
#include <Eigen/Core>
#include <Eigen/Dense>
#include <random>
#include <torch/torch.h>
#include <vector>

#include "FileReader.h"
#include "MetaClient.h"
#include "MetaData.h"
// #include "Utils.h"

using namespace std;

const int modelInputSize = 1;
const int modelHidenSize = 100;
const int modelOutputSize = 1;

class NNDataSet : public torch::data::Dataset<NNDataSet>
{
  private:
    std::vector<float>* map_vals;
    std::vector<float>* pos_vals;
    int length;

  public:
    NNDataSet(std::vector<float>& _map_vals, std::vector<float>& _pos_vals);

    torch::data::Example<> get(size_t index) override;

    // Return the length of data
    torch::optional<size_t> size() const override;
};

struct NNModel : torch::nn::Module
{
    NNModel(int64_t input_s, int64_t hiden_s, int64_t output_s = 1)
    {
        // input_size = input_s;
        // hiden_size = hiden_s;
        // output_size = hiden_s;
        input = register_module("input_l", torch::nn::Linear(input_s, hiden_s));
        ac_f1 = register_module("ac_f1", torch::nn::LeakyReLU());
        hiden = register_module("hiden_l", torch::nn::Linear(hiden_s, output_s));
        ac_f2 = register_module("ac_f2", torch::nn::LeakyReLU());
    }

    torch::Tensor forward(torch::Tensor x)
    {
        x = ac_f1(input->forward(x));
        x = ac_f2(hiden->forward(x));
        return x;
    }

    torch::nn::Linear input{nullptr}, hiden{nullptr};
    torch::nn::LeakyReLU ac_f1, ac_f2;
    // int input_size, hiden_size, output_size;
};

class IndexModel
{
  public:
    std::vector<float> mapValVec;
    std::vector<float> positionVec;
    NNModel *nnModel = nullptr;
    std::vector<int> error_bound;
    Eigen::Matrix<float, modelHidenSize, modelInputSize> input_weight;
    Eigen::Matrix<float, 1, modelHidenSize> input_bias;
    Eigen::Matrix<float, modelOutputSize, modelHidenSize> hiden_weight;
    Eigen::Matrix<float, 1, modelOutputSize> hiden_bias;
    Eigen::MatrixXf hidenActive1 = Eigen::MatrixXf::Zero(1, modelHidenSize);
    Eigen::MatrixXf hidenActive2 = Eigen::MatrixXf::Zero(1, modelOutputSize);
    Eigen::MatrixXf result;

  public:
    IndexModel()
    {
        nnModel = nullptr;
    }

    IndexModel(std::vector<MetaData> *metadataVec);
    IndexModel(std::vector<MetaData> &metadataVec);
    IndexModel(std::vector<double> &mapvalvec);
    ~IndexModel();

    void buildModel();

    int prePosition(double meta_key_map_val);
    int preFastPosition(double);

    void loadParameter(string paramPath);

    void getErrorBound();

    void getParamFromScoket(int serverPort, vector<MetaData> &metadataVec);

    void initialIndexModelParam(vector<float> &floatParam);
    void initialIndexModelTrainedParam();
    void trainModel();
    void refreshMetaDataVec(std::vector<MetaData> &metadataVec);
};