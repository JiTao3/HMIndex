#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <vector>
#include <IndexModel.h>

using namespace std;

int main()
{
  vector<double> map_vec = {1.0};
  IndexModel *model = new IndexModel(map_vec);
  model->loadParameter("/data/jitao/dataset/OSM/new_trained_model_param_for_split2/10.csv");
  // torch::Tensor map_val = torch::tensor({0.3}, {torch::kFloat32}).view({-1, 1});
  double forward_cal = model->prePosition(0.3);
  double fast_cal = model->preFastPosition(0.3);
  cout << forward_cal << ", " << fast_cal << endl;
}