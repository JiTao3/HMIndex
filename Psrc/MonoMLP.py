import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np
import sys, os

from certify import certify_grad_with_gurobi, certify_neural_network
from torch.multiprocessing import Pool

import time

from statistics import mean


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MonoMLP(nn.Module):
    def __init__(self, mono_feature=1, mono_hiden=50) -> None:
        super(MonoMLP, self).__init__()
        self.mono_fc_in = nn.Linear(mono_feature, mono_hiden, bias=True)
        self.mono_fc_last = nn.Linear(mono_hiden, 1, bias=True)
        self.mono_feature = mono_feature
        self.mono_hiden = mono_hiden

    def forward(self, x):
        x = self.mono_fc_in(x)
        x = F.relu(x)
        x = self.mono_fc_last(x)
        # x = F.hardtanh(x, min_val=0.0, max_val=1.0)
        return x

    def reg_forward(self, num=512):
        in_list = []
        out_list = []
        input_feature = torch.rand(num, self.mono_feature).to(device)
        input_feature.requires_grad = True
        in_list.append(input_feature)
        x = self.forward(input_feature)
        out_list.append(x)
        return in_list, out_list

    def init_weight(self, weight):
        self.mono_fc_in.weight.data = weight[:50].view(-1, 1)
        self.mono_fc_in.bias.data = weight[50:100].view(-1)
        self.mono_fc_last.weight.data = weight[100:150].view(1, -1)
        self.mono_fc_last.bias.data = weight[150:151].view(-1)

    def model_weight_all(self):
        weight = self.mono_fc_in.weight.data.view(-1)
        weight = torch.cat((weight, self.mono_fc_in.bias.data.view(-1)))
        weight = torch.cat((weight, self.mono_fc_last.weight.data.view(-1)))
        weight = torch.cat((weight, self.mono_fc_last.bias.data.view(-1)))
        return weight.cpu().numpy()



def generate_regularizer(in_list, out_list):
    length = len(in_list)
    reg_loss = 0.0
    min_derivative = 0.0
    for i in range(length):
        xx = in_list[i]
        yy = out_list[i]
        for j in range(yy.shape[1]):
            grad_input = torch.autograd.grad(
                torch.sum(yy[:, j]), xx, create_graph=True, allow_unused=True
            )[0]
            grad_input_neg = -grad_input
            grad_input_neg += 0.2
            grad_input_neg[grad_input_neg < 0.0] = 0.0
            if min_derivative < torch.max(grad_input_neg**2):
                min_derivative = torch.max(grad_input_neg**2)
    reg_loss = min_derivative
    return reg_loss


class IndexDataSet(Dataset):
    def __init__(self, data) -> None:
        super().__init__()
        self.data = data
        self.length = float(len(self.data))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        sample = {
            "map_val": torch.FloatTensor([self.data[index]]),
            "position": torch.FloatTensor([index]) / (self.length - 1),
        }
        return sample


def train(raw_data_path, model_param_path, save_path):

    regression_model = MonoMLP(1, 50)
    model_weight = np.genfromtxt(model_param_path, delimiter=",")
    model_weight = torch.tensor(model_weight, dtype=torch.float, requires_grad=True)

    regression_model.init_weight(model_weight)
    regression_model.to(device)

    raw_data = np.genfromtxt(raw_data_path, delimiter=",")
    train_data = IndexDataSet(raw_data[:, -1])
    train_data_loader = DataLoader(
        train_data, batch_size=256, shuffle=True, pin_memory=True
    )
    lossf = nn.MSELoss()
    optimizer = torch.optim.Adam(regression_model.parameters(), lr=0.005)
    num_epoch = 200
    for epoch in range(num_epoch):
        for sample in train_data_loader:
            x = sample["map_val"].to(device)
            y = sample["position"].to(device)
            optimizer.zero_grad()
            y_p = regression_model(x)
            loss = lossf(y_p, y)

            in_list, out_list = regression_model.reg_forward(1024)
            reg_loss = generate_regularizer(in_list, out_list)

            total_loss = loss + 1e-2* reg_loss

            total_loss.backward()
            optimizer.step()
    model_error(raw_data_path, regression_model)
    if save_path:
        np_weight = regression_model.model_weight_all()
        np.savetxt(save_path, np_weight, delimiter=",")
    mono_flag = certify_neural_network(regression_model, mono_feature_num=1)
    if mono_flag:
        print("Certified Monotonic")
    else:
        print("Not Monotonic")
    return regression_model


def model_error(raw_data_path, regression_model):
    # regression_model = NNRegressionModel(1, 100, 1)
    # model_weight = np.genfromtxt(model_param_path, delimiter=",")

    regression_model.to(torch.device("cpu"))
    raw_data = np.genfromtxt(raw_data_path, delimiter=",")

    train_data = IndexDataSet(raw_data[:, -1])
    train_data_loader = DataLoader(train_data, batch_size=raw_data.shape[0])
    torch.no_grad()
    lower_error = 0
    upper_error = 0
    model_error = []
    for sample in train_data_loader:
        x = sample["map_val"]
        y = sample["position"]
        y_p = regression_model(x)
        for y_g, ypre in zip(y, y_p):
            error = (y_g - ypre) * raw_data.shape[0]
            if error > 0 and error > upper_error:
                # model_error.append(error.item())
                upper_error = error.item()
            elif error < 0 and lower_error < -error:
                # model_error.append(-error.item())
                lower_error = -error.item()
    print(lower_error, upper_error, end=' ')

    return [upper_error, lower_error]


def train_all():
    torch.multiprocessing.set_start_method("spawn")
    dataset_name = sys.argv[1]
    print("data set name: ", dataset_name)

    raw_data_path = ""
    init_param_path = ""
    save_data_path = ""

    if dataset_name == "osm_ne_us":
        raw_data_path = "/data/jitao/dataset/OSM_US_NE/split/"
        init_param_path = "/data/jitao/dataset/OSM_US_NE/initial_param_mono/"
        save_data_path = "/data/jitao/dataset/OSM_US_NE/trained_mono/"
    elif dataset_name == "tiger":
        raw_data_path = "/data/jitao/dataset/Tiger/split/"
        init_param_path = "/data/jitao/dataset/Tiger/initial_param_mono/"
        save_data_path = "/data/jitao/dataset/Tiger/trained_mono/"
    elif dataset_name == "uniform":
        raw_data_path = "/data/jitao/dataset/uniform/split/"
        init_param_path = "/data/jitao/dataset/uniform/initial_param_mono/"
        save_data_path = "/data/jitao/dataset/uniform/trained_mono/"
    elif dataset_name == "skewed":
        raw_data_path = "/data/jitao/dataset/skewed/split/"
        init_param_path = "/data/jitao/dataset/skewed/initial_param_mono/"
        save_data_path = "/data/jitao/dataset/skewed/trained_mono/"
    else:
        print("error distribution")

    data_name_list = os.listdir(raw_data_path)
    training_pool = Pool(30)
    for index, data_name in enumerate(data_name_list):
        training_pool.apply_async(
            train,
            (
                raw_data_path + data_name,
                init_param_path + data_name,
                save_data_path + data_name
            )
        )
    training_pool.close()
    training_pool.join()


def compareErrorBound():
    dataset_name = 'osm_ne_us'
    raw_data_path = ""
    save_param_path = ""

    if dataset_name == "osm_ne_us":
        raw_data_path = "/data/jitao/dataset/OSM_US_NE/split/"
        save_param_path = "/data/jitao/dataset/OSM_US_NE/trained_mono/"
    elif dataset_name == "tiger":
        raw_data_path = "/data/jitao/dataset/Tiger/split/"
        save_param_path = "/data/jitao/dataset/Tiger/trained_mono/"
    elif dataset_name == "uniform":
        raw_data_path = "/data/jitao/dataset/uniform/split/"
        save_param_path = "/data/jitao/dataset/uniform/trained_mono/"
    elif dataset_name == "skewed":
        raw_data_path = "/data/jitao/dataset/skewed/split/"
        save_param_path = "/data/jitao/dataset/skewed/trained_mono/"
    else:
        print("error distribution")


    data_name_list = os.listdir(raw_data_path)
    print(len(data_name_list))
    randomModelErrorList = []
    metaModelErrorList = []
    for dataName in data_name_list:
        regression_model = MonoMLP(1, 50)
        model_weight = np.genfromtxt(save_param_path + dataName, delimiter=",")
        model_weight = torch.FloatTensor(model_weight)
        regression_model.init_weight(model_weight)
        
        meta_error = model_error(
            raw_data_path + dataName, regression_model
        )
        metaModelErrorList.append(meta_error)
        print("meta error: ", meta_error)
    print(
        "meta param: ",
        mean([error[0] for error in metaModelErrorList]),
        " -- ",
        mean([error[1] for error in metaModelErrorList]),
    )


if __name__ == "__main__":
    # s = time.time()
    # train_all()
    # e = time.time()
    # print("train time: ", e - s)

    compareErrorBound()

    # train(
    #     "/data/jitao/dataset/OSM_US_NE/split/1233.csv", 
    #     "/data/jitao/dataset/OSM_US_NE/initial_param_mono/1233.csv", 
    #     "/data/jitao/dataset/OSM_US_NE/trained_mono/1233.csv"
    #     )
