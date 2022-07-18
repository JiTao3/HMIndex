from statistics import mean
import sys
import os
import torch
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, dataloader

import numpy as np

from torch.multiprocessing import Pool

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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


class NNRegressionModel(nn.Module):
    def __init__(self, input_s, hiden_s, output_s=1) -> None:
        super().__init__()
        self.input = nn.Linear(input_s, hiden_s)
        self.ac_f1 = nn.LeakyReLU()
        self.hiden = nn.Linear(hiden_s, output_s)
        self.ac_f2 = nn.LeakyReLU()

    def init_weight(self, weight):
        self.input.weight.data = weight[:100].view(-1, 1)
        self.input.bias.data = weight[100:200].view(-1)
        self.hiden.weight.data = weight[200:300].view(1, -1)
        self.hiden.bias.data = weight[300:301].view(-1)

    def forward(self, x):
        x = self.input(x)
        x = self.ac_f1(x)
        x = self.hiden(x)
        x = self.ac_f2(x)
        return x

    def model_weight_all(self):
        weight = self.input.weight.data.view(-1)
        weight = torch.cat((weight, self.input.bias.data.view(-1)))
        weight = torch.cat((weight, self.hiden.weight.data.view(-1)))
        weight = torch.cat((weight, self.hiden.bias.data.view(-1)))
        return weight.cpu().numpy()


def trainMetaParam(model_param_path, raw_data_path, save_path, index):
    print("index: ", index)
    model_weight = np.genfromtxt(model_param_path, delimiter=",")
    model_weight = torch.tensor(
        model_weight, dtype=torch.float, requires_grad=True
    )
    regression_model = NNRegressionModel(1, 100, 1)
    regression_model.init_weight(model_weight)
    regression_model.to(device)

    raw_data = np.genfromtxt(raw_data_path, delimiter=",")
    train_data = IndexDataSet(raw_data[:, -1])
    train_data_loader = DataLoader(
        train_data, batch_size=raw_data.shape[0], shuffle=True, pin_memory=True
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
            loss.backward()
            optimizer.step()
    np_weight = regression_model.model_weight_all()
    np.savetxt(save_path, np_weight, delimiter=",")
    print("finish: ", index)

    # torch.no_grad()
    # lower_error = 0
    # upper_error = 0
    # for sample in train_data_loader:
    #     x = sample["map_val"].to(device)
    #     y = sample["position"].to(device)
    #     y_p = regression_model(x)
    #     for y_g, ypre in zip(y, y_p):
    #         error = (y_g - ypre) * raw_data.shape[0]
    #         if error > 0 and error > upper_error:
    #             upper_error = error
    #         elif error < 0 and error < lower_error:
    #             lower_error = error
    # print(lower_error, upper_error)
    # return (lower_error.item(), upper_error.item())


def trainRandomInitialModle(raw_data_path, save_path, index):
    regression_model = NNRegressionModel(1, 100, 1)
    # regression_model.init_weight(model_weight)
    regression_model.to(device)

    raw_data = np.genfromtxt(raw_data_path, delimiter=",")
    if len(raw_data) == 0:
        np_weight = regression_model.model_weight_all()
        np.savetxt(save_path, np_weight, delimiter=",")
        return

    train_data = IndexDataSet(raw_data[:, -1])
    train_data_loader = DataLoader(
        train_data, batch_size=raw_data.shape[0], shuffle=True, pin_memory=True
    )
    lossf = nn.MSELoss()
    optimizer = torch.optim.Adam(regression_model.parameters(), lr=0.005)
    num_epoch = 200
    for epoch in range(num_epoch):
        # mse_loss = 0
        for sample in train_data_loader:
            x = sample["map_val"].to(device)
            y = sample["position"].to(device)
            optimizer.zero_grad()
            y_p = regression_model(x)
            loss = lossf(y_p, y)
            # mse_loss += loss.item()
            loss.backward()
            optimizer.step()
        # print("epoch:{}, mse loss {:.6}".format(epoch, mse_loss))
    np_weight = regression_model.model_weight_all()
    # np.savetxt(save_path, np_weight, delimiter=",")
    print("finish: ", index)


def model_error(raw_data_path, model_param_path):
    regression_model = NNRegressionModel(1, 100, 1)
    model_weight = np.genfromtxt(model_param_path, delimiter=",")
    model_weight = torch.FloatTensor(model_weight)
    regression_model.init_weight(model_weight)
    raw_data = np.genfromtxt(raw_data_path, delimiter=",")
    train_data = IndexDataSet(raw_data[:, -1])
    train_data_loader = DataLoader(train_data, batch_size=raw_data.shape[0])
    torch.no_grad()
    lower_error = 0
    upper_error = 0
    for sample in train_data_loader:
        x = sample["map_val"]
        y = sample["position"]
        y_p = regression_model(x)
        for y_g, ypre in zip(y, y_p):
            error = (y_g - ypre) * raw_data.shape[0]
            if error > 0 and error > upper_error:
                upper_error = error
            elif error < 0 and error < lower_error:
                lower_error = error
    # print(lower_error, upper_error)
    return (lower_error.item(), upper_error.item())


def compareErrorBound(dataSetName):
    splitDataPath = ""
    randomParamPath = ""
    metaParamPath = ""

    if dataSetName == "osm_cn":
        splitDataPath = ""
        randomParamPath = (
            ""
        )
        metaParamPath = (
            ""
        )
    elif dataSetName == "osm_ne_us":
        splitDataPath = ""
        randomParamPath = ""
        metaParamPath = (
            ""
        )
    elif dataSetName == "tiger":
        splitDataPath = ""
        randomParamPath = (
            ""
        )
        metaParamPath = (
            ""
        )
    elif dataSetName == "uniform":
        splitDataPath = ""
        randomParamPath = (
            ""
        )
        metaParamPath = (
            ""
        )
    elif dataSetName == "skewed":
        splitDataPath = ""
        randomParamPath = (
            ""
        )
        metaParamPath = (
            ""
        )
    else:
        print("error dataset name")

    data_name_list = os.listdir(splitDataPath)
    print(len(data_name_list))
    randomModelErrorList = []
    metaModelErrorList = []
    for dataName in data_name_list:
        random_error = model_error(
            splitDataPath + dataName, randomParamPath + dataName
        )
        meta_error = model_error(
            splitDataPath + dataName, metaParamPath + dataName
        )
        randomModelErrorList.append(random_error)
        metaModelErrorList.append(meta_error)
        print("random error : ", random_error)
        print("meta error: ", meta_error)
    print(
        "random param: ",
        mean([error[0] for error in randomModelErrorList]),
        " -- ",
        mean([error[1] for error in randomModelErrorList]),
    )
    print(
        "meta param: ",
        mean([error[0] for error in metaModelErrorList]),
        " -- ",
        mean([error[1] for error in metaModelErrorList]),
    )


# if __name__ == "__main__":
#     model_param_path = ""
#     raw_data_path = ""
#     # trainRandomInitialModle(raw_data_path, model_param_path, 2)
#     model_error(raw_data_path, model_param_path)
# model_param_path = (
#     ""
# )
# raw_data_path = ""
# save_path = ""
# trainMetaParam(model_param_path, raw_data_path, save_path, 1)


if __name__ == "__main__":

    # trainMetaParam(
    #     "",
    #     "",
    #     "",
    #     1,
    # )

    torch.multiprocessing.set_start_method("spawn")

    # osm_cn_model_param_path = (
    #     ""
    # )
    # osm_cn_raw_data_path = ""
    # osm_cn_save_path = ""
    # osm_cn_save_random_path = (
    #     ""
    # )

    dataset_name = sys.argv[1]
    print("data set name: ", dataset_name)

    # compareErrorBound(dataset_name)

    # osm_ne_us_model_param_path = (
    #     ""
    # )

    raw_data_path = ""
    save_data_path = ""

    if dataset_name == "osm_ne_us":
        raw_data_path = ""
        save_random_path = (
            ""
        )
    elif dataset_name == "tiger":
        raw_data_path = ""
        save_random_path = (
            ""
        )
    elif dataset_name == "uniform":
        raw_data_path = ""
        save_random_path = (
            ""
        )
    elif dataset_name == "skewed":
        raw_data_path = ""
        save_random_path = (
            ""
        )
    else:
        print("error distribution")

        # osm_ne_us_save_path = (
        #     ""
        # )
    # tiger_model_param_path = (
    #     ""
    # )
    # tiger_raw_data_path = ""
    # tiger_save_path = (
    #     ""
    # )
    # tiger_model_param_path = (
    #     ""
    # )
    # uniform_model_param_path = (
    #     ""
    # )
    # uniform_raw_data_path = ""
    # uniform_save_path = (
    #     ""
    # )
    # skewed_model_param_path = (
    #     ""
    # )
    # skewed_raw_data_path = ""
    # skewed_save_path = (
    #     ""
    # )

    data_name_list = os.listdir(raw_data_path)
    training_pool = Pool(10)
    for index, data_name in enumerate(data_name_list):
        training_pool.apply_async(
            trainRandomInitialModle,
            (
                raw_data_path + data_name,
                save_random_path + data_name,
                index,
            )
            # trainMetaParam,
            # (
            #     skewed_model_param_path + data_name,
            #     skewed_raw_data_path + data_name,
            #     skewed_save_path + data_name,
            #     index,
            # ),
        )
    training_pool.close()
    training_pool.join()

    # skewed_model_param_path = (
    #     ""
    # )
    # skewed_raw_data_path = ""
    # skewed_save_path = (
    #     ""
    # )

    # dataset_split_path = ""
    # dataset_save_random_path = ""

    # if dataset_name == "osm_ne_us":
    #     dataset_split_path = osm_ne_us_raw_data_path
    #     dataset_save_random_path = ""
    # elif dataset_name == "tiger":
    #     dataset_split_path = tiger_raw_data_path
    #     dataset_save_random_path = (
    #         ""
    #     )
    # elif dataset_name == "uniform":
    #     dataset_split_path = uniform_raw_data_path
    #     dataset_save_random_path = (
    #         ""
    #     )
    # elif dataset_name == "skewed":
    #     dataset_split_path = skewed_raw_data_path
    #     dataset_save_random_path = (
    #         ""
    #     )
    # else:
    #     "error data set name"


# nohup python train.py > ../log/OSM_US_NE_trained_modelParam_for_split.log 2>&1 &
# nohup python train.py > ../log/OSM_CN_random_modelParam_for_split.log 2>&1 &
# nohup python train.py > ../log/tiger_meta_modelParam_for_split.log 2>&1 &
# nohup python train.py > ../log/skewed_meta_modelParam_for_split.log 2>&1 &
# nohup python train.py > ../log/v3trainmetatiger.log 2>&1 &
