import torch
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import random

# from model import LeoModel
from model import LeoModel

import time
import sys

device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")


def calcu_map_val(data):
    map_val = 1.0
    for i in data:
        map_val *= i
    return map_val


def leo_cal_weight(model_path, map_vals, batch_size):
    leo_model = torch.load(model_path, map_location=device)
    sample_index = random.sample(range(len(map_vals)), batch_size)
    sample_index.sort()
    sample_map_vals = [map_vals[i] for i in sample_index]
    input = torch.FloatTensor(sample_map_vals).to(device)
    input = input.view(1, 1, input.shape[-1], 1)
    weight = leo_model.encode_decode(input)
    weight = weight.view(weight.shape[-1])
    return weight


def _leo_cal_weight_(leo_model, map_vals, batch_size):
    # leo_model = torch.load(model_path, map_location=device)
    sample_index = random.sample(range(len(map_vals)), batch_size)
    sample_index.sort()
    sample_map_vals = [map_vals[i] for i in sample_index]
    input = torch.FloatTensor(sample_map_vals).to(device)
    input = input.view(1, input.shape[-1], 1)
    weight = leo_model.encode_decode(input)
    weight = weight.view(weight.shape[-1])
    return weight


def generate_all_model_initial_parameter(data_path, model_path, save_path):
    raw_data_path_list = []
    save_path_list = []
    for data_name in os.listdir(data_path):
        raw_data_path_list.append(data_path + data_name)
        save_path_list.append(save_path + data_name)
    leo_model = torch.load(model_path, map_location=device)
    for raw_data_path, save_data_path in tqdm(zip(raw_data_path_list, save_path_list)):
        # raw_data = np.genfromtxt(raw_data_path, delimiter=",")
        raw_df = pd.read_csv(raw_data_path, header=None, delimiter=",")
        raw_data = raw_df.to_numpy()
        weight = _leo_cal_weight_(leo_model, raw_data[:, -1], 1024)
        weight = weight.cpu().detach().numpy()
        np.savetxt(save_data_path, weight, delimiter=",")


if __name__ == "__main__":
    # leo_cal_weight(
    #     "",
    #     [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
    # )
    # generate_all_model_initial_parameter(
    #     "",
    #     "",
    #     "",
    # )
    # generate_all_model_initial_parameter(
    #     "",
    #     "",
    #     "",
    # )
    # generate_all_model_initial_parameter(
    #     "",
    #     "",
    #     "",
    # )
    arg = sys.argv
    s = time.time()
    generate_all_model_initial_parameter(
        arg[1],
        arg[2],
        arg[3],
    )
    e = time.time()
    print("generate param time: ", e - s)
    # generate_all_model_initial_parameter(
    #     "",
    #     "",
    #     "",
    # )
    # generate_all_model_initial_parameter(
    #     "",
    #     "",
    #     "",
    # )
