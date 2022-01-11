import torch
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
# from model import LeoModel
from model import LeoModel

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
    input = input.view(1, 1, input.shape[-1], 1)
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
    for raw_data_path, save_data_path in tqdm(
        zip(raw_data_path_list, save_path_list)
    ):
        # raw_data = np.genfromtxt(raw_data_path, delimiter=",")
        raw_df = pd.read_csv(raw_data_path, header=None, delimiter=",")
        raw_data = raw_df.to_numpy()
        weight = _leo_cal_weight_(leo_model, raw_data[:, -1], 128)
        weight = weight.cpu().detach().numpy()
        np.savetxt(save_data_path, weight, delimiter=",")


if __name__ == "__main__":
    # leo_cal_weight(
    #     "/home/jitao/leo_index/model/osm_train2.pt",
    #     [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
    # )
    # generate_all_model_initial_parameter(
    #     "/data/jitao/dataset/OSM/split1/",
    #     "/home/jitao/leo_index/model/osm_train9.pt",
    #     "/data/jitao/dataset/OSM/model_parameter1/",
    # )
    # generate_all_model_initial_parameter(
    #     "/data/jitao/dataset/OSM_US_NE/split/",
    #     "/home/jitao/leo_index/log/osm_en_us/osm_en_usosm_train17.pt",
    #     "/data/jitao/dataset/OSM_US_NE/model_parameter/",
    # )
    # generate_all_model_initial_parameter(
    #     "/data/jitao/dataset/OSM_US_NE/split/",
    #     "/home/jitao/leo_index/log/osm_en_us/osm_en_us_1.pt",
    #     "/data/jitao/dataset/OSM_US_NE/initial_model_param_for_split/",
    # )
    # generate_all_model_initial_parameter(
    #     "/data/jitao/dataset/Tiger/split/",
    #     "/home/jitao/leo_index/log/tiger/tiger_1.pt",
    #     "/data/jitao/dataset/Tiger/initial_model_param_for_split/",
    # )
    # generate_all_model_initial_parameter(
    #     "/data/jitao/dataset/uniform/split/",
    #     "/home/jitao/leo_index/log/uniform/uniform1.pt",
    #     "/data/jitao/dataset/uniform/initial_model_param_for_split/",
    # )
    generate_all_model_initial_parameter(
        "/data/jitao/dataset/skewed/split/",
        "/home/jitao/leo_index/log/skewed/skewed1.pt",
        "/data/jitao/dataset/skewed/initial_model_param_for_split/",
    )
