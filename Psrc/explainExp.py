import numpy as np
from numpy.core.fromnumeric import mean


def explainExpLog(log_path):
    log_file = open(log_path)
    lines = log_file.readlines()
    error_data = []
    for line in lines:
        if line.startswith(("leaf", "grid")):
            split_line = line.split(",")
            min_bound = split_line[0].split(" ")[-1]
            max_bound = split_line[1].split(" ")[1]
            min_bound = int(min_bound)
            max_bound = int(max_bound)
            error_data.append([min_bound, max_bound])
    np_error_data = np.array(error_data)
    print(np_error_data.shape)
    print(mean(np_error_data[:, 0]), mean(np_error_data[:, 1]))

def explainExpLog(log_path):
    log_file = open(log_path)
    lines = log_file.readlines()
    error_data = []
    for line in lines:
        if len(line.split("error:"))==3:
            min_bound = line.split("error:")[1].split(" ")[2]
            max_bound = line.split("error:")[2].split(" ")[2]
            min_bound = float(min_bound)
            max_bound = float(max_bound)
            error_data.append([min_bound, max_bound])
    np_error_data = np.array(error_data)
    print(np_error_data.shape)
    print(mean(np_error_data[:, 0]), mean(np_error_data[:, 1]))
        

if __name__=='__main__':
    explainExpLog("/home/jitao/cell_tree/log/randon_param_training_osm_cn_split2.log")
