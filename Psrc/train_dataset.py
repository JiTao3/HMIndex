import numpy as np
import pandas as pd
from random import sample
import os
import torch


# class IndexDataset(MetaDataset):
#     def __init__(
#         self,
#         tasks,
#         max_size,
#         num_task=200000,
#         N=5,
#         K=512,
#         embedding_size=1,
#         transform=None,
#         target_transform=None,
#         dataset_transform=None,
#     ):
#         super(IndexDataset, self).__init__(
#             meta_split="train",
#             target_transform=target_transform,
#             dataset_transform=dataset_transform,
#         )
#         self.tasks = tasks
#         self.num_task = num_task
#         self.K = K
#         self.N = N
#         self.embedding_size = embedding_size
#         self.max_size_task = max_size
#         self.transform = transform

#     def __len__(self):
#         return self.num_task

#     def __getitem__(self, index):
#         task_data = self.tasks[index]
#         task = IndexTask(
#             index=index,
#             task_data=task_data,
#             N=self.N,
#             K=self.K,
#             max_size=self.max_size_task,
#         )
#         if self.dataset_transform is not None:
#             task = self.dataset_transform(task)

#         return task


# class IndexTask(Task):
# def __init__(
#     self,
#     index,
#     task_data,
#     N,
#     K,
#     max_size,
#     transform=None,
#     target_transform=None,
# ):
#     super(IndexTask, self).__init__(index, None)
#     self.N = N
#     self.K = K
#     self.max_size = max_size
#     self.index = index
#     self.task_data = task_data
#     self.randomCellIndex = sample(range(len(task_data)), N)

#     # self.map_vals = task_data[:, -1]
#     # self.positions = [
#     #     i / (self.map_vals.shape[0] - 1)
#     #     for i in range(self.map_vals.shape[0])
#     # ]
#     # sample_idx = sample(range(self.map_vals.shape[0]), max_size)
#     # sample_idx.sort()
#     # self.map_vals = np.array([self.map_vals[i] for i in sample_idx])
#     # self.positions = np.array([self.positions[i] for i in sample_idx])
#     # self.map_vals = np.expand_dims(self.map_vals, 1)
#     # self.positions = np.expand_dims(self.positions, 1)
#     self.transform = transform
#     self.target_transform = target_transform

# def __len__(self):
#     return self.map_vals.shape[0]

# def __getitem__(self, index):
#     map_val = self.map_vals[index]
#     position = self.positions[index]
#     if self.transform is not None:
#         map_val = self.transform(map_val)
#     if self.target_transform is not None:
#         position = self.target_transform(position)
#     return (map_val, position)


def get_batch(taskDatas, K, batchSize, N, device):
    data_split = ["train", "val"]
    batch = {}
    for d in data_split:
        batch[d] = {"input": [], "target": [], "name": []}
    for b in range(batchSize):
        choiceClass = sample(range(len(taskDatas)), N)
        inp = {"train": [], "val": []}
        tgt = {"train": [], "val": []}

        for class_idx, choice_data_idx in enumerate(choiceClass):
            # each N
            class_data = taskDatas[choice_data_idx][:, -1]
            data_shape = class_data.shape[0]
            class_position = [i / (data_shape - 1) for i in range(data_shape)]

            random_sample_idx = sample(range(data_shape), 3 * K)

            # 0-k train k-2k val
            task_inp = {
                "train": [
                    [class_data[idx]] for idx in sorted(random_sample_idx[:K])
                ],
                "val": [
                    [class_data[idx]] for idx in sorted(random_sample_idx[K:])
                ],
            }

            task_tar = {
                "train": [
                    [class_position[idx]] for idx in sorted(random_sample_idx[:K])
                ],
                "val": [
                    [class_position[idx]] for idx in sorted(random_sample_idx[K:])
                ],
            }

            for d in data_split:
                inp[d].append(task_inp[d])
                tgt[d].append(task_tar[d])
        for d in data_split:
            batch[d]["input"].append(np.asarray(inp[d]))  # [N*K*EMBedding]
            batch[d]["target"].append(np.asarray(tgt[d]))
    # return batch
    # to tensor
    for d in data_split:
        d_data = batch[d]
        np_input_data = np.asarray(d_data["input"])
        np_target_data = np.asarray(d_data["target"])
        tensor_input_data = torch.FloatTensor(np_input_data).squeeze(1).to(device)
        tensor_target_data = torch.FloatTensor(np_target_data).squeeze(1).to(device)
        batch[d]["input"] = tensor_input_data
        batch[d]["target"] = tensor_target_data
    return batch


def loadAllSplitIndexTask(data_path):
    split_task_l = os.listdir(data_path)
    split_task_l.sort(key=lambda x: int(x.split(".")[0]))
    print("load task")
    return [
        # np.genfromtxt(data_path + split_path, delimiter=",")
        pd.read_csv(
            data_path + split_path, header=None, delimiter=","
        ).to_numpy()
        for split_path in split_task_l
    ]


if __name__ == "__main__":
    tasks = loadAllSplitIndexTask("")
    get_batch(tasks, 256, 12, 1)
