import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

from regression_model import NNRegressionModel

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


class EncoderConv(nn.Module):
    def __init__(self, dropout, latent_size):
        super(EncoderConv, self).__init__()
        self.conv1 = nn.Conv1d(1, 128, 2)
        self.conv2 = nn.Conv1d(1, 128, 16)
        self.conv3 = nn.Conv1d(1, 128, 128)
        # self.conv11 = nn.Conv1d(128, 128, 128)
        # self.conv21 = nn.Conv1d(128, 128, 16)
        # self.conv31 = nn.Conv1d(128, 128, 2)

        self.dropout = nn.Dropout(dropout)
        self.fcn = nn.Sequential(
            nn.Linear(3 * 128 + 1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, latent_size),
        )
        # self.bn1 = nn.BatchNorm1d(128)
        # self.bn2 = nn.BatchNorm1d(128)
        # self.bn3 = nn.BatchNorm1d(128)
        # F.batch_norm
        # self.fc1 = nn.Linear(3 * 128 + 1024, 1024)
        # self.fc2 = nn.Linear(1024, 512)
        # self.fc3 = nn.Linear(512, latent_size)

    def forward(self, x):
        x1 = F.relu(self.conv1(x.permute(0, 2, 1)))  # [12, 128, 1023]
        x2 = F.relu(self.conv2(x.permute(0, 2, 1)))
        x3 = F.relu(self.conv3(x.permute(0, 2, 1)))

        # x1 = F.relu(self.conv11(x1)) 
        # x2 = F.relu(self.conv21(x2))
        # x3 = F.relu(self.conv31(x3))
        
        x1 = F.max_pool1d(x1, x1.size(2))
        x2 = F.max_pool1d(x2, x2.size(2))
        x3 = F.max_pool1d(x3, x3.size(2))
        x = torch.cat([x1, x2, x3, x], 1).permute(0, 2, 1)
        x = self.dropout(x)
        x = self.fcn(x)
        return x


class LeoModel(nn.Module):
    def __init__(self, hidden_size, drop_out, inner_lr, fintune_lr, cuda):
        super(LeoModel, self).__init__()
        self.hidden_size = hidden_size
        self.model_size = 151
        self._cuda = cuda
        # ? parameter of nn.Conv2d and layer size of encoder

        self.encoder = EncoderConv(drop_out, hidden_size)
        self.decoder = nn.Linear(self.hidden_size, self.model_size)
        # self.decoder = nn.Sequential(
        #     nn.Linear(self.hidden_size, self.hidden_size * 4),
        #     nn.ReLU(),
        #     nn.Linear(self.hidden_size * 4, self.hidden_size * 2),
        #     nn.ReLU(),
        #     nn.Linear(self.hidden_size * 2, self.model_size),
        # )
        self.dropout = nn.Dropout(p=drop_out)
        self.inner_l_rate = nn.Parameter(torch.FloatTensor([inner_lr]))
        self.fintune_lr = nn.Parameter(torch.FloatTensor([fintune_lr]))

    def encode(self, inputs):

        inputs = self.dropout(inputs)
        out = self.encoder(inputs)

        return out

    def decode(self, latents):
        weights = self.decoder(latents)
        # regression_weight = self.sample(weights, self.model_size)
        return weights

    def predict(self, inputs, weights):
        # 1. weight->model
        # 2. cal output
        # input [12, 1024, 1]
        input_weight = weights[:, :, :50]  # .view(-1, 1)

        # input_weight = torch.transpose(input_weight, -1, -2)
        input_bias = weights[:, :, 50:100]  # .view(-1)

        hiden_weight = weights[:, :, 100:150].permute(0, 2, 1)

        hiden_bias = weights[:, :, 150:151]  # .view(-1)
        # hiden_bias = hiden_bias.view(
        #     hiden_bias.shape[0], 1, 1, hiden_bias.shape[-1]
        # )
        # output = torch.bmm(inputs, input_weight)

        output = inputs @ input_weight
        output = output + input_bias
        output = F.relu(output)
        # a = F.linear(inputs, input_weight, input_bias)
        # b = F.linear(a, hiden_weight, hiden_bias)
        output = output @ hiden_weight
        output = output + hiden_bias
        # output = F.leaky_relu(output)
        return output

    def reg_forward(self, num, batch_size, weights):
        in_list = []
        out_list = []
        input_feature = torch.rand(batch_size, num, 1).to(device)
        input_feature.requires_grad = True
        in_list.append(input_feature)
        out_feature = self.predict(input_feature, weights)
        out_list.append(out_feature)
        return in_list, out_list

    def generate_regularizer(self, in_list, out_list):
        length = len(in_list)
        reg_loss = 0.0
        min_derivative = 0.0
        for i in range(length):
            xx = in_list[i]
            yy = out_list[i]
            for j in range(yy.shape[-1]):
                grad_input = torch.autograd.grad(
                    torch.sum(yy[:, :, j]), xx, create_graph=True, allow_unused=True
                )[0]
                grad_input_neg = -grad_input
                grad_input_neg += 0.2
                grad_input_neg[grad_input_neg < 0.0] = 0.0
                if min_derivative < torch.max(grad_input_neg**2):
                    min_derivative = torch.max(grad_input_neg**2)
        reg_loss = min_derivative
        return reg_loss

    def cal_target_loss(self, input, regression_weight, target):
        output = self.predict(input, regression_weight)
        in_list, out_list = self.reg_forward(1024, input.shape[0], regression_weight)
        reg_loss = self.generate_regularizer(in_list, out_list)
        criterion = nn.MSELoss()
        total_loss = criterion(output, target)
        total_loss += 1e-4 * reg_loss
        return total_loss

    def encode_decode(self, input):
        latents = self.encode(input)
        weight = self.decode(latents)
        return weight


if __name__ == "__main__":
    pass
