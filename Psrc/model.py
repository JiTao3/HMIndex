import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

from regression_model import NNRegressionModel

device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")


class LeoModel(nn.Module):
    def __init__(
        self, embedding_size, hidden_size, drop_out, inner_lr, fintune_lr, cuda
    ):
        super(LeoModel, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.model_size = 301
        self._cuda = cuda
        # ? parameter of nn.Conv2d and layer size of encoder
        # self.encoder = nn.Conv2d()
        self.encoder = nn.Sequential(
            nn.Linear(embedding_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        # TODO relation network
        self.relation_net = nn.Sequential(
            nn.Linear(2 * self.hidden_size, 2 * self.hidden_size),
            nn.ReLU(),
            nn.Linear(2 * self.hidden_size, 2 * self.hidden_size),
            nn.ReLU(),
            nn.Linear(2 * self.hidden_size, 2 * self.hidden_size),
            nn.ReLU(),
        )
        self.decoder = nn.Linear(self.hidden_size, 2 * self.model_size)
        self.dropout = nn.Dropout(p=drop_out)
        self.inner_l_rate = nn.Parameter(torch.FloatTensor([inner_lr]))
        self.fintune_lr = nn.Parameter(torch.FloatTensor([fintune_lr]))

    def cal_kl_div(self, latents, mean, var):
        if self._cuda:
            return torch.mean(
                self.cal_log_prob(latents, mean, var)
                - self.cal_log_prob(
                    latents,
                    torch.zeros(mean.size()).cuda(device=device),
                    torch.ones(var.size()).cuda(device=device),
                )
            )
        else:
            return torch.mean(
                self.cal_log_prob(latents, mean, var)
                - self.cal_log_prob(
                    latents, torch.zeros(mean.size()), torch.ones(var.size())
                )
            )

    def cal_log_prob(self, x, mean, var):
        eps = 1e-20
        log_unnormalized = -0.5 * ((x - mean) / (var + eps)) ** 2
        log_normalization = torch.log(var + eps) + 0.5 * math.log(2 * math.pi)

        return log_unnormalized - log_normalization

    def encode(self, inputs):
        # # input size [batch_task, data_size, embed_size=1]
        # input = self.dropout(input)
        # out = self.encoder(input)
        # # b_size, l_batch_size, embed_size = out.size()
        # # TODO: some changes?
        # # x1 = torch.repeat_interleave(out, l_batch_size, dim=1)
        # # x2 = torch.repeat_interleave(out, l_batch_size, dim=1)
        # x = torch.repeat_interleave(out, 2, dim=-1)

        # x = self.relation_net(x)
        # x = torch.mean(x, dim=1)
        # latents = self.sample(x, self.hidden_size)
        # mean, var = x[:, : self.hidden_size], x[:, self.hidden_size :]
        # kl_div = self.cal_kl_div(latents, mean, var)

        # return latents, kl_div
        # inputs -> [batch, N, K, embed_size]
        inputs = self.dropout(inputs)
        out = self.encoder(inputs)
        b_size, N, K, hidden_size = out.size()

        # construct input for relation ner
        t1 = torch.repeat_interleave(out, K, dim=2)
        t1 = torch.repeat_interleave(t1, N, dim=1)
        t2 = out.repeat((1, N, K, 1))
        x = torch.cat((t1, t2), dim=-1)

        # x -> [batch, N*N, K*K, hidden_size]
        x = self.relation_net(x)
        x = x.view(b_size, N, N * K * K, -1)
        x = torch.mean(x, dim=2)

        latents = self.sample(x, self.hidden_size)
        mean, var = x[:, :, : self.hidden_size], x[:, :, self.hidden_size :]
        kl_div = self.cal_kl_div(latents, mean, var)

        return latents, kl_div

    def sample(slef, weights, size):
        mean, var = weights[:, :, :size], weights[:, :, size:]
        z = torch.normal(0.0, 1.0, mean.size()).cuda(device=device)
        return mean + var * z

    def decode(self, latents):
        weights = self.decoder(latents)
        regression_weight = self.sample(weights, self.model_size)
        return regression_weight

    def predict(self, inputs, weights):
        # 1. weight->model
        # 2. cal output
        input_weight = weights[:, :, :100]  # .view(-1, 1)
        input_weight = input_weight.view(
            input_weight.shape[0], 1, input_weight.shape[-1], 1
        )
        input_weight = torch.transpose(input_weight, -1, -2)
        input_bias = weights[:, :, 100:200]  # .view(-1)
        input_bias = input_bias.view(
            input_bias.shape[0], 1, 1, input_bias.shape[-1]
        )
        hiden_weight = weights[:, :, 200:300]  # .view(1, -1)
        hiden_weight = hiden_weight.view(
            hiden_weight.shape[0], 1, hiden_weight.shape[-1], 1
        )
        # hiden_weight = torch.transpose(hiden_weight, -1, -2)

        hiden_bias = weights[:, :, 300:301]  # .view(-1)
        hiden_bias = hiden_bias.view(
            hiden_bias.shape[0], 1, 1, hiden_bias.shape[-1]
        )
        # output = torch.bmm(inputs, input_weight)

        output = inputs @ input_weight
        output = output + input_bias
        output = F.relu(output)
        # a = F.linear(inputs, input_weight, input_bias)
        # b = F.linear(a, hiden_weight, hiden_bias)
        output = output @ hiden_weight
        output = output + hiden_bias
        output = F.relu(output)
        return output

    def cal_target_loss(self, input, regression_weight, target):
        output = self.predict(input, regression_weight)
        criterion = nn.MSELoss()
        total_loss = criterion(output, target)
        # total_loss = 0.0
        # batch_size = regression_weight.shape[0]
        # for i in range(batch_size):
        #     model_weight = regression_weight[i]
        #     _input = input[i]
        #     _target = target[i]
        #     regression_model = NNRegressionModel(1, 100)
        #     regression_model.init_weight(model_weight)
        #     # for j in regression_model.parameters():
        #     #     print(j.shape)
        #     output = regression_model(_input)
        #     loss = criterion(output, _target)
        #     total_loss += loss
        # total_loss = total_loss / batch_size

        return total_loss

    def encode_decode(self, input):
        latents, _ = self.encode(input)
        weight = self.decode(latents)
        return weight


# if __name__=='__main__':
def calcu_map_val(data):
    map_val = 1.0
    for i in data:
        map_val *= i
    return map_val


if __name__ == "__main__":
    model = torch.load("model/simple_train_1k_1.pt", map_location=device)
    raw_data = np.load("/home/jitao/test/2d_len_10000_seed_1.npy")
    map_vals_l = [calcu_map_val(data) for data in raw_data]
    map_vals_l.sort()
    input = torch.FloatTensor(map_vals_l).to(device)
    input = input.view(1, input.shape[-1], 1)
    weight = model.encode_decode(input)
    print(weight)
    torch.save(weight, "/home/jitao/test/model/leo_model_weight.pt")
    print(1)
