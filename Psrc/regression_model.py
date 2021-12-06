import torch
import torch.nn as nn


class NNRegressionModel(nn.Module):
    def __init__(self, input_s, hiden_s, output_s=1) -> None:
        super().__init__()
        self.input = nn.Linear(input_s, hiden_s)
        self.ac_f = nn.ReLU()
        self.hiden = nn.Linear(hiden_s, output_s)

    def init_weight(self, weight):
        self.input.weight.data = weight[:100].view(-1, 1)
        self.input.bias.data = weight[100:200].view(-1)
        self.hiden.weight.data = weight[200:300].view(1, -1)
        self.hiden.bias.data = weight[300:301].view(-1)

    def forward(self, x):
        x = self.input(x)
        x = self.ac_f(x)
        x = self.hiden(x)
        return x


if __name__ == "__main__":
    model = NNRegressionModel(1, 100, 1)
    for i in model.named_parameters():
        print(i[0], i[1])
    # print([x for x in list(model.parameters())])

    # weight = torch.ones([301])
    # print(weight.shape)

    # model.init_weight(weight)
    # print([x for x in list(model.parameters())])
    # for i in model.parameters():
    #     print(i.shape)
    print(1)
