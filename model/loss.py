# import torch
from torch import nn
from torch.nn import MSELoss
import torch


class MSEMapLoss(nn.Module):
    def __init__(self):
        super(MSEMapLoss, self).__init__()
        self._loss = MSELoss()

    def forward(self, x, y):
        return self._loss(x, y)

# from torch import nn
# from torch.nn import MSELoss
# import torch
#
#
# class MSEMapLoss(nn.Module):
#     def __init__(self):
#         super(MSEMapLoss, self).__init__()
#         self._loss = MSELoss()
#         self._regularization = nn.L2Loss()
#
#     def forward(self, x, y):
#         mse_loss = self._loss(x, y)
#         reg_loss = self._regularization(self.parameters())
#         return mse_loss + reg_loss
# 这段代码定义了一个名为MSEMapLoss的类，它继承自nn.Module。
# 在类的构造函数中，它调用了父类的构造函数，并初始化了一个MSELoss对象。
# 在forward函数中，它将输入x和y传递给MSELoss对象，并返回结果。
# MSEMapLoss类的作用是计算输入x和y之间的均方误差（MSE）损失。
