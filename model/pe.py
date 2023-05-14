import torch
from torch import nn
import math
from dataset.map_sample_2 import MapSample
import matplotlib.pyplot as plt
class PositionalEncoding(nn.Module):
 #用于给输入张量添加位置编码的模块。
    def __init__(self, max_h, max_w):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(max_w).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, max_h, 2) * (-math.log(10000.0) / max_w))
        pe = torch.zeros(1, 1, max_h, max_w)
        pe[:, :, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [N, C, H, W]
        """
        return x + self.pe[:, :, :x.size(2)]

        
class GaussianRelativePE(nn.Module): #用于给输入张量添加高斯相对位置编码。
    def __init__(self, side, sigma=None):
        super().__init__()
        if sigma is None:
            sigma = (side / 5)
        self.sigma_square = sigma ** 2
        self.alpha = 1 / (2 * math.pi * self.sigma_square)
        self.side = side
        coord_r = torch.stack([torch.arange(side) for _ in range(side)])
        coord_c = coord_r.T
        self.register_buffer('coord_r', coord_r)
        self.register_buffer('coord_c', coord_c)

    def forward(self, x, center):
        pe = self.alpha * torch.exp(- ((self.coord_r.view(1, self.side, self.side) - center[:, 0:1].unsqueeze(1)) ** 2 + \
            (self.coord_c.view(1, self.side, self.side) - center[:, 1:2].unsqueeze(1)) ** 2) / (2 * self.sigma_square))
        pe /= pe.amax(dim=(-1, -2)).view(-1, 1, 1)
        return x + pe.unsqueeze(1)


if __name__ == '__main__':
    path = 'val/1.pt'
    sample = MapSample.load(path)
    map = sample.grid_map
    start = sample.start
    goal = sample.goal
    # Define the GaussianRelativePE module
    grpe = GaussianRelativePE(map.shape[-1])

    # Perform relative positional encoding on the input tensor
    map = grpe(map, goal)

# 假设要展示第0张图像

# 这段代码定义了两个类：PositionalEncoding和GaussianRelativePE。
# PositionalEncoding是一个用于给输入张量添加位置编码的模块，它的实现方式是根据输入张量的宽度和高度生成一个位置编码张量，然后将其加到输入张量上。
# GaussianRelativePE是一个用于给输入张量添加高斯相对位置编码的模块，它的实现方式是根据输入张量的大小和目标位置生成一个高斯相对位置编码张量，
# 然后将其加到输入张量上。这段代码还包含了一个main函数，用于测试GaussianRelativePE模块的功能。
# 具体来说，它生成了一个大小为(side, side)的随机张量map和一个大小为(bsz, 2)的目标位置张量goal，
# 然后将它们作为输入调用GaussianRelativePE模块的forward函数，得到一个输出张量out。