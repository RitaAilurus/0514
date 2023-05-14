import torch
from torch import nn

from dataset.map_sample_2 import MapSample
from img_processing.gaussian_kernel import get_gaussian
from model.pe import GaussianRelativePE
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

class PPCNet(nn.Module):
    """ Path Planning CNN Network """    
    def __init__(self, n_layers=3, gaussian_blur_kernel=0):
        super(PPCNet, self).__init__()
        
        class _conv_block(nn.Module):
            def __init__(self, in_channels, out_channels, activation=None, norm_first=False, transpose=False, last_output_padding=0):
                super(_conv_block, self).__init__()
                self.activation = activation if activation is not None else nn.ReLU()
                self.bn1 = nn.BatchNorm2d(out_channels)
                self.bn2 = nn.BatchNorm2d(out_channels)
                if transpose:               
                    self.bn3 = nn.BatchNorm2d(out_channels // 2)     
                    self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, 3)
                    self.conv2 = nn.ConvTranspose2d(out_channels, out_channels, 3)
                    self.conv3 = nn.ConvTranspose2d(out_channels, out_channels // 2, 3, stride=2, padding=2, output_padding=last_output_padding)
                else:
                    self.bn3 = nn.BatchNorm2d(out_channels * 2)
                    self.conv1 = nn.Conv2d(in_channels, out_channels, 3)
                    self.conv2 = nn.Conv2d(out_channels, out_channels, 3)
                    self.conv3 = nn.Conv2d(out_channels, out_channels * 2, 3, stride=2)
                if norm_first:
                    self._fw = nn.Sequential(
                        self.conv1, self.bn1, self.activation, 
                        self.conv2, self.bn2, self.activation,
                        self.conv3, self.bn3, self.activation)
                else:
                    self._fw = nn.Sequential(
                        self.conv1, self.activation, self.bn1, nn.Dropout(p=0.5),
                        self.conv2, self.activation, self.bn2, nn.Dropout(p=0.5),
                        self.conv3, self.activation, self.bn3, nn.Dropout(p=0.5)
                    )
                    # self._fw = nn.Sequential(
                    #     self.conv1, self.activation, self.bn1,
                    #     self.conv2, self.activation, self.bn2,
                    #     self.conv3, self.activation, self.bn3)
            def forward(self, x):
                return self._fw(x)
        
        self.gaussian_blur_kernel= gaussian_blur_kernel
        if gaussian_blur_kernel > 0:
            gaussian_kernel = torch.tensor(get_gaussian(gaussian_blur_kernel, sigma=0, normalized=True), dtype=torch.float32).view(1, 1, gaussian_blur_kernel, gaussian_blur_kernel)
            self.blur = nn.Conv2d(1, 1, gaussian_blur_kernel, padding=gaussian_blur_kernel // 2, bias=False)
            self.blur.weight.data = gaussian_kernel
        else:
            self.blur = None
        self.pe = GaussianRelativePE(100)
        self.sigm = nn.Sigmoid()
        n_channels = [64 * (2 ** i) for i in range(n_layers)]
        self.conv_down = nn.ModuleList([_conv_block(c if i > 0 else 3, c) for i, c in enumerate(n_channels)])
        self.conv_up = nn.ModuleList([_conv_block(2 * c, 2 * c, transpose=True, last_output_padding=1 if i == len(n_channels) - 1 else 0) for i, c in enumerate(n_channels[::-1])])
        self.bottleneck = nn.Conv2d(64, 1, 3, padding=1)
        self.conv_out = nn.Conv2d(1, 1, kernel_size=3, padding=1)

    def pe_forward(self, x, start, goal):
        zeros = torch.zeros_like(x, device=x.device, dtype=x.dtype)
        pe_start = self.pe(zeros, start)
        pe_goal = self.pe(zeros, goal)
        # print("pe_start:", pe_start)
        # print("pe_goal:", pe_goal)
        # pe_start_np = pe_start.numpy()
        # pe_goal_np = pe_goal.numpy()
        # np.save('pe_start.npy', pe_start_np)
        # np.save('pe_goal.npy', pe_goal_np)
        return torch.cat([x, pe_start, pe_goal], dim=1)



    def forward(self, x, start, goal):
        """Forward pass

        Args:
            x (Tensor): (N, C, H, W) batch of 2d maps.
            start (Tensor): (N, 2) start position.
            goal (Tensor)): (N, 2) goal position.

        Returns:
            Tensor: score map. Score of a pixel should be proportional to the probability of belonging to the shortest path.
        """
        if self.gaussian_blur_kernel > 0:
            with torch.no_grad():
                x = self.blur(x)
        skip_conn = []
        x = self.pe_forward(x, start, goal)
        for i, conv in enumerate(self.conv_down):
            x = conv(x)
            if i < len(self.conv_down) - 1:
                skip_conn.append(x.detach())
        for i, conv in enumerate(self.conv_up):
            x = conv(x)
            if i < len(skip_conn):
                x = x + skip_conn[-1 - i]
        x = self.bottleneck(x)
        # x = self.pe_forward(x, start, goal)        
        x = self.conv_out(x)
        x = self.sigm(x)
        return x


if __name__ == '__main__':
    model = PPCNet(3)
    path = 'val/1.pt'
    sample = MapSample.load(path)
    import cv2
    map = sample.bgr_map()
    grid_map_resized = F.interpolate(sample.grid_map.unsqueeze(0).unsqueeze(0), size=(100, 100), mode='bilinear',align_corners=True)
    # map_with_pe = model.pe_forward(sample.grid_map.unsqueeze(0).unsqueeze(0), sample.start.unsqueeze(0).long(),sample.goal.unsqueeze(0).long())

    # map_with_pe_np =map_with_pe.numpy()
    # np.save('map_with_pe.npy', map_with_pe_np)
    # pe_map = map_with_pe[0, 0].detach().cpu().numpy()
    # cv2.imshow('map with position encoding', cv2.resize(pe_map, (600, 600)))
    cv2.imshow('map', cv2.resize(map, (600, 500)))
    cv2.waitKey(0)

    out = model(grid_map_resized, sample.start.unsqueeze(0).long(), sample.goal.unsqueeze(0).long())
    # out = model(sample.grid_map.unsqueeze(0).unsqueeze(0), sample.start.unsqueeze(0).long(), sample.goal.unsqueeze(0).long())
    out.sum().backward()

# 这段代码定义了一个名为PPCNet的PyTorch神经网络模型，用于在2D地图上进行路径规划。
# 该模型以一批2D地图、起始位置和目标位置作为输入，并输出一个得分图，其中像素的得分与属于最短路径的概率成比例。
# PPCNet模型由多个卷积块组成，每个卷积块包含三个卷积层，后跟批归一化和激活函数。
# 这些卷积层用于提取地图中的特征。批归一化层用于规范化每个卷积层的输出，以加速训练并提高模型的泛化能力。
# 激活函数用于引入非线性性，以使模型能够学习更复杂的函数。
# 该模型还包括一个高斯相对位置编码层，根据起始和目标位置向输入地图添加位置信息。
# 这个编码层的目的是为了让模型能够更好地理解地图中的位置信息，从而更好地进行路径规划。
# 模型使用卷积块之间的跳跃连接来改善训练期间的梯度流。
# 这些跳跃连接允许梯度能够更容易地流经整个模型，从而加速训练并提高模型的性能。
# 模型的最终输出是卷积层的sigmoid激活。
# 这个激活函数将输出的值映射到0到1之间的范围内，表示每个像素属于最短路径的概率。
# 提供的代码还包括一个主函数，该函数从文件中加载样本地图，将PPCNet模型应用于地图，并使用OpenCV显示结果得分图。
# 最后，代码计算输出张量的总和并执行反向传递以计算梯度。