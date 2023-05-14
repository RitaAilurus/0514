import numpy as np

def get_gaussian(k=3, sigma=0, normalized=True):
    if sigma == 0:
        sigma = k / 5
    sigma_square = sigma ** 2
    coord_x = np.stack([np.arange(-(k // 2), k // 2 + 1) for _ in range(k)])
    coord_y = coord_x.T
    alpha = 1 / (2 * np.pi * sigma_square)
    out = alpha * np.exp(- 1 / (2 * sigma_square) * (coord_x ** 2  + coord_y ** 2))
    if normalized:
        out /= out.sum()
    return out

# 这个程序是用来生成高斯核的。高斯核是一种常用的图像处理滤波器，可以用于模糊、降噪、边缘检测等操作。
# 这个程序中的 get_gaussian 函数接受三个参数：k 表示核的大小，sigma 表示高斯分布的标准差，normalized 表示是否对核进行归一化处理。
# 函数内部使用了 numpy 库来生成高斯核。
# 如果 sigma 没有指定，则默认为 k/5。
# 如果 normalized 为 True，则对生成的核进行归一化处理。