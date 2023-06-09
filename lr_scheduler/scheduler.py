from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


class CosineAnnealingWarmRestartsDecay(CosineAnnealingWarmRestarts):
    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1, verbose=False, decay=0.995):
        self.T_cur = last_epoch
        super(CosineAnnealingWarmRestartsDecay, self).__init__(optimizer, T_0, T_mult, eta_min, last_epoch, verbose)
        self.decay = decay

    def step(self, epoch=None):
        if self.last_epoch > 0 and self.last_epoch % self.T_0 == 0:
            self.base_lrs = [lr * self.decay for lr in self.base_lrs]
        return super(CosineAnnealingWarmRestartsDecay, self).step(epoch)


if __name__ == '__main__':
    import torch
    from matplotlib import pyplot as plt
    import numpy as np

    N = 10000
    T_0 = 100
    param = torch.nn.Parameter(torch.rand(1,1))
    optim = torch.optim.SGD([param], lr=0.002)
    scheduler = CosineAnnealingWarmRestartsDecay(optim, T_0=T_0, decay=0.992,capturable=True)

    x = np.arange(N)
    y = [optim.param_groups[0]['lr']]
    for i in range(1, N):
        scheduler.step()
        y.append(optim.param_groups[0]['lr'])

    y = np.array(y)
    fig, axis = plt.subplots()
    axis.plot(x, y)
    plt.show()
    # 这个程序是用来展示如何使用PyTorch中的CosineAnnealingWarmRestarts调度器，并添加了一个名为CosineAnnealingWarmRestartsDecay的自定义调度器。
    # 这个自定义调度器在每个T0步骤之后将学习率乘以一个衰减因子decay。这个程序还使用了matplotlib库来绘制学习率随时间的变化曲线。