import torch


class Checkpoint(object):   #checkpoint用于每回合训练后保存和加载模型状态
    def __init__(self, model, epoch, min_loss, optimizer, scheduler=None, path=None):
        self.data = {
            'model': model.state_dict(),
            'epoch': epoch,
            'min_loss': min_loss,
            'optimizer': optimizer.state_dict()
        }
        if scheduler is not None:
            self.data['scheduler'] = scheduler.state_dict()
        if not path:
            path = "checkpoints/checkpoint_{}.pt".format(epoch)
        self.path = path

    # init函数接受一个模型对象model、一个epoch数、一个最小损失minloss、一个优化器对象optimizer
    # 一个调度器对象scheduler（可选）和一个路径path（可选）作为参数。
    # 它将这些参数保存在一个名为data的字典中，并将路径保存在self.path中。
    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def _save(self, path=None):#将checkpoint对象保存到磁盘
        if not path:
            path = self.path
        torch.save(self, path)
    
    @staticmethod
    def load_checkpoint(path, model, optimizer=None, scheduler=None, device=None):#从磁盘加载checkpoint并恢复模型状态
        if not device:
            device = next(model.parameters()).device
        cp = torch.load(path, map_location=device)
        model.load_state_dict(cp['model'])
        if optimizer is not None:
            optimizer.load_state_dict(cp['optimizer'])
        if scheduler is not None:
            scheduler.load_state_dict(cp['scheduler'])
        return cp

    @staticmethod
    def save(model, epoch, min_loss, optimizer, scheduler=None, path=None, **kwargs):
        cp = Checkpoint(model, epoch, min_loss, optimizer, scheduler, path)
        for k, v in kwargs.items():
            cp[k] = v
        cp._save()

    

    
if __name__ == '__main__':
    # Checkpoint.save(None, None, None, None, None, None, foo='qwerty', xxx=3)
    from model.ppcnet_2 import PPCNet
    from torch.optim import Adam
    model = PPCNet().to('cpu')
    optimizer = Adam(model.parameters())
    Checkpoint.load_checkpoint('checkpoints/checkpoint_5.pt', model,optimizer)

# 这段代码定义了一个名为Checkpoint的类，用于保存和加载模型的状态。
# 具体来说，它包含了一个初始化函数，用于创建一个Checkpoint对象并保存模型、优化器、调度器等状态信息；
# 一个save函数，用于将Checkpoint对象保存到磁盘上；
# 一个loadcheckpoint函数，用于从磁盘上加载Checkpoint对象并恢复模型、优化器、调度器等状态信息；
# 以及一个save函数，用于创建一个Checkpoint对象并将其保存到磁盘上。
# 这个类的作用是方便地保存和加载模型的状态，以便在训练过程中进行断点续训等操作。