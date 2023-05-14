import os
from random import random, choice
from dataset.map_sample import MapSample
from model.checkpoint import Checkpoint
from torch.nn.modules.loss import MSELoss, L1Loss
from dataset.map_dataset import MapDataset
from model.ppcnet import PPCNet
from dataset.utils import b_search, custom_collate_fn
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import cv2
import numpy as np
from threading import Thread
from time import time
from matplotlib import pyplot as plt
from lr_scheduler.scheduler import CosineAnnealingWarmRestartsDecay


TRAIN = 'map_dataset_test_of/train'
VALIDATION = 'map_dataset_test_of/validation'
EPOCHS = 100
SAVE_EVERY = 5
SHOW_EVERY = 100
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
RESULTS_PATH = 'model_results'
CHECKPOINT_PATH = 'checkpoints'
BATCH_SIZE = 48
LR = 1e-3

def save_model_result(map, model_out, start, goal, path):
    if not os.path.exists(os.path.abspath(RESULTS_PATH)):
        os.mkdir(os.path.abspath(RESULTS_PATH))
    sep = np.zeros((map.shape[-1], 50, 3), dtype=np.uint8)
    sep[:, 20:31] = np.array((0, 0, 0))
    map, model_out = map.squeeze(1), model_out.squeeze(1)
    for m, out, s, g, p in zip(map, model_out, start, goal, path):
        path_to_print = b_search(out, s, g, to_torch=True)
        if (path_to_print != 0).any():
            m = m.cpu().detach().numpy()
            model_solution = MapSample.get_bgr_map(m, s, g, path_to_print.numpy())
            gt_solution = MapSample.get_bgr_map(m, s, g, torch.nonzero(p > 0).cpu().numpy())
            # LEFT img: model solution ------ RIGHT img: dstar solution
            comparison = np.concatenate((model_solution, sep, gt_solution), axis=1)
            filename = os.path.join(os.path.abspath(RESULTS_PATH), str(int(time())) + '.png')
            cv2.imwrite(filename, cv2.cvtColor(cv2.resize(comparison, (1200, 600)), cv2.COLOR_BGR2RGB))

def test(model, dataloader, loss=None, device=DEVICE, show_every=5):
    avg_loss = 0.0 if loss is not None else float('inf')
    n = 0 if loss is not None else 1
    model.eval()
    with torch.no_grad():
        for i, (map, start, goal, path) in enumerate(dataloader):
            if start.dtype != torch.long:
                start, goal = start.long(), goal.long()
            if device != 'cpu':
                map, start, goal, path = map.to(device), start.to(device), goal.to(device), path.to(device)
            out = model(map, start, goal)
            if loss is not None:
                l = loss(out.squeeze(1), path)
                avg_loss += l.item()
                n += map.shape[0]
            if show_every > 0 and i % show_every == 0:
                Thread(target=save_model_result, args=(map, out, start, goal, path)).start()
    return avg_loss / n

def train(model, dataloader, loss, optimizer, epochs, device=DEVICE, save_every=10, 
        scheduler=None, min_loss=float('inf'), val_dataloader=None, start_epoch=0, checkpoint_path=None):
    cv2.startWindowThread()
    avg_loss = 0.
    eval_loss = min_loss
    eval_losses = []
    n = 0
    for epoch in range(start_epoch, epochs):
        model.train()
        for i, (map, start, goal, path) in enumerate(dataloader):
            if start.dtype != torch.long:
                start, goal = start.long(), goal.long()
            if device != 'cpu':
                map, start, goal, path = map.to(device), start.to(device), goal.to(device), path.to(device)
            optimizer.zero_grad()
            out = model(map, start, goal)
            l = loss(out.squeeze(1), path)
            l.backward()
            avg_loss += l.item()
            n += map.shape[0]
            optimizer.step()
            if scheduler is not None and type(scheduler) != torch.optim.lr_scheduler.ReduceLROnPlateau:
                scheduler.step()
        avg_loss /= n
        print("[Epoch {}]\n\tavg_loss: {}".format(epoch, avg_loss))
        if val_dataloader is not None:
            eval_loss = test(model, val_dataloader, loss, device, SHOW_EVERY)
            eval_losses.append(eval_loss)
            update_loss_plot(eval_losses)
            if type(scheduler) == torch.optim.lr_scheduler.ReduceLROnPlateau:
                scheduler.step(eval_loss)
            print("\tvalidation avg_loss: {}".format(eval_loss))
            if eval_loss < min_loss:
                print("\tNew min: {} (previous was {})".format(eval_loss, min_loss))
                min_loss = eval_loss
                Checkpoint.save(model, epoch, min_loss, optimizer, scheduler=scheduler, path=os.path.join(checkpoint_path, "checkpoint_new_min_{}.pt".format(epoch)))
        if epoch > 0 and epoch % save_every == 0:
            Checkpoint.save(model, epoch, min_loss, optimizer, scheduler=scheduler, path=os.path.join(checkpoint_path, "checkpoint_{}.pt".format(epoch)))
        avg_loss = 0.0
        n = 0
        visualize_results(RESULTS_PATH)

def update_loss_plot(eval_losses):
    plt.figure(1)
    plt.clf()
    plt.xlabel('Epochs')
    plt.title('Validation loss')
    plt.plot(np.arange(len(eval_losses)), np.array(eval_losses), color='b')            
    plt.pause(0.01)

def visualize_results(path):
    files = [os.path.join(os.path.abspath(path), f) for f in os.listdir(path)]
    if len(files) > 50:
        to_delete = files[:50]
        for file in to_delete:
            os.remove(file)
        files = files[50:]
    plt.figure(2)
    img = cv2.imread(choice(files))
    plt.clf()
    plt.imshow(img)

def main(epochs=EPOCHS, device=DEVICE):
    if not os.path.exists(os.path.abspath('checkpoints')):
        os.mkdir(os.path.abspath('checkpoints'))
    model = PPCNet(gaussian_blur_kernel=3).to(device)
    dataset = MapDataset(TRAIN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, collate_fn=custom_collate_fn, pin_memory=True)
    val_dataset = MapDataset(VALIDATION)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, collate_fn=custom_collate_fn, pin_memory=True)
    loss = L1Loss().to(device)
    optimizer = Adam(model.parameters(), lr=LR)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=200)
    scheduler = CosineAnnealingWarmRestartsDecay(optimizer, T_0=150, decay=0.995)
    train(model, dataloader, loss, optimizer, epochs, val_dataloader=val_dataloader, save_every=SAVE_EVERY, scheduler=scheduler, checkpoint_path=CHECKPOINT_PATH)
    Checkpoint.save(model, EPOCHS, None, optimizer, scheduler=None)

if __name__ == '__main__':
    plt.ion()
    main()

# 脚本定义了几个常量，包括训练和验证数据集的路径，训练的时期数，保存模型检查点和在训练期间显示结果的频率，用于训练的设备（如果可用，则为GPU，否则为CPU）
# 保存模型结果和检查点的路径，批量大小和学习率。
#
# 脚本还定义了几个函数，包括save_model_result，它保存给定地图、起始和目标位置以及路径的模型预测路径和地面真实路径；
# test，它在给定数据集上评估模型并返回平均损失（如果提供）并保存模型的预测路径和地面真实路径的子集；
# 和train，它使用给定的损失函数和优化器在给定的数据集上训练模型，训练给定数量的时期，具有可选的学习率调度和检查点，
# 并返回最终平均损失和在训练期间实现的最佳验证损失。
#
# 脚本的主体加载训练和验证数据集，初始化模型、损失函数、优化器和学习率调度器，并使用train函数训练模型。
# 在训练期间，定期调用test函数在验证数据集上评估模型并保存模型的预测路径和地面真实路径以进行可视化。