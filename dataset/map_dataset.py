from torch.utils.data import Dataset
import os
from random import shuffle
from dataset.map_sample_2 import MapSample
from dataset.utils import collect_files
import cv2

MAP_DS_PATH = 'grid_dataset/train'


class MapDataset(Dataset):
    def __init__(self, datapath, lazy=True):
        super(MapDataset, self).__init__()
        datapath = os.path.abspath(datapath)
        self.samples = list(collect_files(datapath))
        shuffle(self.samples)
        self._lazy = lazy
        if not lazy:
            self.samples = [MapSample.load(sample) for sample in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self._lazy:
            return MapSample.load(self.samples[idx]), self.samples[idx]
        else:
            return self.samples[idx], self.samples[idx]

# if __name__ == '__main__':
#     import cv2
#     abs_path = os.path.abspath('grid_dataset/train')
#     for file in os.listdir(abs_path):
#         if os.path.isfile(os.path.join(abs_path, file)):
#             sample = MapSample.load(os.path.join(abs_path, file))
#             map = cv2.resize(sample.bgr_map(), (500, 500))
#             cv2.imshow('sample', map)
#             import numpy as np
#             import torch
#             sample.path = torch.tensor([])
#             map = cv2.resize(sample.bgr_map(), (500, 500))
#             cv2.imshow('sample2', map)
#             cv2.waitKey(0)
if __name__ == '__main__':
    import cv2
    import matplotlib.pyplot as plt
    import torch
    abs_path = os.path.abspath('grid_dataset/train')
    for file in os.listdir(abs_path):
        if os.path.isfile(os.path.join(abs_path, file)):
            sample = MapSample.load(os.path.join(abs_path, file))
            map1 = cv2.resize(sample.bgr_map(), (600, 600))
            fig, (ax1, ax2) = plt.subplots(1, 2)
            ax1.imshow(map1)
            sample.path = torch.tensor([])
            map2 = cv2.resize(sample.bgr_map(), (600, 600))
            ax2.imshow(map2)
            plt.show()