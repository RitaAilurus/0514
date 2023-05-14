import numpy as np
import torch
import uuid


WHITE = np.array((255, 255, 255), dtype=np.uint8)
RED = np.array((0, 0, 255), dtype=np.uint8)
GREEN = np.array((0, 255, 0), dtype=np.uint8)
BLUE = np.array((255, 0, 0), dtype=np.uint8)
BLACK = np.array((0, 0, 0), dtype=np.uint8)

class MapSample(object):
    def __init__(self, grid_map, start, goal, path, device=None):
        super(MapSample, self).__init__()
        if device is None:
            self._device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        else:
            self._device = device
        self.grid_map = torch.tensor(grid_map, dtype=torch.float32, device=self._device)
        self.start = torch.tensor(start, dtype=torch.long, device=self._device)
        self.goal = torch.tensor(goal, dtype=torch.long, device=self._device)
        self.path = torch.tensor(np.array(path), dtype=torch.long, device=self._device)

    def to(self, device):
        self._device = device
        self.grid_map = self.grid_map.to(device)
        self.start = self.start.to(device)
        self.goal = self.goal.to(device)
        self.path = self.path.to(device)

    def save(self, path=None):
        self.to('cpu')
        if path is None:
            path = str(uuid.uuid4()) + '.pt'
        torch.save(self, path)

    @staticmethod
    def load(path):
        try:
            sample = torch.load(path)
        except IOError as e:
            print(e)
            sample = None
        return sample

    def bgr_map(self, start_color=RED, goal_color=BLUE, path_color=GREEN):
        grid_map_np, start_np, goal_np, path_np = self.numpy()
        return MapSample.get_bgr_map(grid_map_np, start_np, goal_np, path_np, start_color, goal_color, path_color)

    def numpy(self):
        return self.grid_map.cpu().detach().numpy(), self.start.cpu().detach().numpy(), self.goal.cpu().detach().numpy(), self.path.cpu().detach().numpy()

    @staticmethod
    def get_bgr_map(grid_map, start, goal, path, start_color=RED, goal_color=BLUE, path_color=GREEN,
                    remove_first_path=True):
        map_np = np.array(grid_map)
        h, w = map_np.shape
        if remove_first_path:
            path = path[1:]
        if type(path) == list or type(path) == tuple:
            path = np.array(path)
        bgr_map = np.zeros((h, w, 3), dtype=np.uint8)
        idx = np.argwhere(map_np > 0).reshape(-1, 2)
        bgr_map[map_np == 0] = WHITE
        bgr_map[idx[:, 0], idx[:, 1]] = BLACK
        if np.any(path):
             bgr_map[path[:, 0], path[:, 1]] = path_color
        bgr_map[start[0], start[1]] = start_color
        bgr_map[goal[0], goal[1]] = goal_color
        return bgr_map

    @staticmethod
    def from_map(map_np, start, goal, path, resolution=1, device=None):
        grid_map = np.zeros((map_np.shape[0] * resolution, map_np.shape[1] * resolution), dtype=np.float32)
        for i in range(map_np.shape[0]):
            for j in range(map_np.shape[1]):
                if map_np[i, j] > 0:
                    grid_map[i*resolution:(i+1)*resolution, j*resolution:(j+1)*resolution] = 1
        return MapSample(grid_map, (start[0]*resolution, start[1]*resolution), (goal[0]*resolution, goal[1]*resolution), [(p[0]*resolution, p[1]*resolution) for p in path], device=device)

if __name__ == '__main__':
    import cv2
    sample = MapSample.load('grid_dataset/validation/4efae7fd-f67e-4c87-87af-09137102f6e5.pt')
    color_map = sample.bgr_map()
    cv2.imshow('map', cv2.resize(color_map, (1000, 1000)))
    cv2.waitKey(0)