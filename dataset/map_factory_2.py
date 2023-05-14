import numpy as np
import cv2
from random import choice

try:
    from planning.cpp_dstar_lite import DStarLite
except ImportError as e:
    print(e)
    print('Cannot import D* Lite implementation in c++. Let\'s try with the python one.')
    from planning.dstar_lite import DStarLite

from dataset.map_sample_2 import MapSample


class GridFactory(object):
    MAZE_COMBINATIONS = (
        (7, 0.06),
        (7, 0.07),
        (7, 0.08),
        (7, 0.09),
        (5, 0.10),
        (5, 0.11),
        (5, 0.12),
        (5, 0.13),
        (5, 0.14),
        # (3, 0.35),
        # (3, 0.36),
        # (3, 0.37),
        # (3, 0.38),
        # (3, 0.39),
        # (3, 0.40),
    )


    @staticmethod
    def random_grid_map(h=100, w=100, k=7, alpha=0.06):
        maze = np.random.rand(h, w)
        maze = np.where(maze > alpha, False, True)
        return maze


    @classmethod
    def random_grid_map_(cls, h=100, w=100):
        k, alpha = choice(cls.MAZE_COMBINATIONS)
        if k == 1:
            return GridFactory.random_grid_map(h, w, k, alpha)
        else:
            return GridFactory.random_grid_map(h, w, k, alpha)

    @staticmethod  # 随机生成点对（做起终点），比较最小距离，若循环超出则error
    def random_start_goal(maze, min_dist):
            h, w = maze.shape
            max_it = h * w
            start, goal = np.array([[0], [0]], dtype=np.int64), np.array([[0], [0]], dtype=np.int)
            i = 0
            while np.linalg.norm(start - goal) < min_dist and i < max_it:
                x = np.random.randint(0, h, size=(2, 1), dtype=np.int64)  # rows
                y = np.random.randint(0, w, size=(2, 1), dtype=np.int64)  # cols
                start = np.concatenate((x[0], y[0]))
                goal = np.concatenate((x[1], y[1]))
                if maze[start[0], start[1]] > 0:
                    start = goal
                i += 1
            if i == max_it:
                raise ValueError('Cannot find start and goal with enough distance')
            return start, goal

    @staticmethod
    # 这段代码定义了一个静态方法 solve，
    # 它接受一个迷宫 maze，起点 start，终点 goal，最大迭代次数 max_it，障碍物边缘 obst_margin 和终点边缘 goal_margin。
    # 将迷宫转换为浮点数类型，并将所有障碍物的值设置为无穷大。
        # 然后，它使用 DStarLite 算法来解决迷宫，并返回一个布尔值 solved 和一个路径 path。
        #  如果找不到解决方案，则会引发 ValueError。如果路径长度大于1，则 solved 为 True。
    def solve(maze, start, goal, max_it, obst_margin, goal_margin):
            maze = maze.astype(np.float64)
            maze[maze > 0] = np.inf
            solved, path = False, []
            try:
                ds = DStarLite(maze, goal[0], goal[1], start[0], start[1], max_it, False, obst_margin, goal_margin)
                next_step = None
                while next_step is not None or not path:
                    next_step = ds.step()
                    path.append(next_step)
                if path and path[-1] is None:
                    path.pop()
                if len(path) > 1:
                    solved = True
            except ValueError:
                if len(path) <= 1:
                    print("Unfeasible instance")
                else:
                    # solved at the best
                    solved = True
            return solved, path


    # 修改后的 make_sample_ 方法
    @staticmethod
    def make_sample_(h, w, min_dist):
        map = GridFactory.random_grid_map_(h, w)
        start, goal = GridFactory.random_start_goal(map, min_dist)
        return map, start, goal


    # 修改后的 make_sample 方法
    @staticmethod
    def make_sample(h, w, min_dist, max_it, obst_margin, goal_margin):
        map = GridFactory.random_grid_map(h, w)
        start, goal = GridFactory.random_start_goal(map, min_dist)
        solved, path = GridFactory.solve(map, start, goal, max_it, obst_margin, goal_margin)
        if solved:
            sample = MapSample(map, start, goal, path)
        return solved, sample


if __name__ == '__main__':  # 测试，调用make_sample生成一个迷宫样本，并显示
    solved, sample = False, None
    while not solved:
        solved, sample = GridFactory.make_sample(100, 100, 70, 10000, 0, 1)
    bgr = sample.bgr_map()
    cv2.imshow('map', cv2.resize(bgr, (1000, 1000)))
    cv2.waitKey(0)
