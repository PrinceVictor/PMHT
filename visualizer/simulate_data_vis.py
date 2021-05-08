import numpy as np
import matplotlib.pyplot as plt

from src.generator.generator import SimulationGenerator
from src.config import gen_cfg


class DrawSimTarget:
    def __init__(self, cfg=None):
        # gen_cfg.merge_from_file(cfg.yaml_file)
        # gen_cfg.freeze()
        self.gen_cfg = cfg
        self.scene_dimension = self.gen_cfg.SCENE.dimension
        self.scene_area_x = self.gen_cfg.SCENE.area_x
        self.scene_area_y = self.gen_cfg.SCENE.area_y
        self.scene_area_z = self.gen_cfg.SCENE.area_z
        self.vis_freq = self.gen_cfg.TARGET.frequency

        # self.sim_gen = SimulationGenerator(cfg=gen_cfg)

    def run(self):
        # target_state, noises, total_data = self.sim_gen.total_data_vis_obtain()

        self.total_vis_2d(total_data)
    
    def run_pmht(self, total_raw, gt, track=None):

        fig = plt.figure()
        ax = plt.axes()

        total_raw = np.concatenate(total_raw, axis=0)
        gt = np.concatenate(gt, axis=0)
        track = np.concatenate(track, axis=0)

        ax.scatter(x=total_raw[:, 0, :], y=total_raw[:, 1, :], c='blue', marker='o', s=0.5)
        ax.scatter(x=gt[:, 0, :], y=gt[:, 3, :], c='green', marker='^', s=10)
        ax.scatter(x=track[:, 0, :], y=track[:, 2, :], c='red', marker='x', s=10)

        plt.show()


    def total_vis_2d(self, data):

        fig = plt.figure()
        ax = plt.axes()
        self.draw_all(data, ax=ax)

    def draw_all(self, data, ax, c='red', marker='o', s=0.5):
        ax.set_xlim(self.scene_area_x[0], self.scene_area_x[1])
        ax.set_ylim(self.scene_area_y[0], self.scene_area_y[1])

        data = np.concatenate(data, axis=0)

        ax.scatter(x=data[:, 0, :], y=data[:, 1, :], s=s, marker=marker, c=c)
        ax.scatter(x=data[:, 0, :], y=data[:, 1, :], s=s, marker=marker, c=c)
        ax.scatter(x=data[:, 0, :], y=data[:, 1, :], s=s, marker=marker, c=c)
        

    def draw_once(self, data, ax):
        # plt.cla()
        ax.set_xlim(self.scene_area_x[0], self.scene_area_x[1])
        ax.set_ylim(self.scene_area_y[0], self.scene_area_y[1])
        # ax.set_xticks(self.scene_area_x)
        # print(self.scene_area_x[0], self.scene_area_x[1])
        # print(data.shape)
        # print(data)
        print(data[:1, 0, :], data[:1, 3, :])
        ax.scatter(x=data[:, 0, :], y=data[:, 3, :])
        # plt.show()
        # ax.set_xlim(self.scene_area_x[0])

