import numpy as np
import matplotlib.pyplot as plt

from src.generator.generator import SimulationGenerator
from src.config import gen_cfg


class DrawSimTarget:
    def __init__(self, cfg=None):
        gen_cfg.merge_from_file(cfg.yaml_file)
        gen_cfg.freeze()
        self.gen_cfg = gen_cfg
        self.scene_dimension = self.gen_cfg.SCENE.dimension
        self.scene_area_x = self.gen_cfg.SCENE.area_x
        self.scene_area_y = self.gen_cfg.SCENE.area_y
        self.scene_area_z = self.gen_cfg.SCENE.area_z
        self.vis_freq = self.gen_cfg.TARGET.frequency

        self.sim_gen = SimulationGenerator(cfg=gen_cfg)

    def run(self):
        target_state, noises, total_data = self.sim_gen.total_data_vis_obtain()

        self.total_vis_2d(total_data)


    def total_vis_2d(self, data):

        fig = plt.figure()
        ax = plt.axes()
        self.draw_all(data, ax=ax)

    def draw_all(self, data, ax):
        ax.set_xlim(self.scene_area_x[0], self.scene_area_x[1])
        ax.set_ylim(self.scene_area_y[0], self.scene_area_y[1])

        data = np.concatenate(data, axis=0)

        ax.scatter(x=data[:, 0, :], y=data[:, 1, :], s=0.5)
        plt.show()

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

