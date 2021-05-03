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
        target_state = self.sim_gen.target_state_generator()
        """
        @target_state: time_index, nums, state
        """
        print(len(target_state))

        # self.visualizer_2d(target_state)


    def visualizer_2d(self, state):

        fig = plt.figure()
        ax = plt.axes()

        for index, curr_state in enumerate(state):
            if self.scene_dimension == 2:
                self.draw_once(curr_state, ax)

            plt.pause(1.0 / self.vis_freq)
            
            # raise SystemExit

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

