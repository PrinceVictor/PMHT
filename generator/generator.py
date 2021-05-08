# import matplotlib as mpl
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt

import numpy as np
import math
import argparse

from utils.rotate import direction_cosine_matrix3x3 as rotate3x3
from utils.motion_model import cv_state_trans_matrix as cv_model
from utils.logger import setup_logger
from config import gen_cfg as cfg


class SimulationGenerator:
    def __init__(self, cfg=None):
        assert cfg is not None, 'parameter cfg is none!'
        self.scene_params = cfg.SCENE
        self.target_params = cfg.TARGET
        self.radar_params = cfg.RADAR
        self.false_alarm_params = cfg.FALSE_ALARM

    def total_data_obtain(self, target_nums=None, total_times=None, lamda_para=None):
        target_state = self.target_state_generator(target_nums, total_times)
        noises = self.noise_generator(lamda_para, total_times)

        sim_dim = self.scene_params.dimension

        total_data = []
        assert len(target_state) == len(noises), 'length of target state is not equal with noise'
        for index in range(len(target_state)):
            if index == 0:
                curr_data = target_state[index][:, 0:sim_dim*3:3, :]
            else:
                curr_data = np.concatenate((target_state[index][:, 0:sim_dim*3:3, :],
                                            noises[index][:, 0:sim_dim, :]), axis=0)
            total_data.append(curr_data)

        return target_state, noises, total_data

    def target_state_generator(self, target_nums=None, total_times=None):

        if target_nums is None:
            target_nums = self.target_params.nums
        if total_times is None:
            total_times = self.target_params.time

        total_state_time_seq_nums = int(total_times * self.target_params.frequency)
        target_init_state = self.target_init_generator(target_nums)
        n, d, _ = target_init_state.shape

        target_state = np.zeros(shape=(n, d, 1), dtype=np.float)
        target_state_list = [target_init_state]

        for state_idx in range(1, total_state_time_seq_nums):
            for target_idx in range(n):
                target_state[target_idx] = \
                    cv_model(1 / self.target_params.frequency, state_dims=3, space_dims=3) @\
                    target_state_list[state_idx - 1][target_idx]
            target_state_list.append(target_state)

        return target_state_list

    def target_init_generator(self, target_nums):
        # initialize speed direction YAW ROLL PITCH
        init_speed_direction = np.random.uniform(low=0, high=np.pi * 2,
                                                 size=(target_nums, 3))
        init_speed = np.random.uniform(low=0, high=self.target_params.start_speed,
                                       size=(target_nums, 1))

        init_target_direction = np.random.uniform(low=np.deg2rad(self.radar_params.min_degree + 15),
                                                  high=np.deg2rad(self.radar_params.max_degree - 15),
                                                  size=(target_nums, 3))
        init_target_dist = np.random.uniform(low=self.radar_params.min_dist * 2,
                                             high=self.radar_params.max_dist,
                                             size=(target_nums, 1))

        # print("init speed direct \n{}".format(init_speed_direction))
        # print("init speed \n{}".format(init_speed))

        body_init_pos = np.zeros(shape=(3, 1), dtype=np.float)
        body_init_speed = np.zeros(shape=(3, 1), dtype=np.float)
        world_init_acc = np.zeros(shape=(3, 1), dtype=np.float)

        target_init_state = np.zeros(shape=(target_nums, 9, 1), dtype=np.float)

        for nums_index in range(target_nums):

            body_init_speed[0] = init_speed[nums_index]
            body_init_pos[0] = init_target_dist[nums_index]
            # print("init nums: {} previous speed \n{}".format(nums_index, body_init_speed))

            if self.scene_params.dimension == 2:
                speed_rotate_matrix = rotate3x3(**{"yaw": -init_speed_direction[nums_index][-1]})
                pos_rotate_matrix = rotate3x3(**{"yaw": -init_target_direction[nums_index][-1]})

            elif self.scene_params.dimension == 3:
                speed_rotate_matrix = rotate3x3(**{"yaw": -init_speed_direction[nums_index][-1],
                                                   "pitch": -init_speed_direction[nums_index][-2],
                                                   "roll": -init_speed_direction[nums_index][-3]})
                pos_rotate_matrix = rotate3x3(**{"yaw": -init_target_direction[nums_index][-1],
                                                 "pitch": -init_target_direction[nums_index][-2],
                                                 "roll": -init_target_direction[nums_index][-3]})

            world_init_speed = speed_rotate_matrix @ body_init_speed
            world_init_pos = pos_rotate_matrix @ body_init_pos

            for dims in range(3):
                target_init_state[nums_index, dims * 3] = world_init_pos[dims]
                target_init_state[nums_index, dims * 3 + 1] = world_init_speed[dims]
                target_init_state[nums_index, dims * 3 + 2] = world_init_acc[dims]

        # print("target init state:\n{}".format(target_init_state))
        return target_init_state

    def noise_generator(self, lambda_para=None, total_times=None):

        if lambda_para is None:
            lambda_para = self.false_alarm_params.expect_x
        if total_times is None:
            total_times = self.target_params.time

        total_state_time_seq_nums = int(total_times * self.target_params.frequency)
        total_noise_nums = np.random.poisson(lam=lambda_para,
                                             size=total_state_time_seq_nums)
        total_noise_list = []
        for index in range(total_state_time_seq_nums):
            curr_noise_x = np.random.uniform(low=self.scene_params.area_x[0],
                                             high=self.scene_params.area_x[1],
                                             size=(total_noise_nums[index], 1))
            curr_noise_y = np.random.uniform(low=self.scene_params.area_y[0],
                                             high=self.scene_params.area_y[1],
                                             size=(total_noise_nums[index], 1))
            curr_noise_z = np.random.uniform(low=self.scene_params.area_z[0],
                                             high=self.scene_params.area_z[1],
                                             size=(total_noise_nums[index], 1))
            curr_noise = np.stack((curr_noise_x, curr_noise_y, curr_noise_z), axis=1)
            total_noise_list.append(curr_noise)

        return total_noise_list




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='data generator')
    parser.add_argument('--config-file', type=str, default="config/gen.yaml",
                        help='source file path')

    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.freeze()

    if cfg.LOG_CONFIG.log:
        LOG = setup_logger(__file__, cfg.LOG_CONFIG.time, "./log/")
    else:
        LOG = setup_logger(__file__)

    for key, value in vars(args).items():
        LOG.info(str(key) + ': ' + str(value))

    LOG.info("Running with config:\n{}".format(cfg))
    # target_state_generator(cfg)
