import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors
import os
import sys
import time
from utils.preprocess import load_dp


class DrawRawData:
    # def __init__(self, src_data, scan_freq=30, deg_res=1, range=1,
    #              transparence=True, edgecolor=True, start_circle=1):
    def __init__(self, cfg):
        self.press_key = ""
        self.src_data = load_dp(cfg.src_data)
        self.scan_freq = cfg.freq
        self.rad_res = np.deg2rad(cfg.deg_res)
        self.pause = False
        self.play = True
        self.range = cfg.range
        self.x = None
        self.y = None
        self.c = None
        self.s = None
        self.mean_data_len = 0
        self.transparence = cfg.clarity
        self.edgecolor = cfg.edge_color
        self.start_circle = cfg.start_circle

    def on_key_press(self, event):

        if event != None:
            self.press_key = event.key

    def press_key_handle(self, one_circle, curr_yaw,
                         yaw_start, yaw_end, curr_index, last_yaw, color_map):

        if self.press_key != "":
            # print(self.press_key)

            if self.press_key == "escape":
                print("pressed esc and quit visualizer")
                self.play = False

            elif self.press_key == " ":
                if self.pause:
                    self.pause = False
                else:
                    self.pause = True
                print("pressed space and pause flag {}".format(self.pause))

            elif self.press_key == "left":
                curr_yaw -= 4 * self.rad_res
                self.rad_back(curr_index, curr_yaw)

                if curr_yaw < yaw_start:
                    curr_yaw = yaw_start

                    if curr_index > 1:
                        # self.rad_back(curr_yaw - self.rad_res)
                        curr_index -= 1
                        curr_yaw = yaw_end

                print("pressed left and current degree change {}"
                      .format(np.round(np.rad2deg(curr_yaw))))

            elif self.press_key == "right":
                last_yaw = curr_yaw
                curr_yaw += 2 * self.rad_res

                self.rad_front(one_circle, curr_yaw, last_yaw, color_map)
                if curr_yaw > yaw_end:
                    # curr_yaw = yaw_end
                    curr_index += 1
                    curr_yaw = yaw_start

                print("pressed right and current degree change {}"
                      .format(np.round(np.rad2deg(curr_yaw))))

            elif self.press_key == "up":
                self.rad_res += np.deg2rad(1)
                if self.rad_res > np.deg2rad(25):
                    self.rad_res = np.deg2rad(25)
                print("pressed up and rad res change {}"
                      .format(np.round(np.rad2deg(self.rad_res))))

            elif self.press_key == "down":
                self.rad_res -= np.deg2rad(1)
                if self.rad_res < np.deg2rad(1):
                    self.rad_res = np.deg2rad(1)
                print("pressed down and rad res change {}"
                      .format(np.round(np.rad2deg(self.rad_res))))

            elif self.press_key == "pageup":
                self.scan_freq += 1
                self.scan_freq = np.floor(self.scan_freq)
                if self.scan_freq > 50:
                    self.scan_freq = 50
                print("pressed pageup and freq change {}".format(self.scan_freq))

            elif self.press_key == "pagedown":
                self.scan_freq -= 1
                if self.scan_freq < 1:
                    self.scan_freq = 0.5
                print("pressed pagedown and freq change {}".format(self.scan_freq))

            elif self.press_key == "-":
                self.range -= 1
                if self.range < 1:
                    self.range = 1
                print("pressed - and range change {}".format(self.range))

            elif self.press_key == "+":
                self.range += 1
                if self.range > 9:
                    self.range = 9
                print("pressed + and range change {}".format(self.range))

            elif self.press_key == "r":
                self.range = 1
                self.scan_freq = 10
                self.rad_res = np.deg2rad(5)
                self.pause = False
                print("pressed reset and reset all")

            self.press_key = ""

        return curr_yaw, curr_index - 1

    def range_cutting(self, curr_index, curr_rad):

        if len(self.x) < self.range * self.mean_data_len:
            return

        end_index = curr_index - self.range
        if end_index > 0:

            curr_data = self.src_data[self.src_data["circle"] == curr_index]
            end_data = self.src_data[self.src_data["circle"] == end_index]

            curr_index_list = curr_data[curr_data["yaw"] <= curr_rad].index
            end_index_list = end_data[end_data["yaw"] <= curr_rad].index

            if len(curr_index_list) and len(end_index_list):
                keep_len = curr_index_list[-1] - end_index_list[-1]

                self.x = self.x[-keep_len:]
                self.y = self.y[-keep_len:]
                self.c = self.c[-keep_len:]
                self.s = self.s[-keep_len:]

    def rad_back(self, curr_index, curr_rad):

        if self.range == 1 or len(self.x) <= self.mean_data_len:
            back_index = np.flatnonzero(
                (self.x > curr_rad) & (self.x <= curr_rad + 4 * self.rad_res))

            if len(back_index):
                self.x = self.x[:-len(back_index)]
                self.y = self.y[:-len(back_index)]
                self.c = self.c[:-len(back_index)]
                self.s = self.s[:-len(back_index)]

        else:
            if curr_rad < np.deg2rad(-60):
                back_index = np.flatnonzero(
                    (self.x <= np.deg2rad(60)) & (self.x >= np.deg2rad(-30)))
            else:
                back_index = np.flatnonzero(self.x <= curr_rad)

            if len(back_index):
                end_index = back_index[-1]

                self.x = self.x[:end_index + 1]
                self.y = self.y[:end_index + 1]
                self.c = self.c[:end_index + 1]
                self.s = self.s[:end_index + 1]

    def rad_front(self, one_circle, curr_rad, last_yaw, color_map):

        self.data_get(one_circle=one_circle,
                      curr_yaw=curr_rad,
                      last_yaw=last_yaw,
                      color_map=color_map)

    def data_get(self, one_circle, curr_yaw, last_yaw, color_map):

        this_mask = (one_circle[:, 0] <= curr_yaw) & (one_circle[:, 0] > last_yaw)
        # rad_queue.append(one_circle[:, 0][this_mask])

        if self.x is None:
            self.x = one_circle[this_mask][:, 0]
            self.y = one_circle[this_mask][:, 1]
            if self.transparence:
                self.c = np.concatenate(
                    (color_map[one_circle[this_mask][:, 6].astype(np.int)],
                     one_circle[this_mask][:, 7].reshape(-1, 1)),
                    axis=1)
            else:
                self.c = color_map[one_circle[this_mask][:, 6].astype(np.int)]
            self.s = one_circle[this_mask][:, 7]
        else:

            self.x = np.concatenate((self.x, one_circle[this_mask][:, 0]))
            self.y = np.concatenate((self.y, one_circle[this_mask][:, 1]))

            if self.transparence:
                temp = np.concatenate(
                    (color_map[one_circle[this_mask][:, 6].astype(np.int)],
                     one_circle[this_mask][:, 7].reshape(-1, 1)),
                    axis=1)
                self.c = np.concatenate((self.c, temp), axis=0)
            else:
                self.c = np.concatenate(
                    (self.c, color_map[one_circle[this_mask][:, 6].astype(np.int)]),
                    axis=0)
            self.s = np.concatenate((self.s, one_circle[this_mask][:, 7]))

    def draw_once(self, curr_idnex, curr_rad, ax, x, y, c=None, s=None):

        plt.cla()

        ax.set_thetamin(-70)
        ax.set_thetamax(70)
        ax.set_rlim(0, 80000)
        ax.set_rticks([20000, 40000, 60000, 80000])
        ax.set_theta_zero_location(loc="E", offset=90)

        if self.edgecolor:
            ax.scatter(x=x, y=y, c=c, s=s * 50, edgecolors=1 - c[:, :3])
        else:
            ax.scatter(x=x, y=y, c=c, s=s * 50)
        ax.plot([curr_rad, curr_rad], [0, 80000],
                color="g", linewidth=3, linestyle="dashed")
        ax.set_title("scan circle " + str(curr_idnex), y=0.82)

    def run(self):

        max_dist = self.src_data["dist"].max()
        self.src_data["confidence"] = self.src_data["confidence"] * 3
        self.src_data["confidence"][self.src_data["confidence"] > 1] = 1

        group_data = self.src_data.groupby(["circle"])
        circle_index = group_data.count().index

        circle_nums = len(circle_index)
        self.mean_data_len = int(len(self.src_data) / circle_nums)

        print("total circles {}".format(circle_nums))
        print("mean data len {}".format(self.mean_data_len))

        np.random.seed(0)
        color_map = np.random.rand(circle_nums, 3)

        circle_list = []

        for index, index_name in enumerate(circle_index):
            one_circle = group_data.get_group((index_name))
            circle_list.append(one_circle.to_numpy())

        fig = plt.figure()
        fig.canvas.mpl_connect("key_press_event", self.on_key_press)
        ax = plt.axes([-0.15, -0.15, 1.3, 1.3], polar=True)
        # ax = plt.axes(polar=True)

        yaw_start, yaw_end = np.deg2rad(-60), np.deg2rad(60.0)
        index = self.start_circle - 1

        while index < circle_nums:
            one_circle = circle_list[index]

            # yaw_start, yaw_end = np.min(one_circle[:, 0]), np.max(one_circle[:, 0])
            # sample_interval = np.deg2rad(120) / (one_circle[-1, 3] - one_circle[0, 3]) / self.scan_freq

            curr_yaw = yaw_start + self.rad_res
            last_yaw = yaw_start
            while last_yaw <= yaw_end:

                if not self.pause:
                    self.data_get(one_circle=one_circle,
                                  curr_yaw=curr_yaw,
                                  last_yaw=last_yaw,
                                  color_map=color_map)

                curr_yaw, temp_index = self.press_key_handle(one_circle,
                                                             curr_yaw,
                                                             yaw_start,
                                                             yaw_end,
                                                             index + 1,
                                                             last_yaw,
                                                             color_map)

                if not self.play:
                    return

                if temp_index != index:
                    if temp_index >= circle_nums:
                        return
                    else:
                        index = temp_index
                        one_circle = circle_list[index]

                self.range_cutting(index + 1, curr_yaw)
                self.draw_once(curr_idnex=index + 1, curr_rad=curr_yaw, ax=ax,
                               x=self.x, y=self.y, c=self.c, s=self.s)

                if not self.pause:
                    last_yaw = curr_yaw
                    curr_yaw += self.rad_res

                plt.pause(1.0 / self.scan_freq)

            index += 1
