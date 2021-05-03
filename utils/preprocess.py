import pandas as pd
import numpy as np
import glob
import os
import sys
import time
import argparse
import json

def cvtDP_Txt2CSV(source_dir, output_dir, save_csv = False, save_json=True):
    raw_data = pd.read_csv(os.path.join(source_dir, "raw.txt"), header=None, index_col=None)
    raw_csv = format_dp_data(raw_data)
    # dp_data.sort_values(by=["frame", "yaw"], inplace=True, ascending=True)
    raw_csv.sort_values(by=["circle", "yaw"], inplace=True, ascending=True)
    raw_csv.reset_index(inplace=True, drop=True)

    raw_csv["yaw"] = np.deg2rad(raw_csv["yaw"])
    raw_csv["pitch"] = np.deg2rad(raw_csv["pitch"])

    raw_csv["time"] = raw_csv["time"] - raw_csv["time"].iloc[0]
    raw_csv["frame"] = raw_csv["frame"] - raw_csv["frame"].iloc[0]

    print(raw_csv.iloc[0:2])

    if save_csv:
        # data_output_dir = output_dir+"source/"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        raw_csv.to_csv(os.path.join(output_dir, "raw.csv"), index=0)

    if save_json:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        raw_json = raw_csv.to_json(orient="records", indent=1)
        with open(os.path.join(output_dir, "raw.json"), 'w', encoding='utf-8') as json_file:
            json_file.write(raw_json)

def load_dp(src_file):

    src_data = pd.read_csv(src_file, index_col=None, header=0)
    # print(src_data[0:20])
    print("loading data length {}".format(len(src_data)))

    return src_data

def format_dp_data(input, to_numpy=True):
    columns = ["yaw",
               "dist",
               "dist_gate",
               "time",
               "pitch",
               "frame",
               "circle",
               "confidence"]

    if to_numpy:

        input = input.to_numpy()
        _, columns_size = input.shape
        if columns_size > 8:
            input = input[:, :8]

    return pd.DataFrame(data=input, columns=columns)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='raw data preprocess')
    parser.add_argument('--src-data', type=str, default="data/raw/narrow_data",
                        help='source file path')
    parser.add_argument('--output-data', type=str, default="data/preprocessed",
                        help='source file path')
    args = parser.parse_args()

    print("this is preprocess!!!")

    # source_dir = "source_data/"
    # output_dir = "result/"
    source_dir = args.src_data
    dir_name = source_dir.split("/")[-1]
    output_dir = os.path.join(args.output_data, dir_name)

    if os.path.isdir(output_dir) is not True:
        print("make new dir {}".format(output_dir))
        os.makedirs(output_dir)

    cvtDP_Txt2CSV(source_dir, output_dir, save_csv=True, save_json=False)