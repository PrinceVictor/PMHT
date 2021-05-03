import numpy as np
import argparse

import visualizer.raw_data_vis as raw_vis
import visualizer.simulate_data_vis as sim_vis

from utils.logger import setup_logger
from config import vis_cfg as cfg

parser = argparse.ArgumentParser(description='visualizer tool')
parser.add_argument('--config-file', type=str, default="param/vis.yaml",
                    help='source file path')
parser.add_argument('--src-data', type=str, default="data/preprocessed/narrow_data/raw.csv",
                    help='source file path')
parser.add_argument('--deg-res', type=int, default=1,
                    help='scan resolution, press up and down to adjust')
parser.add_argument('--freq', type=int, default=30,
                    help='scan frequency, press PgUp and PgDn to adjust')
parser.add_argument('--range', type=int, default=1,
                    help='keep circle range, press - and + to adjust')
parser.add_argument('--clarity', action='store_true', default=True,
                    help='scatter point\'s transparence')
parser.add_argument('--edgecolor', action='store_true', default=False,
                    help='scatter point\'s edgecolor')
parser.add_argument('--start-circle', type=int, default=1,
                    help='start circle')

args = parser.parse_args()

if __name__ == "__main__":

    cfg.merge_from_file(args.config_file)
    cfg.freeze()

    if cfg.LOG_CONFIG.log:
        LOG = setup_logger(__file__, cfg.LOG_CONFIG.time, "./log/")
    else:
        LOG = setup_logger(__file__)

    LOG.info("This is visualizer running!")
    for key, value in vars(args).items():
        LOG.info(str(key) + ': ' + str(value))

    if cfg.RAW_VIS.ENABLE:
        raw_vis.DrawRawData(cfg=cfg.RAW_VIS).run()

    if cfg.SIM_VIS.ENABLE:
        LOG.info("SIM VISUALIZER")
        sim_vis.DrawSimTarget(cfg=cfg.SIM_VIS).run()

