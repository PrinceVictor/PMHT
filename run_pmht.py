import numpy as np
import argparse

from generator.generator import SimulationGenerator

from utils.logger import setup_logger
from config import gen_cfg as cfg


parser = argparse.ArgumentParser(description='PMHT runing')
parser.add_argument('--config-file', type=str, default="param/gen.yaml",
                    help='source file path')
parser.add_argument('--log', action='store_true', default=False,
                    help='log the message')
args = parser.parse_args()

if __name__ == '__main__':
    print(args)

    cfg.merge_from_file(args.config_file)
    cfg.freeze()

    if args.log:
        LOG = setup_logger(__file__, True, "./log/")
    else:
        LOG = setup_logger(__file__)
    
    sim_gen = SimulationGenerator(cfg=cfg)
    sim_gen.target_state_generator()

    
        

