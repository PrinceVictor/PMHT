import numpy as np
import argparse

from pmht.pmht import PMHT
from generator.generator import SimulationGenerator
from visualizer.simulate_data_vis import DrawSimTarget

from utils.logger import setup_logger
from config import gen_cfg as cfg


parser = argparse.ArgumentParser(description='PMHT runing')
parser.add_argument('--config-file', type=str, default="param/gen.yaml",
                    help='source file path')
parser.add_argument('--log', action='store_true', default=False,
                    help='log the message')
args = parser.parse_args()

def main(cfg, LOG):
    
    PMHT_batch_T = 3

    sim_gen = SimulationGenerator(cfg=cfg)
    
    target_state, noises, total_data = \
        sim_gen.total_data_obtain(batch_T=PMHT_batch_T)
    noise_expected = sim_gen.noise_expected()

    print(noise_expected)

    LOG.info(f"total times {len(total_data)}")

    pmht_mananger = PMHT(times=len(total_data),
                         batch_T=PMHT_batch_T, 
                         noise_expected=noise_expected,
                         sample_T=cfg.RADAR.period)

    pmht_mananger.pmht_init(target_state[0])
    for t_idx in range(PMHT_batch_T, len(total_data)):
        pmht_mananger.run(t_idx, total_data[t_idx])
    
    track_info = pmht_mananger.get_track_info()

    draw = DrawSimTarget(cfg=cfg)
    draw.run_pmht(total_data, target_state, track_info)
        

if __name__ == '__main__':
    print(args)

    cfg.merge_from_file(args.config_file)
    cfg.freeze()

    if args.log:
        LOG = setup_logger(__file__, True, "./log/")
    else:
        LOG = setup_logger(__file__)
    
    main(cfg, LOG)
    



    
        

