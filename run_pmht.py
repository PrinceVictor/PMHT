import numpy as np
import argparse

from pmht.pmht import PMHT
from pmht.track import MOT
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
    
    PMHT_batch_T = 1

    sim_gen = SimulationGenerator(cfg=cfg)
    
    target_state, noises, total_data = \
        sim_gen.total_data_obtain(batch_T=PMHT_batch_T)
    noise_expected = sim_gen.noise_expected()

    LOG.info(f"total times {len(total_data)}")

    # pmht_mananger1 = PMHT(times=len(total_data),
    #                      batch_T=1, 
    #                      noise_expected=noise_expected,
    #                      sample_T=cfg.RADAR.period,
    #                      meas_sigma=cfg.TARGET.meas_sigma)

    # pmht_mananger1.pmht_init(target_state[0])
    # for t_idx in range(1, len(total_data)):
    #     pmht_mananger1.run(t_idx, total_data[t_idx])
    
    # track_info1 = pmht_mananger1.get_track_info()

    # draw1 = DrawSimTarget(cfg=cfg)
    # draw1.run_pmht(total_data, target_state, track_info1, 
    #                f'Target Nums={cfg.TARGET.nums} batch T=1')

    mot_track = MOT(times=len(total_data), 
                    delta_t=cfg.RADAR.period, keep_T=3)

    for t_id, data in enumerate(total_data):
        mot_track.run_track(t_id, data)

    raise SystemExit

    pmht_mananger2 = PMHT(times=len(total_data),
                         batch_T=3, 
                         noise_expected=noise_expected,
                         sample_T=cfg.RADAR.period,
                         meas_sigma=cfg.TARGET.meas_sigma)

    pmht_mananger2.pmht_init(target_state[0])
    for t_idx in range(3, len(total_data)):
        pmht_mananger2.run(t_idx, total_data[t_idx])
    
    track_info2 = pmht_mananger2.get_track_info()

    draw2 = DrawSimTarget(cfg=cfg)
    draw2.run_pmht(total_data, target_state, track_info2, 
                   f'Target Nums={cfg.TARGET.nums} batch T=3')
    
        

if __name__ == '__main__':
    print(args)

    cfg.merge_from_file(args.config_file)
    cfg.freeze()

    if args.log:
        LOG = setup_logger(__file__, True, "./log/")
    else:
        LOG = setup_logger(__file__)
    
    main(cfg, LOG)
    



    
        

