# my_project/generator_config.py
from yacs.config import CfgNode as CN

_C = CN()

_C.LOG_CONFIG = CN()
_C.LOG_CONFIG.log = False
_C.LOG_CONFIG.time = False

_C.RAW_VIS = CN()
_C.RAW_VIS.ENABLE = False
_C.RAW_VIS.src_data = "data/preprocessed/narrow_data/raw.csv"
_C.RAW_VIS.deg_res = 1
_C.RAW_VIS.freq = 30
_C.RAW_VIS.range = 1
_C.RAW_VIS.clarity = True
_C.RAW_VIS.edge_color = False
_C.RAW_VIS.start_circle = 1

_C.SIM_VIS = CN()
_C.SIM_VIS.ENABLE = True

_C.SIM_VIS.yaml_file = "param/gen.yaml"
