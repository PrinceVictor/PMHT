# my_project/generator_config.py
from yacs.config import CfgNode as CN

_C = CN()

_C.LOG_CONFIG = CN()
_C.LOG_CONFIG.log = False
_C.LOG_CONFIG.time = False

_C.SCENE = CN()

_C.SCENE.dimension = 2
_C.SCENE.coordinate = "ENU"
_C.SCENE.shape = "square"
_C.SCENE.length = 2e5
_C.SCENE.width = 2e5
_C.SCENE.height = 4e4

_C.SCENE.area_x = [-_C.SCENE.width/2, _C.SCENE.width/2]
_C.SCENE.area_y = [-_C.SCENE.length/2, _C.SCENE.length/2]
_C.SCENE.area_z = [0, _C.SCENE.height]

_C.RADAR = CN()

_C.RADAR.pos = [0, 0, 0]
_C.RADAR.min_degree = -60
_C.RADAR.max_degree = 60
_C.RADAR.period = 5
_C.RADAR.frequency = 1.0/_C.RADAR.period
_C.RADAR.max_dist = 8e4
_C.RADAR.min_dist = 0.3e4

_C.TARGET = CN()

_C.TARGET.start_pos_mode = "fix"
_C.TARGET.state = "CV"
_C.TARGET.time = 1*60
_C.TARGET.frequency =  1.0/_C.RADAR.period
_C.TARGET.nums = 1
_C.TARGET.start_speed = 2000*1.85/3.6
# area sequence x y z
_C.TARGET.area = [[-0.5e5+_C.RADAR.pos[0], 0.5e5+_C.RADAR.pos[0]],
                  [0.2e4+_C.RADAR.pos[1], 1e5+_C.RADAR.pos[1]],
                  [0.0e5+_C.RADAR.pos[2], _C.SCENE.height]]

_C.FALSE_ALARM = CN()
_C.FALSE_ALARM.expect_x = 10

def get_cfg_defaults():

  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern

  return _C.clone()