from yacs.config import CfgNode as CN

_DATA = CN()  
_DATA.HOME = "../simulation_data/"

_DATA.TRIAL_START = 150
_DATA.TRIAL_END = 1000
_DATA.IMAGE_RANGE = 100
_DATA.G_VALUES = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
_DATA.NUM_SPOTS = 20
_DATA.GT_JSON = "../simulation_data/brownian_traj_gt/3d_traj_brownian_20spots.json"
_DATA.GT_STRIDE = 8


def get_default_data_config():
    return _DATA.clone() 


_C = CN()

_C.EXPR_NAME = "EXPR"
_C.LOG_DIR = "experiments/"

# TRAIN or EVAL.
_C.TYPE = "TRAIN"

# Choose between:
# MK1 - PIN based on plain CNN
# MK2 - PIN based on ResNet
# MK3 - PIN based on FPN
_C.MODEL = "MK1"

# Choose PSF among Double-Helix, Born-Wolf, and Astigmatism
_C.PSF = "Double-Helix"

# Training configurations
_C.TRAIN = CN()
_C.TRAIN.START_EPOCH = 1
_C.TRAIN.EPOCH = 11
_C.TRAIN.BATCH_SIZE = 64
_C.TRAIN.NUM_WORKERS = 4
_C.TRAIN.PRINT_FREQ = 50
_C.TRAIN.EVAL_FREQ = 1000
_C.TRAIN.LOSS_CLIP_VALUE = 200.
_C.TRAIN.LR = CN()
_C.TRAIN.LR.MOMENTUM = 0.9
_C.TRAIN.LR.WEIGHT_DECAY = 0.1
_C.TRAIN.LR.BASE_LR = 0.001
_C.TRAIN.LR.STEP_SIZE = 4
_C.TRAIN.DATA = get_default_data_config()
_C.TRAIN.VALID_DATA = get_default_data_config()
_C.TRAIN.VALID_DATA.TRIAL_START = 100
_C.TRAIN.VALID_DATA.TRIAL_END = 150
_C.TRAIN.CONTINUE_FROM = ""

_C.EVAL = CN()
_C.EVAL.DATA = get_default_data_config()
_C.EVAL.DATA.TRIAL_START = 0
_C.EVAL.DATA.TRIAL_END = 100
_C.EVAL.THRESHOLD = 0.7
_C.EVAL.BATCH_SIZE = 100
_C.EVAL.NUM_WORKERS = 4
_C.EVAL.PRINT_FREQ = 50
_C.EVAL.RESTORE_FROM = ""

_C.REAL_WORLD = CN()
_C.REAL_WORLD.DATA = get_default_data_config()
_C.REAL_WORLD.DATA.TRIAL_START = 0
_C.REAL_WORLD.DATA.TRIAL_END = 10

def get_default_detection_config():
    return _C.clone()
