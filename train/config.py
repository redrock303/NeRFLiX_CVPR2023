from easydict import EasyDict as edict


class Config:
    model_version = 'nerflix_cvpr2023'
    # dataset
    DATASET = edict()
    
    
    DATASET.PATCH_WIDTH = 128
    DATASET.PATCH_HEIGHT = 128
    DATASET.NFRAME = 3
    DATASET.SEED = 0

    # dataloader
    DATALOADER = edict()
    DATALOADER.IMG_PER_GPU = 4
    DATALOADER.NUM_WORKERS = 4

    # model
    MODEL = edict()

    MODEL.DEVICE = 'cuda'


    # solver
    SOLVER = edict()
    SOLVER.OPTIMIZER = 'Adam'
    SOLVER.BASE_LR = 5e-4
    SOLVER.WARM_UP_FACTOR = 0.1
    SOLVER.WARM_UP_ITER = 2000
    SOLVER.MAX_ITER = 300000
    SOLVER.WEIGHT_DECAY = 0
    SOLVER.MOMENTUM = 0
    SOLVER.BIAS_WEIGHT = 0.0

    SOLVER.ROOT_PATH = '/newdata/kunzhou/project/rendering/NeRFLiX_package/train/logs'
    # initialization

    INIT_MODEL = None

    # log and save
    LOG_PERIOD = 10
    SAVE_PERIOD = 20000

    # validation
    VAL = edict()
    VAL.PERIOD = 50
    VAL.IMG_PER_GPU = 1
    VAL.NUM_WORKERS = 1
    VAL.MAX_NUM = 5
    VAL.SAVE_IMG = True
    VAL.TO_Y = True


config = Config()



