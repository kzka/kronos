"""Default config variables."""

import os.path as osp

from yacs.config import CfgNode as CN

# ============================================== #
# Beginning of config file
# ============================================== #
_C = CN()

# ============================================== #
# Directories
# ============================================== #
_C.DIRS = CN()

_C.DIRS.DIR = osp.dirname(osp.realpath(__file__))

_C.DIRS.LOG_DIR = "/tmp/kronos/"

_C.DIRS.PENN_ACTION_DIR = osp.join(_C.DIRS.DIR, "data/Penn_Action/")
_C.DIRS.MIME_DIR = osp.join(_C.DIRS.DIR, "data/mime_processed/")
_C.DIRS.DEMO_DIR = osp.join(_C.DIRS.DIR, "data/reptile/")
_C.DIRS.EMBODIED_DIR = osp.join(_C.DIRS.DIR, "data/embodied/")
_C.DIRS.EMBODIED_PENS_IN_CUP_DIR = osp.join(_C.DIRS.EMBODIED_DIR, "embodied_pens_in_two_cups/")
_C.DIRS.EMBODIED_FLIP_BOOK_DIR = osp.join(_C.DIRS.EMBODIED_DIR, "embodied_flip_book/")
_C.DIRS.EMBODIED_CUP_IN_PLATE_DIR = osp.join(_C.DIRS.EMBODIED_DIR, "embodied_cup_in_plate/")
_C.DIRS.EMBODIED_GLASSES_DIR = osp.join(_C.DIRS.EMBODIED_DIR, "embodied_glasses/")
_C.DIRS.CHAMFER_DIR = osp.join(_C.DIRS.DIR, "data/beta/")
_C.DIRS.MAGICAL_DIR = "/home/kevin/Desktop/all_but_gripper"
# "/home/kevin/Desktop/pngdir"  # "/home/kevin/repos/magical_early"

# ============================================== #
# Experiment params
# ============================================== #
# seed for python, numpy and pytorch.
# set this to `None` if you do not want
# to seed the results.
_C.SEED = 1

# opt level for mixed precision training
# using NVIDIA Apex.
# can be one of [0, 1, 2].
_C.FP16_OPT = 0

# can be one of: ['tcc', 'sal', 'svtcn]
_C.TRAINING_ALGO = "tcc"

_C.BATCH_SIZE = 4
_C.TRAIN_MAX_ITERS = 20_000
_C.DATASET = "magical"

# which action classes to select for creating
# the dataset.
# leave it empty to load all action classes.
_C.ACTION_CLASS = []

# ============================================== #
# Frame sampling params
# ============================================== #
_C.SAMPLING = CN()

_C.SAMPLING.PRETRAIN_BATCH_SAMPLER = "random"  # "same_action"
_C.SAMPLING.DOWNSTREAM_BATCH_SAMPLER = "same_action_downstream"

# the widlcard pattern for the video frames
_C.SAMPLING.IMAGE_EXT = "*.png"

## TCC

# can be one of:
# 'all', 'stride', 'offset_uniform', 'svtcn'
_C.SAMPLING.STRATEGY = "offset_uniform"

_C.SAMPLING.CONTEXT_STRIDE = 3
_C.SAMPLING.STRIDE_ALL_SAMPLER = 1
_C.SAMPLING.STRIDE_STRIDED_SAMPLER = 3
_C.SAMPLING.USE_OFFSET_STRIDED_SAMPLER = True
_C.SAMPLING.RANDOM_OFFSET_OFFSET_UNIFORM_SAMPLER = 1
_C.SAMPLING.NUM_CONTEXT_FRAMES = 1
_C.SAMPLING.NUM_FRAMES_PER_SEQUENCE = 15

## TCN

## SAL

# the authors of SAL control the ratio of positive
# examples `p` per mini-batch, where positive means
# not shuffled.
# This value is set to 25% (section 5, page 10).
# here, we control the percentage of shuffled
# or negative examples, which is just `1 - p`.
_C.SAMPLING.SHUFFLE_FRACTION = 0.75

# ============================================== #
# Data augmentation params
# ============================================== #
_C.AUGMENTATION = CN()

# MIME:         (240, 400)
# Penn Action:  (224, 224)
# Embodied:     (216, 384)
_C.IMAGE_SIZE = (224, 224)

_C.AUGMENTATION.TRAIN = [
    # can be one of ['center_crop', 'random_resized_crop', 'global_resize']
    "global_resize",
    "horizontal_flip",
    # "vertical_flip",
    "color_jitter",
    "rotate",
    # "normalize",
]

_C.AUGMENTATION.EVAL = [
    # can be one of ['center_crop', 'random_resized_crop', 'global_resize']
    "global_resize",
    # "normalize",
]

# ============================================== #
# Evaluator params
# ============================================== #
_C.EVAL = CN()

# how many iterations of the validation
# data loader to run
_C.EVAL.VAL_ITERS = 10

# downstream task evaluators
_C.EVAL.DOWNSTREAM_TASK_EVALUATORS = [
    # "linear_probe",
    "reward_visualizer",
    "kendalls_tau",
    # 'phase_alignment_topk',
    # 'cycle_consistency',
    "nn_visualizer",
]

# what distance metric to use in the
# embedding space.
# can be one of ['cosine', 'sqeuclidean']
_C.EVAL.DISTANCE = "sqeuclidean"

# Kendall's Tau
_C.EVAL.KENDALLS_TAU = CN()
_C.EVAL.KENDALLS_TAU.STRIDE = 1

# Reward visualizer
_C.EVAL.REWARD_VISUALIZER = CN()
_C.EVAL.REWARD_VISUALIZER.L2_NORMALIZE = False

# Cycle consistency
_C.EVAL.CYCLE_CONSISTENCY = CN()
# can be one of:
# ['two_way', 'three_way', 'both']
_C.EVAL.CYCLE_CONSISTENCY.MODE = "both"
_C.EVAL.CYCLE_CONSISTENCY.STRIDE = 1

# Phase alignment topk
_C.EVAL.PHASE_ALIGNMENT_TOPK = CN()
_C.EVAL.PHASE_ALIGNMENT_TOPK.TOPK = [1, 5, 10]

# Nearest neighbour visualizer
_C.EVAL.NEAREST_NEIGHBOUR_VISUALIZER = CN()
_C.EVAL.NEAREST_NEIGHBOUR_VISUALIZER.NUM_FRAMES = 5

# Linear probe
_C.EVAL.LINEAR_PROBE = CN()
_C.EVAL.LINEAR_PROBE.NUM_FRAMES_PER_SEQ = 20

# ============================================== #
# Model params
# ============================================== #
_C.MODEL = CN()

## Feature extraction network
_C.MODEL.FEATURIZER = CN()

# can be one of:
# 'resnet18', 'resnet34', 'resnet50', 'resnet101'
_C.MODEL.FEATURIZER.NETWORK_TYPE = "resnet18"

# whether to use pretrained ImageNet weights
_C.MODEL.FEATURIZER.PRETRAINED = True

# this variable controls how we want proceed with fine-tuning the base model.
# 'frozen': weights are fixed and batch_norm stats are also fixed.
# 'all': everything is trained and batch norm stats are updated.
# 'bn_only': only tune batch_norm variables and update batch norm stats.
_C.MODEL.FEATURIZER.TRAIN_BASE = "bn_only"

# this variable controls how the batch norm layers
# apply normalization. If set to False, the mean and
# variance are calculated from the batch.
# If set to True, batch norm layers will normalize
# using the mean and variance of its moving (i.e. running)
# statistics, learned during training.
_C.MODEL.FEATURIZER.BN_USE_RUNNING_STATS = False

# these variables specify where to place the output head
# of the base model.
# Inception networks use a string.
# Resnet networks use an integer.
_C.MODEL.FEATURIZER.OUT_LAYER_NAME = "Mixed_5d"
_C.MODEL.FEATURIZER.OUT_LAYER_IDX = 7

## Embedding network
_C.MODEL.EMBEDDER = CN()

_C.MODEL.EMBEDDER.EMBEDDING_SIZE = 32
_C.MODEL.EMBEDDER.NUM_CONV_LAYERS = 2
_C.MODEL.EMBEDDER.NUM_CHANNELS = 256
_C.MODEL.EMBEDDER.NUM_FC_LAYERS = 2
_C.MODEL.EMBEDDER.WIDEN_FACTOR = 2
_C.MODEL.EMBEDDER.FEATURIZER_DROPOUT_RATE = 0.0
_C.MODEL.EMBEDDER.FEATURIZER_DROPOUT_SPATIAL = False
_C.MODEL.EMBEDDER.FC_DROPOUT_RATE = 0.1
_C.MODEL.EMBEDDER.USE_BN = True

# can be one of: ['avg_pool', 'max_pool', 'spatial_softmax']
_C.MODEL.EMBEDDER.FLATTEN_METHOD = "max_pool"

## Classifier network (for SAL)
_C.MODEL.CLASSIFIER = CN()

_C.MODEL.CLASSIFIER.NUM_FC_LAYERS = 3
_C.MODEL.CLASSIFIER.HIDDEN_DIMS = [128, 64]
_C.MODEL.CLASSIFIER.FC_DROPOUT_RATE = 0.0

# ============================================== #
# Loss params
# ============================================== #
_C.LOSS = CN()

_C.LOSS.STOCHASTIC_MATCHING = False

# tcc: ['regression_mse_var']
# tcn: ['npairs', 'triplet_semihard']
_C.LOSS.LOSS_TYPE = "regression_mse_var"

_C.LOSS.CYCLE_LENGTH = 2
_C.LOSS.LABEL_SMOOTHING = 0.1
_C.LOSS.SOFTMAX_TEMPERATURE = 0.1
_C.LOSS.NORMALIZE_INDICES = True
_C.LOSS.L2_NORMALIZE_EMBEDDINGS = False
_C.LOSS.VARIANCE_LAMBDA = 0.001
_C.LOSS.HUBER_DELTA = 0.1
_C.LOSS.SIMILARITY_TYPE = "l2"  # cosine

_C.LOSS.POS_RADIUS = 6  # 0.2 seconds * 29fps ~ 6 timesteps
_C.LOSS.NEG_RADIUS = 12  # 2 * pos_radius
_C.LOSS.MARGIN = 1.0

# [1.0, 0.99, 0.9, 0.5, 0.1, 0.01, 0.0]
_C.LOSS.WEIGHT_TCC = 0.9
_C.LOSS.MARGIN_HINGE = 0.3

# ============================================== #
# Optimizer params
# ============================================== #
_C.OPTIMIZER = CN()

# supported optimizers are: adam, sgd
_C.OPTIMIZER.TYPE = "adam"
_C.OPTIMIZER.MOMENTUM = 0.9

# featurizer-specific optimization params
_C.OPTIMIZER.FEATURIZER_LR = 1e-4
_C.OPTIMIZER.FEATURIZER_L2_REG = 0

# encoder-specific optimization params
_C.OPTIMIZER.ENCODER_LR = 1e-4
_C.OPTIMIZER.ENCODER_L2_REG = 1e-5

# auxiliary-specific optimization params
# e.g. classification head in SAL
_C.OPTIMIZER.AUXILIARY_LR = _C.OPTIMIZER.ENCODER_LR
_C.OPTIMIZER.AUXILIARY_L2_REG = _C.OPTIMIZER.ENCODER_L2_REG

# ============================================== #
# Logging params
# ============================================== #
_C.LOGGING = CN()

# number of steps between summary logging
_C.LOGGING.REPORT_INTERVAL = 100

# number of steps between eval logging
_C.LOGGING.EVAL_INTERVAL = 2_000

# ============================================== #
# Checkpointing params
# ============================================== #
_C.CHECKPOINT = CN()

# number of steps between consecutive checkpoints
_C.CHECKPOINT.SAVE_INTERVAL = 1_000

# ============================================== #
# End of config file
# ============================================== #
CONFIG = _C
